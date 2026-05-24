import json
import os
import re
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv(override=True)

_llm = None

DEFAULT_DISCLAIMER = (
    "This summary is for educational purposes only and is not medical advice. "
    "Always discuss your lab results with a qualified healthcare provider."
)

DEFAULT_GROQ_MODEL = "llama3-70b-8192"


def get_groq_api_key() -> str:
    return (
        os.getenv("GROQ_API_KEY_MEDICAL") or os.getenv("GROQ_API_KEY") or ""
    ).strip()


def validate_groq_api_key() -> str | None:
    key = get_groq_api_key()
    placeholders = (
        "your_key_here",
        "your_groq_key_here",
        "paste_your_groq_key_here",
        "paste_your_gemini_key_here",
    )
    if not key or key in placeholders:
        return (
            "**GROQ_API_KEY_MEDICAL is missing.** Edit `.env` and paste your key from "
            "[console.groq.com/keys](https://console.groq.com/keys) "
            "(starts with `gsk_`)."
        )
    if key.startswith("AIza"):
        return (
            "**Wrong key type:** that is a Gemini key (`AIza`). Use a Groq key (`gsk_...`) "
            "from [console.groq.com/keys](https://console.groq.com/keys)."
        )
    if not key.startswith("gsk_"):
        return (
            "**Invalid Groq API key format.** Keys usually start with **`gsk_`**. "
            "Get one at [console.groq.com/keys](https://console.groq.com/keys)."
        )
    return None


# Backward-compatible alias for app imports
validate_gemini_api_key = validate_groq_api_key


def reset_llm() -> None:
    global _llm
    _llm = None


def _message_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _retry_delay_seconds(error: Exception) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error), re.I)
    if match:
        return float(match.group(1)) + 2
    return float(os.getenv("GROQ_RETRY_DELAY", "10"))


def _is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "rate limit" in text or "quota" in text


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        err = validate_groq_api_key()
        if err:
            raise ValueError(err.replace("**", ""))
        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            temperature=0,
            api_key=get_groq_api_key(),
        )
    return _llm


def invoke_llm(prompt: str) -> str:
    """Call Groq with retries on rate-limit errors."""
    max_retries = int(os.getenv("GROQ_MAX_RETRIES", "3"))
    pause = float(os.getenv("GROQ_CALL_DELAY", "1"))

    last_error: Exception | None = None
    for attempt in range(max_retries):
        if attempt > 0 and pause > 0:
            time.sleep(pause)
        try:
            response = get_llm().invoke([HumanMessage(content=prompt)])
            return _message_text(response)
        except Exception as e:
            last_error = e
            if _is_quota_error(e) and attempt < max_retries - 1:
                time.sleep(_retry_delay_seconds(e))
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("LLM invoke failed without a captured error")


def parse_llm_json(content: str):
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def combined_analysis_agent(lab_data: list[dict], context: str) -> str:
    """Single API call for explanations, flags, and doctor questions."""
    prompt = f"""You are a medical report assistant for patients with no medical background.

Lab results: {lab_data}
Medical context: {context}

Produce a patient-friendly report as ONE JSON object with this exact structure:
{{
  "explanations": [{{"test": "...", "plain_english": "..."}}],
  "flags": [{{"test": "...", "severity": "Normal|Watch|High priority|Critical", "reason": "..."}}],
  "questions": [{{"question": "..."}}],
  "disclaimer": "one short safety sentence: not medical advice, see your doctor"
}}

Rules:
- One plain-English sentence per test in explanations.
- severity must be exactly: Normal, Watch, High priority, or Critical.
- Include 3-5 doctor questions.
- No diagnosis or treatment prescriptions.
Return ONLY valid JSON, no markdown fences or extra text."""

    return invoke_llm(prompt)


def explainer_agent(lab_data: list[dict], context: str) -> str:
    prompt = f"""You are a medical explainer for patients with no medical background.

Lab results: {lab_data}
Medical context: {context}

For each test, write ONE plain-English sentence explaining what it measures and whether the patient's value is normal.
Return JSON: [{{"test": "...", "plain_english": "..."}}]
Return ONLY valid JSON, no extra text."""
    return invoke_llm(prompt)


def flagging_agent(lab_data: list[dict], context: str) -> str:
    prompt = f"""You are a medical flagging system.

Lab results: {lab_data}
Medical context: {context}

Assign each test a severity: Normal, Watch, High priority, or Critical.
Return JSON: [{{"test": "...", "severity": "...", "reason": "..."}}]
Return ONLY valid JSON, no extra text."""
    return invoke_llm(prompt)


def research_agent(lab_data: list[dict], flags: str, context: str) -> str:
    prompt = f"""You are helping a patient prepare for their doctor's appointment.

Lab results: {lab_data}
Flagged values: {flags}
Medical context: {context}

Generate 3-5 specific, intelligent questions the patient should ask their doctor.
Return JSON: [{{"question": "..."}}]
Return ONLY valid JSON, no extra text."""
    return invoke_llm(prompt)


def safety_agent(explainer_out: str, flags_out: str, questions_out: str) -> str:
    prompt = f"""You are a medical safety reviewer.

Review this AI-generated patient report for diagnostic language or treatment advice:

Explanations: {explainer_out}
Flags: {flags_out}
Questions: {questions_out}

Return a one-sentence safety disclaimer only, no JSON."""
    return invoke_llm(prompt).strip()
