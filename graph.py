import json
import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents import (
    DEFAULT_DISCLAIMER,
    combined_analysis_agent,
    explainer_agent,
    flagging_agent,
    parse_llm_json,
    research_agent,
    safety_agent,
)
from rag_setup import get_vectorstore


class ReportState(TypedDict):
    lab_data: list[dict]
    rag_context: str
    explainer_output: str
    flags_output: str
    questions_output: str
    disclaimer: str
    final_report: dict


def retrieve_context(state: ReportState) -> ReportState:
    vs = get_vectorstore()
    tests = [item["test"] for item in state["lab_data"]]
    query = ", ".join(tests)
    docs = vs.similarity_search(query, k=4)
    context = "\n".join(d.page_content for d in docs)
    return {**state, "rag_context": context}


def run_combined_analysis(state: ReportState) -> ReportState:
    raw = combined_analysis_agent(state["lab_data"], state["rag_context"])
    data = parse_llm_json(raw)
    return {
        **state,
        "explainer_output": json.dumps(data.get("explanations", [])),
        "flags_output": json.dumps(data.get("flags", [])),
        "questions_output": json.dumps(data.get("questions", [])),
        "disclaimer": data.get("disclaimer") or DEFAULT_DISCLAIMER,
    }


def run_explainer(state: ReportState) -> ReportState:
    out = explainer_agent(state["lab_data"], state["rag_context"])
    return {**state, "explainer_output": out}


def run_flagging(state: ReportState) -> ReportState:
    out = flagging_agent(state["lab_data"], state["rag_context"])
    return {**state, "flags_output": out}


def run_research(state: ReportState) -> ReportState:
    out = research_agent(
        state["lab_data"], state["flags_output"], state["rag_context"]
    )
    return {**state, "questions_output": out}


def run_safety(state: ReportState) -> ReportState:
    disclaimer = safety_agent(
        state["explainer_output"],
        state["flags_output"],
        state["questions_output"],
    )
    return {**state, "disclaimer": disclaimer}


def compile_report(state: ReportState) -> ReportState:
    try:
        report = {
            "explanations": parse_llm_json(state["explainer_output"]),
            "flags": parse_llm_json(state["flags_output"]),
            "questions": parse_llm_json(state["questions_output"]),
            "disclaimer": state.get("disclaimer") or DEFAULT_DISCLAIMER,
        }
    except (json.JSONDecodeError, TypeError) as e:
        report = {"error": str(e), "raw": state}
    return {**state, "final_report": report}


def build_graph():
    g = StateGraph(ReportState)
    g.add_node("retrieve", retrieve_context)
    g.add_node("compile", compile_report)

    use_multi = os.getenv("GROQ_MULTI_AGENT", os.getenv("GEMINI_MULTI_AGENT", "")).lower() in ("1", "true", "yes")

    if use_multi:
        g.add_node("explainer", run_explainer)
        g.add_node("flagging", run_flagging)
        g.add_node("research", run_research)
        g.add_node("safety", run_safety)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "explainer")
        g.add_edge("explainer", "flagging")
        g.add_edge("flagging", "research")
        g.add_edge("research", "safety")
        g.add_edge("safety", "compile")
    else:
        g.add_node("analyze", run_combined_analysis)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "analyze")
        g.add_edge("analyze", "compile")

    g.add_edge("compile", END)
    return g.compile()
