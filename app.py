import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from agents import reset_llm, validate_groq_api_key  # noqa: E402
from graph import build_graph
from pdf_parser import extract_lab_values

reset_llm()

st.set_page_config(
    page_title="MediScan AI — Lab Report Simplifier",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg:         #FFF0F5;
        --surface:    #FFFFFF;
        --surface2:   #FDE8F0;
        --border:     rgba(254,129,212,0.18);
        --border-med: rgba(254,129,212,0.30);
        --heading:    #1a0f14;
        --body:       #4a2840;
        --muted:      #9b6080;
        --pink:       #FE81D4;
        --rose:       #F06FA0;
        --blush:      #FBC3C1;
        --peach:      #FFEABB;
        --green:      #16a34a;
        --green-lit:  #22c55e;
        --blue:       #2563eb;
        --blue-lit:   #3b82f6;
        --hover-glow: rgba(254,129,212,0.20);
    }

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        background-color: var(--bg) !important;
        color: var(--heading) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem 4rem !important; max-width: 1100px !important; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #f5c6dd 0%, #fad4e8 50%, #fde8c8 100%);
        border: 1px solid var(--border-med);
        border-radius: 20px;
        padding: 2.8rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 28px rgba(200,60,120,0.22);
        border: 1.5px solid rgba(254,129,212,0.45);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -40px;
        width: 240px; height: 240px;
        background: radial-gradient(circle, rgba(254,129,212,0.22) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -50px; left: 50px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(255,234,187,0.35) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(254,129,212,0.15);
        border: 1px solid rgba(254,129,212,0.45);
        color: #c0397a;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        padding: 3px 12px;
        border-radius: 20px;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0 0 0.5rem;
        background: linear-gradient(90deg, #1a0f14 0%, #c0397a 50%, #d4860a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 1rem;
        color: var(--muted);
        font-weight: 300;
        margin: 0;
        max-width: 600px;
    }

    /* ── Tip box ── */
    .tip-box {
        background: var(--surface);
        border: 1px solid var(--border-med);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        font-size: 0.8rem;
        color: var(--muted);
        line-height: 1.95;
        box-shadow: 0 2px 8px rgba(254,129,212,0.08);
    }

    /* ── Stats bar ── */
    .stats-bar {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1.4rem;
        flex-wrap: wrap;
    }
    .stat-pill {
        background: var(--surface);
        border: 1px solid var(--border-med);
        border-radius: 10px;
        padding: 0.45rem 1rem;
        font-size: 0.82rem;
        color: var(--muted);
        box-shadow: 0 1px 4px rgba(254,129,212,0.07);
    }
    .stat-pill strong { color: var(--heading); }

    /* ── Section headings ── */
    .section-head {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--heading);
        margin: 2.2rem 0 1rem;
        padding-bottom: 0.55rem;
        border-bottom: 1.5px solid var(--border-med);
    }
    .section-head .dot {
        width: 9px; height: 9px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Explanation cards ── */
    .explain-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--rose);
        border-radius: 12px;
        padding: 0.95rem 1.2rem;
        margin-bottom: 0.55rem;
        transition: box-shadow 0.2s, border-left-color 0.2s;
        box-shadow: 0 1px 6px rgba(254,129,212,0.06);
    }
    .explain-card:hover {
        box-shadow: 0 4px 16px var(--hover-glow);
        border-left-color: var(--pink);
    }
    .explain-test {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--rose);
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 0.28rem;
    }
    .explain-text {
        font-size: 0.91rem;
        color: var(--body);
        line-height: 1.6;
        margin: 0;
    }

    /* ── Flag rows ── */
    .flag-row {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 6px rgba(254,129,212,0.06);
        transition: box-shadow 0.2s;
    }
    .flag-row:hover { box-shadow: 0 4px 16px var(--hover-glow); }

    .severity-chip {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.9px;
        text-transform: uppercase;
        padding: 3px 11px;
        border-radius: 20px;
        white-space: nowrap;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .chip-normal   { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .chip-watch    { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
    .chip-high     { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
    .chip-critical { background: #fce7f3; color: #be185d; border: 1px solid #f9a8d4; }

    .flag-test {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--heading);
    }
    .flag-reason { font-size: 0.87rem; color: var(--muted); margin-top: 2px; }

    /* ── Question cards ── */
    .q-card {
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 6px rgba(254,129,212,0.06);
        transition: box-shadow 0.2s;
    }
    .q-card:hover { box-shadow: 0 4px 16px var(--hover-glow); }
    .q-num {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: var(--blue);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        width: 26px; height: 26px;
        border-radius: 7px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .q-text { font-size: 0.91rem; color: var(--body); line-height: 1.55; margin: 0; padding-top: 2px; }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        margin-top: 1.5rem;
        display: flex;
        gap: 0.7rem;
        align-items: flex-start;
    }
    .disclaimer-icon { font-size: 1.1rem; flex-shrink: 0; }
    .disclaimer-text { font-size: 0.84rem; color: #92400e; line-height: 1.55; }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, #e0559a 0%, #be185d 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.8rem !important;
        letter-spacing: 0.3px !important;
        transition: box-shadow 0.2s, opacity 0.2s !important;
        box-shadow: 0 2px 10px rgba(190,24,93,0.25) !important;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 20px rgba(190,24,93,0.35) !important;
        opacity: 0.92 !important;
    }
    .stButton > button:disabled { opacity: 0.4 !important; box-shadow: none !important; }

    .stDownloadButton > button {
        background: #eff6ff !important;
        color: var(--blue) !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 1.5px dashed rgba(254,129,212,0.4) !important;
        border-radius: 14px !important;
        padding: 0.5rem !important;
    }
    div[data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border-med) !important;
        border-radius: 12px !important;
    }
    .stSuccess {
        background: #f0fdf4 !important;
        border: 1px solid #86efac !important;
        border-radius: 10px !important;
        color: #15803d !important;
    }
    .stError {
        background: #fff1f2 !important;
        border: 1px solid #fecdd3 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stSpinner"] p { color: var(--muted) !important; font-size: 0.88rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">🧬 AI-Powered · Groq LLM · LangGraph</div>
        <div class="hero-title">MediScan AI</div>
        <p class="hero-sub">Upload your lab report PDF and get a plain-English breakdown — what each value means, what needs attention, and what to ask your doctor.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── API key validation ────────────────────────────────────────────────────────
key_error = validate_groq_api_key()
if key_error:
    st.error(key_error)
    st.info("After updating `.env`, refresh this page.")

# ── Upload row ────────────────────────────────────────────────────────────────
col_upload, col_tip = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader("Upload your lab report (PDF)", type="pdf", label_visibility="collapsed")

with col_tip:
    st.markdown(
        """
        <div class="tip-box">
            ✅ Text-based PDFs work best<br>
            ✅ Blood, urine, thyroid reports<br>
            🔒 File not stored anywhere<br>
            ⚡ Results in ~10 seconds
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Processing ────────────────────────────────────────────────────────────────
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    with st.spinner("Extracting lab values from PDF…"):
        lab_data = extract_lab_values(pdf_path)

    Path(pdf_path).unlink(missing_ok=True)

    if not lab_data:
        st.error("Could not parse lab values. Please try a text-based (not scanned) PDF.")
        st.stop()

    st.markdown(
        f"""
        <div class="stats-bar">
            <div class="stat-pill">📄 <strong>{uploaded.name}</strong></div>
            <div class="stat-pill">🧪 <strong>{len(lab_data)}</strong> tests detected</div>
            <div class="stat-pill">📦 <strong>{round(uploaded.size/1024, 1)} KB</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔬 Preview extracted values"):
        st.json(lab_data)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    if st.button("🔍 Analyse Report", disabled=bool(key_error)):
        if key_error:
            st.stop()

        graph = build_graph()
        try:
            with st.spinner("Running AI agents — explanations, flags, research, safety…"):
                result = graph.invoke({"lab_data": lab_data})
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate limit" in err or "quota" in err:
                st.error("**Groq rate limit hit.** Wait a few seconds and click **Analyse Report** again.")
                st.stop()
            if "api key" in err or "401" in err or "403" in err:
                st.error(
                    "Groq rejected your API key. Update `GROQ_API_KEY_MEDICAL` in `.env`."
                )
                st.stop()
            raise

        report = result.get("final_report", {})
        if "error" in report:
            st.error(f"Agent error: {report['error']}")
            st.json(report.get("raw", {}))
            st.stop()

        # ── Explanations ──────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-head"><span class="dot" style="background:#F06FA0"></span>Plain English Explanations</div>',
            unsafe_allow_html=True,
        )
        for item in report.get("explanations", []):
            st.markdown(
                f"""
                <div class="explain-card">
                    <div class="explain-test">{item['test']}</div>
                    <p class="explain-text">{item['plain_english']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Flags ─────────────────────────────────────────────────────────────
        flags = report.get("flags", [])
        counts = {"Normal": 0, "Watch": 0, "High priority": 0, "Critical": 0}
        for f in flags:
            sev = f.get("severity", "Normal")
            counts[sev] = counts.get(sev, 0) + 1

        st.markdown(
            f"""
            <div class="section-head">
                <span class="dot" style="background:#FFEABB"></span>
                Severity Flags
                <span style="margin-left:auto;display:flex;gap:0.6rem;font-size:0.75rem;font-weight:500">
                    <span style="color:#15803d">● {counts['Normal']} Normal</span>
                    <span style="color:#a16207">● {counts['Watch']} Watch</span>
                    <span style="color:#c2410c">● {counts['High priority']} High</span>
                    <span style="color:#be185d">● {counts['Critical']} Critical</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chip_map = {
            "Normal": "chip-normal",
            "Watch": "chip-watch",
            "High priority": "chip-high",
            "Critical": "chip-critical",
        }
        for item in flags:
            sev = item.get("severity", "Normal")
            st.markdown(
                f"""
                <div class="flag-row">
                    <span class="severity-chip {chip_map.get(sev, 'chip-normal')}">{sev}</span>
                    <div>
                        <div class="flag-test">{item['test']}</div>
                        <div class="flag-reason">{item['reason']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Doctor questions ──────────────────────────────────────────────────
        st.markdown(
            '<div class="section-head"><span class="dot" style="background:#3b82f6"></span>Questions to Ask Your Doctor</div>',
            unsafe_allow_html=True,
        )
        for i, q in enumerate(report.get("questions", []), 1):
            st.markdown(
                f"""
                <div class="q-card">
                    <div class="q-num">{i}</div>
                    <p class="q-text">{q['question']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="disclaimer">
                <span class="disclaimer-icon">⚠️</span>
                <span class="disclaimer-text">{report.get('disclaimer', '')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="mediscan_report.json",
            mime="application/json",
        )