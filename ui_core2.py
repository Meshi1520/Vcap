import html
import json

from dotenv import load_dotenv
import streamlit as st

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

# =====================================================================
# PAGE SETUP
# =====================================================================
st.set_page_config(
    page_title="CineExtract AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# STYLING
# =====================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---------- animated background ---------- */
    .stApp {
        background: linear-gradient(-45deg, #070b1a, #1a1040, #0b1b3a, #2a0f3a);
        background-size: 400% 400%;
        animation: bgshift 18s ease infinite;
    }
    @keyframes bgshift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp::before, .stApp::after {
        content: "";
        position: fixed;
        width: 520px; height: 520px;
        border-radius: 50%;
        filter: blur(110px);
        opacity: 0.35;
        z-index: 0;
        pointer-events: none;
        animation: float 14s ease-in-out infinite;
    }
    .stApp::before { background: #6366f1; top: -120px; left: -120px; }
    .stApp::after  { background: #ec4899; bottom: -150px; right: -120px; animation-delay: -7s; }
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50%      { transform: translate(60px, 40px) scale(1.15); }
    }

    header[data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1250px; padding-top: 1.5rem; position: relative; z-index: 1; }

    /* ---------- hero ---------- */
    .hero { text-align: center; padding: 1rem 0 0.4rem 0; }
    .badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        color: #c7d2fe;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(129,140,248,0.4);
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 4.2rem;
        font-weight: 900;
        line-height: 1.05;
        margin: 0;
        background: linear-gradient(90deg, #fbbf24, #f472b6, #818cf8, #38bdf8, #fbbf24);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 6s linear infinite;
    }
    @keyframes shine { to { background-position: 300% 0; } }
    .hero p { color: #94a3b8; font-size: 1.1rem; margin: 0.6rem 0 0 0; }

    /* ---------- pipeline strip ---------- */
    .pipeline {
        display: flex; justify-content: center; align-items: center;
        gap: 0.6rem; flex-wrap: wrap; margin: 1.4rem 0 2rem 0;
    }
    .step {
        padding: 0.55rem 1.1rem;
        border-radius: 12px;
        font-size: 0.85rem; font-weight: 600; color: #e2e8f0;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(8px);
    }
    .arrow { color: #6366f1; font-weight: 800; }

    /* ---------- glass panels ---------- */
    .panel-title {
        color: #f1f5f9; font-weight: 800; font-size: 1.1rem;
        margin-bottom: 0.6rem; letter-spacing: 0.02em;
    }
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 16px !important;
        color: #f1f5f9 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.28), 0 0 30px rgba(99,102,241,0.25) !important;
    }

    [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        border: none; border-radius: 16px;
        padding: 0.9rem 1rem;
        font-weight: 800; font-size: 1.05rem; color: white;
        background: linear-gradient(90deg, #6366f1, #ec4899, #f59e0b);
        background-size: 200% 100%;
        transition: all 0.25s ease;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background-position: 100% 0;
        transform: translateY(-3px);
        box-shadow: 0 14px 34px rgba(236,72,153,0.4);
        color: white;
    }

    /* ---------- tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px 12px 0 0;
        padding: 0.5rem 1.2rem;
        color: #cbd5e1;
    }
    .stTabs [aria-selected="true"] { background: rgba(99,102,241,0.25); color: white; }

    /* ---------- result cards ---------- */
    .title-card {
        position: relative; overflow: hidden;
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(236,72,153,0.25));
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 1rem;
        animation: rise 0.5s ease-out both;
    }
    .title-card .eyebrow {
        color: #c7d2fe; font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.2em; text-transform: uppercase;
    }
    .title-card .title { color: white; font-size: 2.3rem; font-weight: 900; margin-top: 0.3rem; line-height: 1.15; }
    .title-card .dir { color: #e2e8f0; margin-top: 0.5rem; font-size: 1rem; }
    .title-card .dir b { color: #fbbf24; }

    .tile {
        text-align: center;
        padding: 1rem 0.5rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
        animation: rise 0.55s ease-out both;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .tile:hover { transform: translateY(-4px); border-color: rgba(129,140,248,0.6); }
    .tile .num {
        font-size: 1.9rem; font-weight: 900;
        background: linear-gradient(90deg, #fbbf24, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .tile .lbl { color: #94a3b8; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; }

    .card {
        padding: 1.2rem 1.4rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
        animation: rise 0.6s ease-out both;
    }
    .card .lbl {
        color: #a5b4fc; font-size: 0.72rem; font-weight: 800;
        letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.6rem;
    }
    .card .txt { color: #f1f5f9; font-size: 1.02rem; line-height: 1.7; }
    .muted { color: #64748b; font-style: italic; }

    .chip {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        margin: 0 0.4rem 0.5rem 0;
        border-radius: 999px;
        font-size: 0.85rem; font-weight: 600; color: white;
        background: linear-gradient(90deg, rgba(99,102,241,0.55), rgba(236,72,153,0.45));
        border: 1px solid rgba(255,255,255,0.18);
        transition: transform 0.15s ease;
    }
    .chip:hover { transform: scale(1.08); }
    .chip.cast { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.18); }

    .ring-wrap { display: flex; align-items: center; justify-content: center; }
    .ring {
        --p: 0;
        width: 150px; height: 150px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        background: conic-gradient(#fbbf24 calc(var(--p) * 1%), rgba(255,255,255,0.1) 0);
        box-shadow: 0 0 40px rgba(251,191,36,0.25);
    }
    .ring-in {
        width: 116px; height: 116px; border-radius: 50%;
        background: #0f1630;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .ring-in .v { color: #fbbf24; font-size: 2.1rem; font-weight: 900; line-height: 1; }
    .ring-in .s { color: #94a3b8; font-size: 0.65rem; letter-spacing: 0.14em; margin-top: 0.3rem; font-weight: 700; }

    .placeholder {
        text-align: center; color: #64748b;
        border: 2px dashed rgba(255,255,255,0.14);
        border-radius: 24px; padding: 5rem 1rem;
    }
    .placeholder .big { font-size: 4rem; animation: bob 3s ease-in-out infinite; }
    @keyframes bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }

    .footer { text-align: center; color: #475569; font-size: 0.8rem; margin-top: 2.5rem; }

    @keyframes rise {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# HEADER
# =====================================================================
st.markdown(
    """
    <div class="hero">
        <div class="badge">AI POWERED MOVIE INTELLIGENCE</div>
        <h1>🎬 CineExtract AI</h1>
        <p>Drop in any paragraph about a movie and watch it turn into structured insight.</p>
    </div>
    <div class="pipeline">
        <div class="step">📝 Paragraph</div><div class="arrow">➜</div>
        <div class="step">🧩 Chat Prompt Template</div><div class="arrow">➜</div>
        <div class="step">⚡ Groq Model</div><div class="arrow">➜</div>
        <div class="step">📊 Structured Output</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# YOUR ORIGINAL LOGIC (unchanged)
# =====================================================================
class MovieInfo(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: float
    summary: str


parser = PydanticOutputParser(pydantic_object=MovieInfo)


@st.cache_resource
def get_model():
    return init_chat_model("openai/gpt-oss-120b", model_provider="groq")


model = get_model()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
    extract movie information from the given paragraph
    {format_instructions} """,
    ),
    (
        "human",
        """
    {paragraph} """,
    ),
])


# =====================================================================
# DISPLAY HELPERS (UI only)
# =====================================================================
def esc(x):
    return html.escape(str(x))


def chips(items, cls=""):
    if not items:
        return '<span class="muted">Not mentioned</span>'
    return "".join(f'<span class="chip {cls}">{esc(i)}</span>' for i in items)


def to_percent(r):
    try:
        r = float(r)
    except (TypeError, ValueError):
        return 0, "—"
    if r <= 10:
        return max(0, min(100, r * 10)), "/ 10"
    return max(0, min(100, r)), "/ 100"


def render_dashboard(d):
    title = d.get("title") or "Unknown title"
    year = d.get("release_year")
    director = d.get("director")
    genres = d.get("genre") or []
    cast = d.get("cast") or []
    rating = d.get("rating")
    summary = d.get("summary") or ""

    dir_html = (
        f'🎬 Directed by <b>{esc(director)}</b>'
        if director
        else '<span class="muted">Director not mentioned</span>'
    )
    st.markdown(
        f"""
        <div class="title-card">
            <div class="eyebrow">Now Extracting</div>
            <div class="title">{esc(title)}</div>
            <div class="dir">{dir_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    tiles = [
        (c1, esc(year) if year else "—", "Release Year"),
        (c2, len(genres), "Genres"),
        (c3, len(cast), "Cast Members"),
    ]
    for col, num, lbl in tiles:
        with col:
            st.markdown(
                f'<div class="tile"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    left, right = st.columns([1, 1.6])
    with left:
        pct, scale = to_percent(rating)
        shown = esc(rating) if rating is not None else "—"
        st.markdown(
            f"""
            <div class="card">
                <div class="lbl">⭐ Rating</div>
                <div class="ring-wrap">
                    <div class="ring" style="--p:{pct}">
                        <div class="ring-in"><div class="v">{shown}</div><div class="s">{scale}</div></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="card"><div class="lbl">🏷️ Genres</div>{chips(genres)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card"><div class="lbl">🎭 Cast</div>{chips(cast, "cast")}</div>',
            unsafe_allow_html=True,
        )

    summ = esc(summary) if summary else '<span class="muted">No summary returned</span>'
    st.markdown(
        f'<div class="card"><div class="lbl">📝 Summary</div><div class="txt">{summ}</div></div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# LAYOUT
# =====================================================================
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown('<div class="panel-title">📥 Your paragraph</div>', unsafe_allow_html=True)
    with st.form("extract_form", border=False):
        para = st.text_area(
            "give your paragraph :",
            height=360,
            placeholder="Paste a paragraph about any movie here...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🚀 Extract Movie Info")

with right:
    st.markdown('<div class="panel-title">📤 Results</div>', unsafe_allow_html=True)

    if submitted and para.strip():
        with st.spinner("Reading the paragraph and extracting details..."):
            final_prompt = prompt.invoke({
                "paragraph": para,
                "format_instructions": parser.get_format_instructions(),
            })
            response = model.invoke(final_prompt)

        st.toast("Extraction complete!", icon="✅")

        # display-only parsing of the model's JSON text
        try:
            data = parse_json_markdown(response.content)
            if not isinstance(data, dict):
                data = None
        except Exception:
            data = None

        tab_dash, tab_raw, tab_json = st.tabs(["📊 Dashboard", "📄 Raw output", "🧾 JSON"])

        with tab_dash:
            if data:
                render_dashboard(data)
            else:
                st.warning("Couldn't read the output as JSON. See the Raw output tab.")

        with tab_raw:
            st.code(response.content, language=None)

        with tab_json:
            if data:
                st.json(data)
            else:
                st.info("No valid JSON to display.")

    elif submitted:
        st.warning("Please paste a paragraph first.")
    else:
        st.markdown(
            """
            <div class="placeholder">
                <div class="big">🎥</div>
                <div style="margin-top:0.8rem;font-size:1.1rem;">
                    Your extracted movie details will appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer">Built with Streamlit · LangChain · Groq</div>',
    unsafe_allow_html=True,
)