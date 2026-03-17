import streamlit as st
from src.ingest import load_documents, chunk_documents
from src.embeddings import get_embedding_model
from src.vectorstore import get_vector_store
from src.retriever import build_index, get_retriever
from src.rag_pipeline import get_llm, generate_answer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brototype AI Guide",
    page_icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAe1BMVEUAAAD////ExMTu7u7Jycnf39/8/PywsLBZWVno6OhRUVGcnJzj4+OUlJS7u7uEhIS8vLyqqqpfX1/U1NR7e3sjIyPNzc2vr69MTExCQkL19fV0dHRra2sTExOPj48cHBw6OjowMDCkpKQqKipsbGw8PDx3d3dGRkYNDQ3pTOIeAAAFSklEQVR4nO2afXOiMBDGiWgsguILUqW1or1e+/0/4aECZjchxBnAu7nnN+M/EMw+JNndbPA8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMB/xvFrncQXkvfdtv/uwvGomekqnh34E6/N7Rev83Vu7W43HwWZFDWZn55ftC46JRUtZONf9Imo5YHonDf0tV8F5i4W6x4VjtsUXmwmGifWttfxGX0aetoZX6a8PRG9PVWhEFNnhSWx1tGi5YnN/pkKpRg/qFCsaDd7v6WHYiyTJyokFrsoLAwmEmeynL/NTxS/+VMVitkjCi8oQ/JtV1ejz+0hFaaPKZQiq2Pd0W8bweqZPnxqg0LdIFl5AtcxvLunkZvAAn84hQaVH0aFsqF1QTWI727qrv/x2q9CKaZhyXmVZszokVGhzdzynbSlCCpZ3qtCxZ0UHFgEC4wKNx/zkoWWsGyu7d+M8oMoyAwvhUTePhS+kHs0CanWyISYpAaFmI36bZqyVKZoMpl/X+4cvxI9zcmOQyqkCygzKiQLJ2bmXlzjiS/QTM0Bv7U533n6ZlNIjTMpFMw1MHtjg+rsRA3YsPvLIRVuyUoxK6Tr5oNaG3rXUGHpooDlcxm/36fCPVlXkYPCNbV2UVwK9EuU3+S+FPmACmPSczV9bApnVE7xyJZeId66hA1i1wm4ReHJV8ZQ1knjAwqLAXuhVwJPh0WlcCiF24Rmk7La1doUsvRloTkafZJ6XqI2kJ27GprT+EGNYNRSbApX1NhiOxTSGGnaIO1pR5MeFdrw60hsUXhgaco703y9orGlETMyNBlC4X1fY4n4LEW5hD62yL4NJvxQV2Naqr0rlKqDa8za1jw/ueSlLBzmJhtoav4UhWRyUR1BVStN9W1uois0FeHYP3Yd8t0Uyum9auu8A74OBlUojTXuv0FhYW4dSByrGPKWQy/p5ZPJhr9DoZCVr3Gt09ziGlNoLIrSyPScdXi1+eSs8LIeS6c/pYtTy7sLjjTG9KswXVQsN/51qimMnRUWqoJy4Z7pDVO9MKdN+o34O/WW5v0/nRWKtPJMbD+14v17Wqo3NjTpTiGdRD+shhS7KpT35IyZbxoglvacB1RI0+YqurcrjJSYkLMg+aWbYHyPAyn8NNUTHcZQNZK4EWmYpu/sHZicUW8KD7Tv1KKQtJT5/T9YHYaXabT/kx0LtCs0Zv2sXno5kA95VXB0/4+Q3eJbhym7n3odY1XIyi5jTWE961jZV973EDumQERkFJf8dudHbFaFpHtZbtDN+8MpbapsGyOekMtF1c0+1A5OpTGv604hrRPxUmdoUZirMqSaYs+5iAJ/MipyikDoBzpdR0OuMNnNamJeqy33UA17fLZNuteTji0H3JTuTxCJwpaD6INNISuqKSeB3NdYeujez/BKlFVu6SCb6jTkuhS/7304H69JNcz0obCFmV0hrV2XR2tXds5nwH0c5LsqdKh5+6S56rUSx2P8Hk6AHxjDoConNiokTlOqUV/zymZM5eKhFErh17vzRoVHNlJqZHuTt7+x9NB5Of8hhWoi0lwRZrVRsg3KJ3aFIujr6z03herrbVbIyvP+D+kpMX+YeEP2NICew9eXxdsNydm6parP/uyDduUlZQ5RfwJWfpgoJrxllyx9C0G0mca8EJ+qLTIyFdcZeXzjcT4/xj6bqv5m3tdXif3zY7yar5Nwevuk+Byv/111AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIB/lT/7qzu+lBL58wAAAABJRU5ErkJggg==",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&family=Inter:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #050b12;
    --surface:   #0c1520;
    --card:      #0f1e2e;
    --border:    #1a3048;
    --green:     #00ff88;
    --cyan:      #00cfff;
    --amber:     #ffb700;
    --text:      #cde8ff;
    --muted:     #5a7a94;
    --glow-g:    0 0 20px rgba(0,255,136,0.35);
    --glow-c:    0 0 20px rgba(0,207,255,0.35);
}

/* ── Global Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] { display: none !important; }

/* ── Style the header bar to match dark theme without hiding it ── */
[data-testid="stHeader"] {
    background-color: #050b12 !important;
    border-bottom: 1px solid #1a3048 !important;
}

.block-container { max-width: 780px !important; padding: 2rem 1.5rem 4rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Animated Background Grid ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,207,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,207,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Header Hero ── */
.hero-wrap {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
}

.hero-logo {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.4rem;
}

.logo-mark {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--green), var(--cyan));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem; font-weight: 700;
    color: #050b12;
    box-shadow: var(--glow-g);
    flex-shrink: 0;
}

.logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.55rem; font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, var(--green), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-tagline {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--green);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    opacity: 0.85;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.9rem, 5vw, 2.8rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0 auto 0.9rem;
    max-width: 600px;
    color: #ffffff;
    text-align: center;
}

.hero-title .accent { color: var(--green); }

.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    max-width: 480px;
    margin: 0 auto 2rem;
    line-height: 1.65;
    font-weight: 300;
    text-align: center;
}

/* ── Divider ── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--green), transparent);
    margin: 0 0 2rem;
    opacity: 0.5;
}

/* ── Stats Bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
}

.stat-item {
    text-align: center;
}

.stat-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--green);
}

.stat-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Input Area ── */
.input-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--cyan);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 8px;
}

.input-label::before {
    content: '>';
    color: var(--green);
    font-weight: 700;
}

/* Streamlit input override */
[data-testid="stTextInput"] > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.9rem 1.2rem !important;
    caret-color: var(--green) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--green) !important;
    box-shadow: var(--glow-g) !important;
    outline: none !important;
}

[data-testid="stTextInput"] > div > div > input::placeholder {
    color: var(--muted) !important;
    font-style: italic;
}

/* ── Thinking Spinner ── */
.thinking-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--cyan);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
    display: flex; align-items: center; gap: 14px;
}

.thinking-dot {
    width: 8px; height: 8px;
    background: var(--cyan);
    border-radius: 50%;
    animation: pulse 1.2s ease-in-out infinite;
    box-shadow: var(--glow-c);
    flex-shrink: 0;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.3; transform: scale(0.7); }
}

.thinking-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: var(--cyan);
}

/* ── Answer Card ── */
.answer-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.4rem;
    position: relative;
    overflow: hidden;
}

.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--green), var(--cyan));
    border-radius: 4px 0 0 4px;
}

.answer-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 1rem;
}

.answer-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--green);
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 4px;
    padding: 2px 8px;
}

.answer-body {
    font-size: 0.97rem;
    line-height: 1.75;
    color: var(--text);
}

/* ── Suggested Questions ── */
.suggest-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 2rem 0 0.8rem;
}

/* ── Streamlit button overrides (suggested questions) ── */
[data-testid="stButton"] > button {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: border-color 0.2s, color 0.2s !important;
    margin-bottom: 0.3rem !important;
}

[data-testid="stButton"] > button:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    background: rgba(0,207,255,0.06) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}

.footer-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 1.5px;
}

.footer-text span { color: var(--green); }

/* ── Streamlit spinner / status override ── */
[data-testid="stSpinner"] > div {
    color: var(--cyan) !important;
}

/* ── Hide default Streamlit elements ── */
#MainMenu, footer { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── RAG Initialization ────────────────────────────────────────────────────────
@st.cache_resource
def initialize_rag():
    documents = load_documents()
    nodes = chunk_documents(documents)
    embed_model = get_embedding_model()
    storage_context = get_vector_store()
    index = build_index(nodes, embed_model, storage_context)
    retriever = get_retriever(index)
    llm = get_llm()
    return retriever, llm


retriever, llm = initialize_rag()


# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">

  <div class="hero-logo">
    <div class="logo-mark">B</div>
    <div class="logo-text">Brototype</div>
  </div>

  <div class="hero-tagline">Your Rules &amp; Regulations Guide</div>

  <h1 class="hero-title" style="text-align:center;">
    Know the Code.<br>
    <span class="accent">Own Your Journey.</span>
  </h1>

  <p class="hero-sub" style="text-align:center;">
    Ask anything about Brototype's rules, policies, and programmes.
    Powered by AI — built for the next generation of software engineers.
  </p>

</div>

<div class="glow-divider"></div>

<div class="stats-bar">
  <div class="stat-item">
    <div class="stat-num">12mo</div>
    <div class="stat-label">Intensive Program</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">100%</div>
    <div class="stat-label">Placement Support</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">₹0</div>
    <div class="stat-label">Upfront Fees</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">AI</div>
    <div class="stat-label">Powered Guide</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Suggested Questions ───────────────────────────────────────────────────────
SUGGESTED = [
    "💼  What are the attendance rules at Brototype?",
    "📜  What happens if I violate the code of conduct?",
    "💰  When do I start paying fees after placement?",
    "🕐  What are the working hours and schedule?",
]

# Session state for pre-fill
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

st.markdown('<div class="input-label">Ask a question</div>', unsafe_allow_html=True)
question = st.text_input(
    label="question",
    value=st.session_state.prefill,
    placeholder="e.g. What are the attendance rules?",
    label_visibility="collapsed",
    key="main_input",
)

# ── Suggested Pills ───────────────────────────────────────────────────────────
st.markdown('<div class="suggest-label">Try asking</div>', unsafe_allow_html=True)
cols = st.columns(2)
for i, q in enumerate(SUGGESTED):
    with cols[i % 2]:
        if st.button(q, key=f"sq_{i}"):
            st.session_state.prefill = q.split("  ", 1)[-1]
            st.rerun()


# ── Answer Generation ─────────────────────────────────────────────────────────
if question and question.strip():
    clean_q = question.strip().lstrip(">").strip()
    # Clear prefill so next run is clean
    st.session_state.prefill = ""

    with st.spinner(""):
        st.markdown("""
        <div class="thinking-card">
          <div class="thinking-dot"></div>
          <div class="thinking-text">Scanning the knowledge base — hold tight...</div>
        </div>
        """, unsafe_allow_html=True)

        answer = generate_answer(clean_q, retriever, llm)

    st.markdown(f"""
    <div class="answer-card">
      <div class="answer-header">
        <span class="answer-badge">✦ AI Answer</span>
      </div>
      <div class="answer-body">{answer}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-text">
    Built for <span>Brototype</span> · Powered by RAG + LLM · Kerala's #1 Tech Institute
  </div>
</div>
""", unsafe_allow_html=True)
