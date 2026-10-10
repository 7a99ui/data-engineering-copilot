# ══════════════════════════════════════════════════════
# Data Engineering Copilot — Interface v2
# Thème sombre "enterprise" · dégradés indigo / cyan / violet
# Nécessite Streamlit >= 1.40  (pip install -U streamlit)
# ══════════════════════════════════════════════════════
import html
import streamlit as st
import requests
import pandas as pd

# ══════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE (doit être le premier appel Streamlit)
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Data Engineering Copilot",
    page_icon=":material/database:",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ══════════════════════════════════════════════════════
# DESIGN SYSTEM — couleurs, intents, exemples
# ══════════════════════════════════════════════════════
# Chaque intent a un label lisible et une couleur vive.
INTENTS = {
    "rag_search":    {"label": "Documentation search", "color": "#3B9EFF"},
    "db_inspect":    {"label": "Database inspect",     "color": "#10E0A0"},
    "sql_query":     {"label": "SQL query",            "color": "#FFB020"},
    "log_analysis":  {"label": "Log analysis",         "color": "#FF5C8A"},
    "direct_answer": {"label": "Direct answer",        "color": "#B26BFF"},
    "chitchat":      {"label": "Chat",                 "color": "#7C8BA8"},
    "unknown":       {"label": "Unknown",              "color": "#7C8BA8"},
}

EXAMPLES = [
    ("bolt",       "Why does Spark fail with OutOfMemoryError?"),
    ("table_chart", "What tables are in the database?"),
    ("error",      "Show me all failed pipelines"),
    ("schedule",   "Why did the pipeline fail last night?"),
    ("hub",        "What is Apache Airflow?"),
]


def intent_meta(intent: str) -> dict:
    return INTENTS.get(intent, INTENTS["unknown"])


def score_color(score: float) -> str:
    """Couleur du score de pertinence (vert / ambre / rouge vifs)."""
    if score >= 0.80:
        return "#10E0A0"
    if score >= 0.60:
        return "#FFB020"
    return "#FF5C6C"


# ══════════════════════════════════════════════════════
# CSS GLOBAL
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #070B16;
    --surface:   #0E1426;
    --surface-2: #131B33;
    --border:    rgba(130, 150, 255, 0.16);
    --border-hi: rgba(130, 150, 255, 0.38);
    --text:      #E8ECF8;
    --muted:     #8D9AB8;
    --indigo:    #6366F1;
    --cyan:      #22D3EE;
    --violet:    #A855F7;
    --grad:      linear-gradient(135deg, #6366F1 0%, #22D3EE 100%);
    --grad-2:    linear-gradient(135deg, #A855F7 0%, #6366F1 55%, #22D3EE 100%);
}

/* ── Base ── */
html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
}
code, pre, kbd, [data-testid="stCode"] * {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
}
.stApp {
    background:
        radial-gradient(1100px 560px at 88% -8%,  rgba(99,102,241,.26), transparent 60%),
        radial-gradient(900px 520px  at -8% 4%,   rgba(34,211,238,.14), transparent 55%),
        radial-gradient(700px 500px  at 50% 115%, rgba(168,85,247,.14), transparent 60%),
        var(--bg);
}
.block-container { padding-top: 1.6rem; padding-bottom: 7rem; max-width: 1100px; }

/* Header Streamlit transparent (on garde le bouton de sidebar) */
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stToolbar"] { visibility: hidden; }

/* ── Hero ── */
.hero {
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
    padding: 20px 24px; margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(99,102,241,.14), rgba(34,211,238,.06));
    border: 1px solid var(--border);
    border-radius: 20px;
    backdrop-filter: blur(10px);
}
.hero-left { display: flex; align-items: center; gap: 16px; }
.logo {
    width: 48px; height: 48px; border-radius: 14px; flex: none;
    background: var(--grad-2);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 8px 28px rgba(99,102,241,.55), inset 0 1px 0 rgba(255,255,255,.35);
}
.hero h1 {
    font-size: 24px; font-weight: 800; letter-spacing: -0.02em; margin: 0; padding: 0;
    background: linear-gradient(90deg, #FFFFFF 0%, #B9C4FF 60%, #7FE7FF 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { margin: 2px 0 0 0; font-size: 13px; color: var(--muted); }
.stack { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.stack span {
    font-size: 11.5px; font-weight: 600; color: #C7D2FE;
    padding: 5px 11px; border-radius: 999px;
    background: rgba(99,102,241,.14); border: 1px solid rgba(99,102,241,.38);
}

/* ── Messages de chat (composant natif stChatMessage) ── */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,.28);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(99,102,241,.30), rgba(34,211,238,.12));
    border-color: rgba(99,102,241,.5);
}
[data-testid="stChatMessageAvatarUser"] {
    background: var(--grad) !important; color: #fff !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--grad-2) !important; color: #fff !important;
    box-shadow: 0 0 18px rgba(168,85,247,.55);
}
[data-testid="stChatMessage"] p { line-height: 1.7; font-size: 14.8px; }

/* ── Badges d'intent et chemin des steps ── */
.meta-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 12px; }
.badge {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 999px;
    color: var(--c);
    background: color-mix(in srgb, var(--c) 14%, transparent);
    border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);
}
.badge i { width: 7px; height: 7px; border-radius: 50%; background: var(--c); box-shadow: 0 0 8px var(--c); }
.steps { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 6px; font-size: 11.5px; color: var(--muted); }
.steps span { padding: 2px 9px; border-radius: 8px; background: rgba(255,255,255,.04); border: 1px solid var(--border); }
.steps span + span::before { content: "›"; margin-right: 8px; color: var(--cyan); }

/* ── Sources ── */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.chip {
    display: inline-flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text);
    padding: 5px 11px; border-radius: 10px;
    background: var(--surface-2); border: 1px solid var(--border);
}
.chip b { font-weight: 700; color: var(--c); }
.src-card {
    padding: 14px 16px; margin-bottom: 10px; border-radius: 14px;
    background: var(--surface-2); border: 1px solid var(--border);
}
.src-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.src-name { font-size: 13px; font-weight: 600; color: var(--text); }
.src-score { font-size: 15px; font-weight: 800; color: var(--c); }
.bar { height: 5px; border-radius: 99px; background: rgba(255,255,255,.07); overflow: hidden; margin-bottom: 10px; }
.bar span { display: block; height: 100%; border-radius: 99px; background: var(--c); box-shadow: 0 0 10px var(--c); }
.src-chunk { font-size: 12.5px; line-height: 1.6; color: var(--muted); white-space: pre-wrap; }

/* ── Bloc SQL ── */
.sql-head {
    display: flex; align-items: center; gap: 10px; margin: 14px 0 8px 0;
    font-size: 13px; font-weight: 700; color: #FFB020;
}
.sql-head.ok  { color: #10E0A0; }
.sql-head.ko  { color: #FF5C6C; }
.sql-head::before {
    content: ""; width: 8px; height: 8px; border-radius: 50%;
    background: currentColor; box-shadow: 0 0 10px currentColor;
}

/* ── Boutons (zone principale) ── */
.stButton > button {
    border-radius: 12px; font-weight: 600; font-size: 13.5px;
    background: var(--surface-2); color: var(--text);
    border: 1px solid var(--border);
    transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease;
}
.stButton > button:hover {
    border-color: var(--cyan); color: #fff;
    box-shadow: 0 0 0 3px rgba(34,211,238,.14);
}
.stButton > button[kind="primary"] {
    background: var(--grad); border: none; color: #04101F;
    box-shadow: 0 6px 22px rgba(34,211,238,.35);
}
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 10px 28px rgba(34,211,238,.5); }

/* ── Cartes de démarrage (écran d'accueil) ── */
.welcome { text-align: center; padding: 34px 10px 22px 10px; }
.welcome h2 { font-size: 26px; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 8px 0; color: #fff; }
.welcome p  { font-size: 14.5px; color: var(--muted); margin: 0; }
.st-key-welcome_cards .stButton > button {
    width: 100%; min-height: 78px; padding: 16px 18px; text-align: left; justify-content: flex-start;
    background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(19,27,51,.9));
    border: 1px solid var(--border-hi); border-radius: 16px; font-size: 14px;
}
.st-key-welcome_cards .stButton > button:hover {
    border-color: var(--cyan);
    background: linear-gradient(135deg, rgba(99,102,241,.26), rgba(34,211,238,.12));
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1122 0%, #070B16 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stButton > button { width: 100%; justify-content: flex-start; text-align: left; }
.side-brand { display: flex; align-items: center; gap: 12px; padding: 6px 2px 18px 2px; }
.side-brand .logo { width: 40px; height: 40px; border-radius: 12px; }
.side-brand b { display: block; font-size: 15px; font-weight: 700; color: #fff; }
.side-brand small { color: var(--muted); font-size: 12px; }
.side-title { font-size: 11.5px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
              color: var(--muted); margin: 18px 0 8px 0; }
.status {
    display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: 12px;
    font-size: 13px; font-weight: 600; color: var(--c);
    background: color-mix(in srgb, var(--c) 10%, transparent);
    border: 1px solid color-mix(in srgb, var(--c) 40%, transparent);
}
.status i { width: 9px; height: 9px; border-radius: 50%; background: var(--c); box-shadow: 0 0 10px var(--c); }
.status.live i { animation: pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.stat {
    padding: 12px 14px; border-radius: 14px;
    background: var(--surface-2); border: 1px solid var(--border);
}
.stat b { display: block; font-size: 24px; font-weight: 800; background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
.stat small { color: var(--muted); font-size: 11.5px; }
.intent-row { display: flex; justify-content: space-between; align-items: center; font-size: 12.5px; padding: 5px 2px; color: var(--text); }
.intent-row span:first-child { display: flex; align-items: center; gap: 8px; }
.intent-row i { width: 8px; height: 8px; border-radius: 50%; background: var(--c); box-shadow: 0 0 8px var(--c); }
.intent-row em { font-style: normal; color: var(--muted); font-weight: 600; }

/* ── Champ de saisie ── */
[data-testid="stChatInput"] > div {
    background: var(--surface); border: 1px solid var(--border-hi); border-radius: 16px;
    box-shadow: 0 10px 40px rgba(0,0,0,.45);
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--cyan); box-shadow: 0 0 0 3px rgba(34,211,238,.18), 0 10px 40px rgba(0,0,0,.45);
}
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent; }

/* ── Divers ── */
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 14px; background: transparent; }
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: rgba(130,150,255,.25); border-radius: 8px; }

@media (max-width: 760px) {
    .hero { flex-direction: column; align-items: flex-start; }
    .stack { justify-content: flex-start; }
}
@media (prefers-reduced-motion: reduce) { .status.live i { animation: none; } }
</style>
""", unsafe_allow_html=True)

LOGO_SVG = (
    '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="white" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<ellipse cx="12" cy="6" rx="8" ry="3"/>'
    '<path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/>'
    '<path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>'
)


# ══════════════════════════════════════════════════════
# APPELS API
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=5, show_spinner=False)
def check_api_health() -> bool:
    """Vérifie que FastAPI répond (résultat mis en cache 5 s)."""
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def _error_result(error: str, answer: str) -> dict:
    return {
        "error": error, "answer": answer, "intent": "unknown",
        "sources": [], "steps": [], "thread_id": None,
        "sql_query": None, "pending_approval": False,
    }


def call_agent(question=None, thread_id=None, sql_approved=None, endpoint="/agent", history=None) -> dict:
    """
    Appelle l'API FastAPI.
    - question seule            -> nouvelle question
    - thread_id + sql_approved  -> décision HITL sur une requête SQL
    """
    payload = {}
    if question:
        payload["question"] = question
    if thread_id:
        payload["thread_id"] = thread_id
    if sql_approved is not None:
        payload["sql_approved"] = sql_approved
    
    # ── Envoyer les 6 derniers messages pour la mémoire conversationnelle ──
    # [:-1] exclut la question qu'on vient d'ajouter
    # pour éviter de l'envoyer deux fois dans le payload
    if history:
        payload["history"] = history

    try:
        response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return _error_result(
            "API unreachable. Start it with: uvicorn app.main:app --port 8000",
            "The API is unreachable. Start the backend and try again.",
        )
    except requests.exceptions.Timeout:
        return _error_result("Timeout", "The request timed out. Try again.")
    except Exception as e:
        return _error_result(str(e), f"Unexpected error: {e}")


def make_assistant_msg(result: dict, default_intent: str = "unknown") -> dict:
    """Transforme la réponse de l'API en message d'historique."""
    return {
        "role": "assistant",
        "content": result.get("answer", ""),
        "intent": result.get("intent", default_intent),
        "steps": result.get("steps", []),
        "sources": result.get("sources", []),
        "pending_approval": result.get("pending_approval", False),
        "sql_query": result.get("sql_query"),
        "thread_id": result.get("thread_id"),
        "error": result.get("error"),
        "sql_result": result.get("sql_result"),
    }


# ══════════════════════════════════════════════════════
# AFFICHAGE
# ══════════════════════════════════════════════════════
def render_sources(sources: list):
    chips = ""
    for s in sources:
        c = score_color(s["score"])
        chips += (
            f'<span class="chip" style="--c:{c}">{html.escape(str(s["source"]))}'
            f'<b>{s["score"]:.2f}</b></span>'
        )
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    with st.expander(f"View source excerpts ({len(sources)})", icon=":material/menu_book:"):
        cards = ""
        for s in sources:
            c = score_color(s["score"])
            chunk = str(s.get("chunk", ""))
            chunk = html.escape(chunk[:400] + ("…" if len(chunk) > 400 else ""))
            cards += (
                f'<div class="src-card" style="--c:{c}">'
                f'<div class="src-top"><span class="src-name">{html.escape(str(s["source"]))}'
                f' · page {html.escape(str(s.get("page", "-")))}</span>'
                f'<span class="src-score">{s["score"]:.2f}</span></div>'
                f'<div class="bar"><span style="width:{max(0, min(100, s["score"] * 100)):.0f}%"></span></div>'
                f'<div class="src-chunk">{chunk}</div></div>'
            )
        st.markdown(cards, unsafe_allow_html=True)


def render_sql_block(msg: dict, idx: int):
    """Affiche la requête SQL et, si besoin, les boutons d'approbation."""
    status = msg.get("sql_status")
    if msg.get("pending_approval"):
        st.markdown('<div class="sql-head">Approval required before execution</div>', unsafe_allow_html=True)
    elif status == "approved":
        st.markdown('<div class="sql-head ok">Approved and executed</div>', unsafe_allow_html=True)
    elif status == "rejected":
        st.markdown('<div class="sql-head ko">Rejected — not executed</div>', unsafe_allow_html=True)

    st.code(msg["sql_query"], language="sql")

    if not msg.get("pending_approval"):
        return

    col_ok, col_ko, _ = st.columns([1.2, 1.2, 4])
    with col_ok:
        if st.button("Approve", key=f"approve_{idx}", type="primary", icon=":material/check:"):
            with st.spinner("Executing SQL query…"):
                result = call_agent(thread_id=msg["thread_id"], sql_approved=True)
            msg["pending_approval"] = False
            msg["sql_status"] = "approved"
            st.session_state.messages.append(make_assistant_msg(result, "sql_query"))
            st.rerun()
    with col_ko:
        if st.button("Reject", key=f"reject_{idx}", icon=":material/close:"):
            result = call_agent(thread_id=msg["thread_id"], sql_approved=False)
            msg["pending_approval"] = False
            msg["sql_status"] = "rejected"
            st.session_state.messages.append({
                "role": "assistant",
                "content": "The SQL query was rejected and has not been executed.",
                "intent": "sql_query",
                "steps": result.get("steps", []),
                "sources": [],
            })
            st.rerun()


def display_message(msg: dict, idx: int):
    if msg["role"] == "user":
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(msg["content"])
        return

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        if msg.get("intent"):
            meta = intent_meta(msg["intent"])
            steps = "".join(f"<span>{html.escape(str(s))}</span>" for s in msg.get("steps", []))
            st.markdown(
                f'<div class="meta-row">'
                f'<span class="badge" style="--c:{meta["color"]}"><i></i>{meta["label"]}</span>'
                f'<div class="steps">{steps}</div></div>',
                unsafe_allow_html=True,
            )

        # Rendu Markdown natif : tableaux, code, listes fonctionnent
        st.markdown(msg["content"])
        
        sql_res = msg.get("sql_result")
        if sql_res and sql_res.get("success") and sql_res.get("rows"):
            df = pd.DataFrame(sql_res["rows"])
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption(f"{sql_res.get('row_count', len(df))} row(s) returned")

        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("sql_query"):
            if msg.get("pending_approval") or msg.get("sql_status"):
                # Message en attente ou décidé → affichage complet avec badge
                render_sql_block(msg, idx)
            else:
                # Message de résultat → requête repliée, pas de répétition
                with st.expander("View executed SQL", icon=":material/code:"):
                    st.code(msg["sql_query"], language="sql")


# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════
st.session_state.setdefault("messages", [])
st.session_state.setdefault("mode", "agent")

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f'<div class="side-brand"><div class="logo">{LOGO_SVG}</div>'
        f'<div><b>DE Copilot</b><small>Data Engineering Assistant</small></div></div>',
        unsafe_allow_html=True,
    )

    if check_api_health():
        st.markdown('<div class="status live" style="--c:#10E0A0"><i></i>API connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status" style="--c:#FF5C6C"><i></i>API offline</div>', unsafe_allow_html=True)
        st.caption("Run: `uvicorn app.main:app --port 8000`")

    st.markdown('<div class="side-title">Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Response mode",
        ["Agent (Smart)", "RAG only"],
        label_visibility="collapsed",
    )
    st.session_state.mode = "agent" if mode == "Agent (Smart)" else "ask"

    # Statistiques de session
    user_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
    intent_counts: dict = {}
    for m in st.session_state.messages:
        if m["role"] == "assistant" and m.get("intent"):
            intent_counts[m["intent"]] = intent_counts.get(m["intent"], 0) + 1

    st.markdown('<div class="side-title">Session</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="stat-grid">'
        f'<div class="stat"><b>{user_count}</b><small>Questions</small></div>'
        f'<div class="stat"><b>{len(intent_counts)}</b><small>Intents used</small></div></div>',
        unsafe_allow_html=True,
    )

    if intent_counts:
        rows = ""
        for intent, count in intent_counts.items():
            meta = intent_meta(intent)
            rows += (
                f'<div class="intent-row" style="--c:{meta["color"]}">'
                f'<span><i></i>{meta["label"]}</span><em>{count}</em></div>'
            )
        st.markdown(f'<div style="margin-top:12px;">{rows}</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-title">Quick questions</div>', unsafe_allow_html=True)
    for i, (icon, question) in enumerate(EXAMPLES):
        short = question if len(question) <= 38 else question[:36] + "…"
        if st.button(short, key=f"ex_{i}", icon=f":material/{icon}:"):
            st.session_state.pending_question = question
            st.rerun()

    st.markdown('<div class="side-title">Actions</div>', unsafe_allow_html=True)
    if st.button("Clear conversation", icon=":material/delete:"):
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════════════════
# ZONE PRINCIPALE
# ══════════════════════════════════════════════════════
st.markdown(
    f"""
<div class="hero">
    <div class="hero-left">
        <div class="logo">{LOGO_SVG}</div>
        <div>
            <h1>Data Engineering Copilot</h1>
            <p>Search docs, inspect your database and analyze pipeline logs — in one place.</p>
        </div>
    </div>
    <div class="stack"><span>Qwen2.5 3B</span><span>LangGraph</span><span>Qdrant</span><span>PostgreSQL</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Écran d'accueil (aucun message) ──
if not st.session_state.messages:
    st.markdown(
        '<div class="welcome"><h2>What do you want to know about your data platform?</h2>'
        '<p>Ask a question, or start with one of these.</p></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="welcome_cards"):
        cols = st.columns(2)
        for i, (icon, question) in enumerate(EXAMPLES[:4]):
            with cols[i % 2]:
                if st.button(question, key=f"welcome_{i}", icon=f":material/{icon}:"):
                    st.session_state.pending_question = question
                    st.rerun()

# ── Historique ──
for i, msg in enumerate(st.session_state.messages):
    display_message(msg, i)

# ── Nouvelle question (saisie ou exemple cliqué) ──
typed = st.chat_input("Ask about Spark, Airflow, your database or pipeline logs…")
question = st.session_state.pop("pending_question", None) or typed

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    # On affiche tout de suite la question, puis un spinner dans la bulle de réponse
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        with st.spinner("Analyzing your question…"):
            endpoint = "/agent" if st.session_state.mode == "agent" else "/ask"
            # 6 derniers messages ; [:-1] exclut la question qu'on vient d'ajouter
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1][-6:]
            ]
            result = call_agent(question=question, endpoint=endpoint, history=history)

    st.session_state.messages.append(
    make_assistant_msg(result, "rag_search" if endpoint == "/ask" else "unknown")
    )
    st.rerun()