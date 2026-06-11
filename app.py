import os
import urllib.parse
import json
import base64
import random
import requests
import uuid
import time

import streamlit as st
import streamlit.components.v1 as components

from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from dotenv import load_dotenv

# ============================================================
# RAG simple sobre PDFs locales (carpeta ./rag_docs)
# ============================================================
from typing import List
from pypdf import PdfReader
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None

RAG_FOLDER = "rag_docs"
RAG_INDEX_PATH = "rag_index.faiss"
RAG_METADATA_PATH = "rag_metadata.json"

_rag_model = None
_rag_index = None
_rag_texts: List[str] = []


def rag_load_model():
    global _rag_model
    if _rag_model is None and SentenceTransformer is not None:
        _rag_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _rag_model


def rag_read_pdfs(folder: str) -> List[str]:
    texts = []
    if not os.path.isdir(folder):
        return texts
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            reader = PdfReader(fpath)
            pages_text = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    pages_text.append(t)
            if pages_text:
                doc_text = "\n".join(pages_text)
                texts.append(doc_text)
        except Exception as e:
            print(f"RAG: error leyendo {fpath}: {e}")
    return texts


def rag_build_index():
    global _rag_index, _rag_texts
    if faiss is None:
        print("RAG: faiss no disponible, índice no creado.")
        return

    model = rag_load_model()
    if model is None:
        print("RAG: modelo de embeddings no disponible.")
        return

    texts = rag_read_pdfs(RAG_FOLDER)
    if not texts:
        print("RAG: no se encontraron PDFs en la carpeta rag_docs.")
        return

    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    _rag_index = index
    _rag_texts = texts

    try:
        faiss.write_index(index, RAG_INDEX_PATH)
        with open(RAG_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump({"texts": texts}, f)
    except Exception as e:
        print(f"RAG: error guardando índice: {e}")


def rag_load_index():
    global _rag_index, _rag_texts
    if _rag_index is not None:
        return

    if faiss is None:
        print("RAG: faiss no disponible, no se puede cargar índice.")
        return

    if os.path.exists(RAG_INDEX_PATH) and os.path.exists(RAG_METADATA_PATH):
        try:
            _rag_index = faiss.read_index(RAG_INDEX_PATH)
            with open(RAG_METADATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            _rag_texts = data.get("texts", [])
            return
        except Exception as e:
            print(f"RAG: error cargando índice, se reconstruirá: {e}")

    rag_build_index()


def rag_retrieve_context(query: str, top_k: int = 3) -> str:
    rag_load_index()
    if _rag_index is None or not _rag_texts:
        return ""

    model = rag_load_model()
    if model is None:
        return ""

    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    k = min(top_k, len(_rag_texts))
    distances, indices = _rag_index.search(q_emb, k)
    indices = indices[0]

    chunks = []
    for idx in indices:
        if 0 <= idx < len(_rag_texts):
            chunks.append(_rag_texts[idx])

    if not chunks:
        return ""

    context = "\n\n".join(chunks)
    max_chars = 4000
    if len(context) > max_chars:
        context = context[:max_chars]
    return context

# ------------------------------------------------------------
# Configuración inicial de Streamlit
# ------------------------------------------------------------
st.set_page_config(
    page_title="NICO | Asistente Virtual UMSNH",
    page_icon="🦊",
    layout="wide",
)

# ------------------------------------------------------------
# FIX redirección /oauth2callback (al inicio del archivo)
# ------------------------------------------------------------
_request_uri = os.environ.get("STREAMLIT_SERVER_REQUEST_URI", "")
if "/oauth2callback" in _request_uri:
    parsed = urllib.parse.urlparse(_request_uri)
    query = urllib.parse.parse_qs(parsed.query)
    query_clean = {k: v[0] for k, v in query.items()}
    st.query_params.update(query_clean)
    st.rerun()

# ------------------------------------------------------------
# Cargar variables de entorno
# ------------------------------------------------------------
load_dotenv()

CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID", os.getenv("GOOGLE_CLIENT_ID", ""))
CLIENT_SECRET = st.secrets.get(
    "GOOGLE_CLIENT_SECRET", os.getenv("GOOGLE_CLIENT_SECRET", "")
)
GOOGLE_REDIRECT_URI = st.secrets.get(
    "GOOGLE_REDIRECT_URI",
    os.getenv("GOOGLE_REDIRECT_URI", "https://nicooapp-umsnh.streamlit.app/"),
)

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
GEMINI_MODEL = st.secrets.get(
    "GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite-preview-02-05")
)

# ============================================================
# Funciones auxiliares
# ============================================================

def get_flow(state=None):
    client_config = {
        "web": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [
                "https://nicooapp-umsnh.streamlit.app/",
                "http://localhost:8501/",
                "http://127.0.0.1:8501/",
            ],
        }
    }

    flow = Flow.from_client_config(
        client_config, scopes=SCOPES, redirect_uri=GOOGLE_REDIRECT_URI
    )
    if state:
        flow.redirect_uri = GOOGLE_REDIRECT_URI
    return flow


def ensure_session_defaults():
    st.session_state.setdefault("logged", False)
    st.session_state.setdefault("profile", {})
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("voice_on", True)
    st.session_state.setdefault("temperature", 0.7)
    st.session_state.setdefault("top_p", 0.9)
    st.session_state.setdefault("max_tokens", 256)
    st.session_state.setdefault("current_video", None)
    st.session_state.setdefault("open_cfg", False)
    st.session_state.setdefault("greeted", False)
    st.session_state.setdefault("input_val", "")
    st.session_state.setdefault("trigger_run", False)
    st.session_state.setdefault("is_exchanging_token", False)


def header_html():
    video_path = "assets/videos/nico_header_video.mp4"
    video_tag = '<div class="nico-placeholder">🦊</div>'
    
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        video_tag = f"""
        <video class="nico-video" autoplay loop muted playsinline>
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        </video>
        """

    return f"""
    <style>
    .nico-header {{
        background: linear-gradient(90deg, #0f2347 0%, #1a3b6e 100%);
        color: #fff;
        padding: 16px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .nico-wrap {{ display: flex; align-items: center; gap: 16px; }}
    .nico-video, .nico-placeholder {{
        width: 60px; height: 60px; border-radius: 50%;
        background: #fff; object-fit: cover; border: 2px solid #ffd700;
        display: flex; align-items: center; justify-content: center; font-size: 30px;
    }}
    .nico-title {{ font-size: 24px; font-weight: 800; margin: 0; }}
    .nico-subtitle {{ margin: 0; font-size: 16px; opacity: 0.8; font-weight: 300; }}
    .chat-bubble {{
        background: #f0f2f6; border-radius: 12px; padding: 16px; margin-top: 8px;
        color: #31333F; border-left: 4px solid #0f2347;
    }}
    </style>
    <div class="nico-header">
        <div class="nico-wrap">
            {video_tag}
            <div>
                <p class="nico-title">NICO</p>
                <p class="nico-subtitle">Asistente Virtual UMSNH</p>
            </div>
        </div>
    </div>
    """


def login_view():
    st.markdown(header_html(), unsafe_allow_html=True)
    st.info("Inicia sesión con tu cuenta de Google para usar NICO.")

    if not CLIENT_ID or not CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
        st.error("Faltan variables de configuración OAuth.")
        return

    if "oauth_state" not in st.session_state:
        st.session_state["oauth_state"] = str(uuid.uuid4())

    state_key = st.session_state["oauth_state"]
    flow = get_flow(state=state_key)

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes=False,
        prompt="consent",
        state=state_key,
    )

    st.query_params["oauth_state"] = state_key
    st.markdown(f"[🔐 Iniciar sesión con Google]({auth_url})")


def exchange_code_for_token():
    try:
        params = st.query_params
        code = params.get("code")
        state = params.get("state")
    except:
        return

    if not code or not state:
        return

    if st.session_state.get("is_exchanging_token"):
        return

    st.session_state["is_exchanging_token"] = True

    try:
        if "oauth_state" not in st.session_state:
            st.session_state["oauth_state"] = state

        if state != st.session_state.get("oauth_state"):
            st.warning("⚠️ El estado OAuth se regeneró automáticamente.")
            st.session_state["oauth_state"] = state

        flow = get_flow(state=state)
        flow.fetch_token(code=code)
        creds = flow.credentials

        request = grequests.Request()
        idinfo = id_token.verify_oauth2_token(creds.id_token, request, CLIENT_ID)

        st.session_state["logged"] = True
        st.session_state["profile"] = {
            "email": idinfo.get("email"),
            "name": idinfo.get("name"),
            "picture": idinfo.get("picture"),
        }
        
        st.session_state["is_exchanging_token"] = False
        st.query_params.clear()
        st.rerun() 

    except Exception as e:
        st.error(f"Error al autenticar: {e}")
        st.session_state["is_exchanging_token"] = False
        st.query_params.clear()
        st.rerun()


# ============================================================
# Gemini 2.0 con búsqueda en internet (prompt único)
# ============================================================
def gemini_generate(prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "topP": float(top_p),
            "maxOutputTokens": int(max_tokens),
        },
        "tools": [{"google_search": {}}],
    }

    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=40)
        r.raise_for_status()
        data = r.json()
        text = ""
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                text += part.get("text", "")
        return text.strip() or "No obtuve respuesta del modelo."
    except Exception as e:
        return f"⚠️ Error con Gemini: {e}"


def speak_browser(text: str):
    if not text:
        return
    payload = json.dumps(text)

    js_code = f"""
    <script>
    (function() {{
        const text = {payload};
        const synth = window.speechSynthesis;
        if (!synth) return;

        function findVideo() {{
            const v = parent.document.querySelector('video');
            return v;
        }}

        function speak() {{
            synth.cancel();
            const utter = new SpeechSynthesisUtterance(text);
            const voices = synth.getVoices() || [];
            let chosen = null;
            
            const preferNames = ["miguel", "diego", "jorge", "pablo", "male", "hombre"];
            for (const v of voices) {{
                const name = (v.name || "").toLowerCase();
                const lang = (v.lang || "").toLowerCase();
                if (lang.startsWith("es")) {{
                    for (const pref of preferNames) {{
                        if (name.includes(pref)) {{ chosen = v; break; }}
                    }}
                }}
                if (chosen) break;
            }}
            if (!chosen) {{
                for (const v of voices) {{
                    if (v.lang.toLowerCase().startsWith("es")) {{ chosen = v; break; }}
                }}
            }}
            if (chosen) utter.voice = chosen;

            utter.rate = 0.95;
            utter.pitch = 0.65;

            utter.onstart = () => {{ const v = findVideo(); if (v) v.play(); }};
            utter.onend = () => {{ const v = findVideo(); if (v) v.pause(); }};

            synth.speak(utter);
        }}

        if (synth.getVoices().length === 0) {{
            synth.addEventListener('voiceschanged', function handler() {{
                synth.removeEventListener('voiceschanged', handler);
                speak();
            }});
        }} else {{
            speak();
        }}
    }})();
    </script>
    """
    components.html(js_code, height=0)


# ============================================================
# Lógica principal de la app
# ============================================================

ensure_session_defaults()
exchange_code_for_token()

if not st.session_state.get("logged"):
    login_view()
    st.stop()

st.markdown(header_html(), unsafe_allow_html=True)

conv_col, video_col = st.columns([0.7, 0.3])

with video_col:
    video_container = st.empty()
    
    if not st.session_state["current_video"]:
        try:
            video_files = [f for f in os.listdir("assets/videos") if f.lower().endswith((".mp4", ".webm"))]
            if video_files:
                chosen = random.choice(video_files)
                video_path = os.path.join("assets/videos", chosen)
                with open(video_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                
                st.session_state["current_video"] = f"""
                <video width="220" loop muted playsinline style="border-radius:12px;">
                    <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                </video>
                """
        except:
            pass
            
    if st.session_state["current_video"]:
        video_container.markdown(st.session_state["current_video"], unsafe_allow_html=True)

with conv_col:
    c1, c2, c3 = st.columns([0.15, 0.15, 0.7])
    with c1:
        if st.button("🎙️ Voz: " + ("ON" if st.session_state["voice_on"] else "OFF")):
            st.session_state["voice_on"] = not st.session_state["voice_on"]
            st.rerun() 
    with c2:
        if st.button("⚙️ Config"):
            st.session_state["open_cfg"] = True
    with c3:
        st.write(f"Bienvenido, **{st.session_state['profile'].get('name', '')}**")

    if st.session_state.get("open_cfg"):
        with st.expander("Configuración del Modelo"):
            st.slider("Temperatura", 0.0, 1.5, key="temperature")
            st.slider("Top-P", 0.0, 1.0, key="top_p")
            st.slider("Máx. tokens", 64, 2048, key="max_tokens", step=32)
            if st.button("Cerrar Config"):
                st.session_state["open_cfg"] = False
                st.rerun()

    st.markdown("### 💬 Conversación")

    def action_submit():
        if st.session_state["input_val"].strip():
            st.session_state["trigger_run"] = True

    def action_clear():
        st.session_state["input_val"] = ""
        st.session_state["trigger_run"] = False

    st.text_input(
        "Escribe tu pregunta:", 
        key="input_val", 
        on_change=action_submit
    )

    btn_c1, btn_c2, _ = st.columns([0.15, 0.15, 0.7])
    with btn_c1:
        st.button("Enviar 🚀", on_click=action_submit)
    with btn_c2:
        st.button("Borrar 🗑️", on_click=action_clear)

    if st.session_state["trigger_run"]:
        user_msg = st.session_state["input_val"]
        
        st.session_state["history"].append({"role": "user", "content": user_msg})

        try:
            video_files = [f for f in os.listdir("assets/videos") if f.lower().endswith((".mp4", ".webm"))]
            if video_files:
                chosen = random.choice(video_files)
                video_path = os.path.join("assets/videos", chosen)
                with open(video_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                
                html_video = f"""
                <video width="220" loop muted playsinline style="border-radius:12px;">
                    <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                </video>
                """
                st.session_state["current_video"] = html_video
                video_container.markdown(html_video, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Video error: {e}")

        full_name = st.session_state['profile'].get('name', 'Usuario')
        first_name = full_name.split(' ')[0] if full_name else 'Amigo'

        sys_prompt = (
            "Eres NICO, asistente institucional de la Universidad Michoacana de San Nicolás de Hidalgo (UMSNH). "
            f"El usuario se llama {first_name}. "
            "se responsable e incluyente, eficiente y ético"
            "Tu personalidad es alegre y jovial"
            "Eres la mascota de la UMSNH eres un zorro, puedes hablar en español o purepecha"
            "NO uses negritas, NO uses Markdown, NO uses símbolos como **, *, _, #, ~~, etc.  "
            "NO generes listas con guiones viñetas asteriscos o puntos. "
            "Tu objetivo principal es proporcionar información precisa, actualizada y relevante de la UMSNH. "
            "ANTE CUALQUIER PREGUNTA SOBRE NOTICIAS, CONTACTOS, O ACTUALIDAD (DESPUÉS DE 2023), DEBES EJECUTAR LA HERRAMIENTA DE BÚSQUEDA WEB DE GOOGLE (GoGoGoogleSearchh)"
            "Responde siempre en español de mexico o en purépecha si es solicitado de forma clara, breve y amable. "
            "**IMPORTANTE: NO saludes al inicio de tu respuesta (ej. no digas 'Hola', 'Buenos días', 'Qué tal {nombre}'). El sistema ya saluda por ti la primera vez. Comienza directamente con la información solicitada o la respuesta a la pregunta.**"
            "Usa su nombre ocasionalmente en la conversación para que suene natural, pero no en cada frase.\n "
            "para nombres de funcionarios busca la web en https://umich.mx/unidades-administrativas/"
            "Prioriza sitios *.umich.mx."
            "- https://www.umich.mx\n"
            "-https://www.gacetanicolaita.umich.mx/n"
            "-https://umich.mx/unidades-administrativas/n"
            "- https://www.dce.umich.mx\n"
            "- https://siia.umich.mx\n"
            "Solo si te preguntan quien es la rectora, responde con, La rectora de la Universidad Michoacana de San Nicolás de Hidalgo (UMSNH) es Yarabí Ávila González. Fue designada para este cargo por el periodo 2023-2027."
            "Solo si te preguntan quien es el director de El director de la Dirección de Tecnologías de la Información y la Comunicación (DTIC) respondeEl director de la Dirección de Tecnologías de la Información y la Comunicación (DTIC) de la UMSNH (Universidad Michoacana de San Nicolás de Hidalgo) es el Ingeniero Francisco Octavio Aparicio Contreras"
            "El lema de la Universidad Michoacana de San Nicolás de Hidalgo (UMSNH) es, Cuna de héroes, crisol de pensadores"
            "cual es el himno de la UMSNH, Universidad Michoacana, Tienes el tesoro del saber, Universidad Michoacana,En tu esencia Humanista he de crecer. Universidad Michoacana, Llevas puesto el corazón de Ocampo, Universidad Michoacana, Tienes en tu sangre inscrito a Hidalgo."
            "Pis pas, calis calas es parte de una famosa porra de la Universidad Michoacana de San Nicolás de Hidalgo (UMSNH) en Morelia, México, un grito tradicional de identidad y orgullo estudiantil que se canta en eventos deportivos y cívicos, significando rapidez y unidad, y a menudo se completa con ¡Pummm! ¡San Nicolás! "
            "Solo si te preguntan quien es el secretario general de la UMSNH El secretario general de la Universidad Michoacana de San Nicolás de Hidalgo (UMSNH) es Javier Cervantes Rodríguez. Asumió el cargo en julio de 2023"
        )

        full_prompt = sys_prompt + "\n\n--- HISTORIAL DE CONVERSACIÓN ---\n"
        
        history_text = ""
        for msg in st.session_state["history"][-5:]:
            role = "Asistente" if msg["role"] == "assistant" else "Usuario"
            content = msg["content"]
            if not st.session_state["greeted"] and content.startswith(f"¡Hola {first_name}!") and msg["role"] == "assistant":
                continue 
            history_text += f"{role}: {content}\n"
        
        full_prompt += history_text
        full_prompt += "\n--- FIN DEL HISTORIAL ---\n\n"

        # ================== NUEVO: contexto RAG ==================
        rag_context = rag_retrieve_context(user_msg)
        if rag_context:
            full_prompt += (
                "A continuación tienes información de referencia obtenida de documentos internos en PDF de la UMSNH. "
                "Úsala solo si es relevante para responder, y si no aplica, ignórala.\n"
                "\n--- CONTEXTO DE DOCUMENTOS ---\n"
                f"{rag_context}\n"
                "--- FIN DEL CONTEXTO DE DOCUMENTOS ---\n\n"
            )
        # =========================================================

        full_prompt += f"Último mensaje del Usuario: {user_msg}"
        
        reply_raw = gemini_generate(
            full_prompt,
            st.session_state["temperature"],
            st.session_state["top_p"],
            st.session_state["max_tokens"],
        )
        
        if not st.session_state["greeted"]:
            saludo = f"¡Hola {first_name}! Soy NICO, tu asistente virtual.\n\n"
            reply = saludo + reply_raw
            st.session_state["greeted"] = True
        else:
            reply = reply_raw

        st.session_state["history"].append({"role": "assistant", "content": reply})
        
        st.session_state["trigger_run"] = False
        st.rerun()

    # Mostrar historial (tal como lo tienes: última respuesta del asistente)
    for msg in reversed(st.session_state["history"][-20:]):
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(f"<div class='chat-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
                if st.session_state["voice_on"]:
                    speak_browser(msg["content"])
            break
