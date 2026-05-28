"""
🤖 BYOK Chatbot — Bring Your Own LLM Key
Supports: OpenAI, Anthropic Claude, Google Gemini, Groq
"""

import streamlit as st
import os

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="BYOK Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import elegant font */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 100%);
}
section[data-testid="stSidebar"] * {
    color: #e0e0f0 !important;
}

/* Header */
.chat-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.chat-header h1 {
    font-size: 1.8rem;
    font-weight: 600;
    background: linear-gradient(135deg, #6ee7f7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.chat-header p {
    color: #888;
    font-size: 0.85rem;
}

/* Provider badge */
.provider-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6ee7f720, #a78bfa20);
    border: 1px solid #a78bfa40;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    color: #a78bfa;
    font-family: 'DM Mono', monospace;
    margin-top: 0.3rem;
}

/* Chat messages */
.stChatMessage {
    border-radius: 12px !important;
    margin-bottom: 0.5rem;
}

/* Clear button */
.stButton > button {
    background: linear-gradient(135deg, #6ee7f710, #a78bfa10);
    border: 1px solid #a78bfa50;
    color: #a78bfa;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6ee7f730, #a78bfa30);
    border-color: #a78bfa;
}

/* Warning / info boxes */
.setup-box {
    background: linear-gradient(135deg, #fef9c340, #fde68a20);
    border: 1px solid #fbbf2460;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #92400e;
    font-size: 0.85rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: call the right LLM ──────────────────────────────────────────────
def get_response(provider: str, api_key: str, model: str,
                 messages: list, system_prompt: str) -> str:
    """Route the chat history to the chosen LLM and return the assistant reply."""

    # ── OpenAI ──────────────────────────────────────────────────────────────
    if provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        payload = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=model, messages=payload, stream=False
        )
        return response.choices[0].message.content

    # ── Anthropic Claude ─────────────────────────────────────────────────────
    elif provider == "Anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    # ── Google Gemini ────────────────────────────────────────────────────────
    elif provider == "Google Gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
        )
        # Convert to Gemini history format
        history = []
        for m in messages[:-1]:
            history.append({
                "role": "user" if m["role"] == "user" else "model",
                "parts": [m["content"]],
            })
        chat = gemini.start_chat(history=history)
        response = chat.send_message(messages[-1]["content"])
        return response.text

    # ── Groq ─────────────────────────────────────────────────────────────────
    elif provider == "Groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        payload = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=model, messages=payload
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    provider = st.selectbox(
        "🔌 LLM Provider",
        ["OpenAI", "Anthropic", "Google Gemini", "Groq"],
        help="Choose which LLM API to use.",
    )

    # Model choices per provider
    MODEL_OPTIONS = {
        "OpenAI":         ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "Anthropic":      ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001", "claude-opus-4-20250514"],
        "Google Gemini":  ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "Groq":           ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
    }
    model = st.selectbox("🧠 Model", MODEL_OPTIONS[provider])

    api_key = st.text_input(
        "🔑 API Key",
        type="password",
        placeholder="Paste your API key here…",
        help="Your key is never stored — it only lives in this browser session.",
    )

    st.markdown("---")

    system_prompt = st.text_area(
        "📝 System Prompt",
        value="You are a helpful, friendly, and concise assistant.",
        height=100,
        help="Set the assistant's personality and instructions.",
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Key links
    KEY_LINKS = {
        "OpenAI":        "https://platform.openai.com/api-keys",
        "Anthropic":     "https://console.anthropic.com/settings/keys",
        "Google Gemini": "https://aistudio.google.com/app/apikey",
        "Groq":          "https://console.groq.com/keys",
    }
    st.markdown(f"[🔗 Get {provider} API Key]({KEY_LINKS[provider]})",
                unsafe_allow_html=False)

    st.markdown("---")
    st.caption("💡 Your API key stays in this session only and is never saved.")


# ── Main chat UI ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="chat-header">
  <h1>🤖 BYOK Chatbot</h1>
  <p>Bring Your Own LLM Key — pick a provider, paste your key, start chatting.</p>
  <span class="provider-badge">{provider} · {model}</span>
</div>
""", unsafe_allow_html=True)

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Gate: require API key
if not api_key:
    st.markdown("""
    <div class="setup-box">
        👈 <strong>To get started:</strong> Open the sidebar, choose your provider,
        paste your API key, and type a message below.
    </div>
    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your message…", disabled=not api_key):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get & show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = get_response(
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    messages=st.session_state.messages,
                    system_prompt=system_prompt,
                )
                st.markdown(reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as e:
                error_msg = str(e)
                if "api_key" in error_msg.lower() or "authentication" in error_msg.lower() or "401" in error_msg:
                    st.error("❌ Invalid API key. Please check the key in the sidebar.")
                elif "model" in error_msg.lower():
                    st.error(f"❌ Model error: {error_msg}")
                else:
                    st.error(f"❌ Error: {error_msg}")
