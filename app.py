import streamlit as st
from groq import Groq

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")

st.title("🤖 Groq Chatbot")

# --- Sidebar: API Key ---
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="Paste your key here…")
    model = st.selectbox("🧠 Model", ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"])
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Get a free key at console.groq.com")

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if not api_key:
    st.info("👈 Open the sidebar and paste your Groq API key to start chatting.")
else:
    if prompt := st.chat_input("Type your message…"):

        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get Groq response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    client = Groq(api_key=api_key)
                    response = client.chat.completions.create(
                        model=model,
                        messages=st.session_state.messages
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                except Exception as e:
                    st.error(f"❌ Error: {e}")