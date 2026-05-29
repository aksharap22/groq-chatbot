# Handles embedding model loading

from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    """Load embedding model."""
    return SentenceTransformer("BAAI/bge-small-en-v1.5")