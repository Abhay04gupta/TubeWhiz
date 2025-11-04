import os
import streamlit as st
import torch
import numpy as np
import faiss
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# CONFIGURATION
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

if os.makedirs("audio", exist_ok=True) is None:
    os.makedirs("audio", exist_ok=True)
if os.makedirs("chunks", exist_ok=True) is None:
    os.makedirs("chunks", exist_ok=True)

# CACHE MODELS
@st.cache_resource
def load_models():
    whisper = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    gemini = genai.GenerativeModel("gemini-2.0-flash")
    return whisper, embedder, gemini


# SUMMARIZATION FUNCTION
def summarize_text(gemini, text):
    prompt = f"Provide a detailed, clear summary of the following transcript:\n\n{text}"
    response = gemini.generate_content(prompt)
    return response.text.strip()


# STEP-1: DOWNLOAD AUDIO
def download_audio(youtube_url, output_file="./audio/audio_file"):
    os.makedirs("audio", exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'noplaylist': True,
        'quiet': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        duration = info.get("duration", None)
        st.info(f"Audio downloaded — Duration: {duration//60 if duration else 'Unknown'} minutes")
    return output_file + ".mp3"


# STEP 2: SPLIT LONG AUDIO
def split_audio(file_path, chunk_length_ms=5 * 60 * 1000):  # 5 minutes
    audio = AudioSegment.from_file(file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    os.makedirs("chunks", exist_ok=True)
    chunk_files = []

    for i, chunk in enumerate(chunks):
        chunk_path = f"chunks/chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunk_files.append(chunk_path)

    st.info(f"Audio Processing Completed.")
    return chunk_files


# STEP 3: TRANSCRIBE AND SUMMARIZE EACH CHUNK 
def transcribe_and_summarize_chunks(whisper, gemini, chunk_files):
    full_transcription = ""
    summaries = []
    text_chunks = []

    progress = st.progress(0)
    for i, chunk in enumerate(chunk_files):
        segments, _ = whisper.transcribe(chunk, beam_size=3)
        chunk_text = " ".join([seg.text for seg in segments]).strip()
        text_chunks.append(chunk_text)

        full_transcription += chunk_text + " "

        chunk_summary = summarize_text(gemini, chunk_text)
        summaries.append(f"# ✳️ Summary for chunk {i+1}\n{chunk_summary}\n")

        progress.progress((i + 1) / len(chunk_files))
    progress.empty()

    st.info("Transcription and summarization completed.")
    return full_transcription.strip(), "\n".join(summaries).strip(), text_chunks


# STEP 4: BUILD FAISS INDEX 
@st.cache_resource
def build_faiss_index(_embedder, text_chunks):
    embeddings = _embedder.encode(text_chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# STEP 5: RETRIEVE RELEVANT CONTEXT 
def retrieve_relevant_chunks(embedder, query, index, text_chunks, top_k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    return [text_chunks[i] for i in indices[0]]


# STEP 6: GEMINI CONVERSATIONAL CHAT 
def chat_with_gemini(gemini, embedder, index, text_chunks):
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    user_input = st.chat_input("Ask something related to the video...")
    if user_input:
        relevant_texts = retrieve_relevant_chunks(embedder, user_input, index, text_chunks)
        context = "\n".join(relevant_texts)
        history = "\n".join(st.session_state.conversation_history[-6:])

        prompt = f"""
        You are an educational assistant that answers questions strictly based on the given transcript.

        INSTRUCTIONS:
        - Use ONLY the information from the provided CONTEXT and CHAT HISTORY.
        - Do NOT use any outside knowledge.
        - If the answer cannot be found in the context, clearly say "I don’t have enough information to answer that."

        Context:
        {context}

        Chat History:
        {history}

        User: {user_input}
        Assistant:
        """

        with st.spinner("Thinking..."):
            response = gemini.generate_content(prompt)
        reply = response.text.strip()

        st.session_state.conversation_history.append(f"User: {user_input}")
        st.session_state.conversation_history.append(f"Assistant: {reply}")

    for msg in st.session_state.conversation_history:
        if msg.startswith("User:"):
            st.chat_message("user").markdown(msg.replace("User:", "").strip())
        else:
            st.chat_message("assistant").markdown(msg.replace("Assistant:", "").strip())


# STEP 7: CLEANUP 
def cleanup_files(audio_path="./audio/audio_file.mp3", chunks_dir="chunks"):
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(chunks_dir):
            for file_name in os.listdir(chunks_dir):
                os.remove(os.path.join(chunks_dir, file_name))
        st.success("Cleanup complete (folders kept).")
    except Exception as e:
        st.error(f"Cleanup error: {e}")


# MAIN PIPELINE
def main():
    st.set_page_config(page_title="EduWhiz - AI Learning Assistant", page_icon="🎓", layout="wide")

    # Sidebar section
    with st.sidebar:
        st.title("🎓 TubeWhiz")
        st.markdown("##### Your AI-powered Educational Assistant")
        st.divider()

        st.markdown("**🔗 Enter a YouTube URL**")
        url = st.text_input("YouTube Link:", placeholder="Paste your video link here...")

        st.markdown("**⚙️ Actions**")
        start_btn = st.button("🚀 Start Processing")
        cleanup_btn = st.button("🧹 Clear Temporary Files")

        st.divider()
        st.caption("💡 TubeWhiz extracts, summarizes, and lets you chat about any educational video.")

    # Main chat area
    st.title("📚 TubeWhiz - Conversational Learning Bot")
    st.markdown("#### _Ask, Learn, and Explore educational content from YouTube effortlessly._")
    st.markdown("---")

    whisper, embedder, gemini = load_models()

    if start_btn:
        if url:
            with st.spinner("⬇️ Downloading audio..."):
                audio_path = download_audio(url)
            
            with st.spinner("🎧 Audio Processing..."):   
                chunk_files = split_audio(audio_path)
            
            with st.spinner("📝 Transcribing and summarizing content..."):    
                transcribed_text, summarized_text, text_chunks = transcribe_and_summarize_chunks(
                    whisper, gemini, chunk_files
                )

            index = build_faiss_index(embedder, text_chunks)
            st.session_state.index = index
            st.session_state.text_chunks = text_chunks

            st.success("✅ Video processed successfully!")

            # Display Summary Before Chat
            st.markdown("### 🧾 Video Summary")
            st.markdown(summarized_text)

            st.markdown("---")
            st.markdown("### 💬 Chat with TubeWhiz")
            st.markdown("_Ask any question related to your video content below:_")
            chat_with_gemini(gemini, embedder, st.session_state.index, st.session_state.text_chunks)
        else:
            st.warning("⚠️ Please enter a valid YouTube URL in the sidebar.")

    # Cleanup button
    if cleanup_btn:
        cleanup_files()


if __name__ == "__main__":
    main()
