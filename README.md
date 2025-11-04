# 🎓 TubeWhiz: The AI-Powered YouTube Learning Assistant

TubeWhiz is a Streamlit application that transforms any educational YouTube video into an interactive learning experience. It downloads the video's audio, transcribes the content, provides detailed summaries, and allows you to chat with an AI assistant (powered by Google's Gemini) to ask questions *strictly* based on the video's content.

This tool is built on a **Retrieval-Augmented Generation (RAG)** pipeline, ensuring the AI's answers are grounded in the provided transcript and not its own general knowledge.

## ✨ Features

* **YouTube Video Processing:** Just paste a YouTube URL to get started.
* **Fast Audio Transcription:** Uses `faster-whisper` for accurate and speedy speech-to-text, with GPU support if available.
* **AI-Powered Summarization:** Leverages the Gemini model to generate clear, chunk-by-chunk summaries of the video content.
* **Conversational Q&A:** Chat with a RAG bot that answers your questions based *only* on the video's transcript.
* **Content-Aware Context:** The bot uses a FAISS vector index to find the most relevant parts of the video to answer your questions accurately.
* **Clean & Simple UI:** Built with Streamlit for an intuitive user experience.
* **File Management:** Includes a simple cleanup function to remove temporary audio files.

## ⚙️ Tech Stack & Pipeline

This project uses a modern AI stack to create its RAG pipeline:

* **Frontend:** Streamlit
* **AI & LLM:** Google Gemini (via `google-generativeai`)
* **Transcription:** `faster-whisper`
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Vector Store:** `faiss` (Facebook AI Similarity Search)
* **Audio Handling:** `yt-dlp` (for downloading) & `pydub` (for splitting)

### How It Works

1.  **Download Audio:** `yt-dlp` downloads the best audio from the provided YouTube URL as an `.mp3` file.
2.  **Split Audio:** `pydub` splits the full audio file into smaller, 5-minute chunks. This improves processing speed and handles long videos.
3.  **Transcribe & Summarize:** Each audio chunk is transcribed by `faster-whisper`. The resulting text for each chunk is then summarized by Gemini.
4.  **Create Vector Index:** The transcribed text chunks (not the summaries) are converted into vector embeddings by `sentence-transformers` and stored in a local FAISS index.
5.  **Chat (RAG):** When you ask a question:
    * Your query is converted into a vector embedding.
    * FAISS performs a similarity search to find the `top_k` (e.g., top 3) most relevant text chunks from the transcript.
    * These relevant chunks, along with your chat history and the new question, are sent to Gemini with a strict prompt to *only* use the provided context.
    * Gemini generates the answer, which is then displayed in the chat.

## 🚀 Setup and Installation

Follow these steps to run TubeWhiz on your local machine.

### 1. Prerequisites

You must have **FFmpeg** installed on your system. This is a crucial dependency for `pydub` and `yt-dlp` to process audio files.

* **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` folder to your system's PATH.
* **macOS (using Homebrew):** `brew install ffmpeg`
* **Linux (using apt):** `sudo apt update && sudo apt install ffmpeg`

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/tubewhiz.git](https://github.com/your-username/tubewhiz.git)
cd tubewhiz
