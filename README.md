# 🧠 MindPaper AI

**MindPaper AI** is an intelligent, offline-capable document assistant built using **Streamlit**, **LangChain**, and **Ollama**. It allows users to upload PDF documents, understand their content through smart chunking and embeddings, and interactively ask questions — all powered by local LLMs.

---

## 📸 Preview

<img width="1900" height="1088" alt="Screenshot 2025-07-19 144542" src="https://github.com/user-attachments/assets/c6cdbdf0-b824-4157-91b3-5958ed8d78b9" />



---

## 🚀 Features

### ✅ Assistant Capabilities
- 📄 **Understands and processes PDF documents**
- 🔍 **Finds relevant content** using smart semantic search
- 🤖 **Answers questions** with context-aware AI reasoning
- ⚡ **Runs fully offline** using local LLMs via Ollama
- 🧠 Uses custom prompt templates for accurate and concise answers
- 🖥️ Clean, dark-themed UI with live chat and file uploader

---

## 🧰 Tech Stack

- 🐍 Python
- ⚙️ Streamlit (UI)
- 🔗 LangChain (RAG, document processing)
- 🤖 Ollama (Local LLM backend)
- 🧠 DeepSeek / LLaMA3 / Nomic Embeddings (model-agnostic design)
- 📄 PDFPlumber (for document reading)

---

## 🛠️ Installation


```bash
1. **Clone the repository**
git clone https://github.com/your-username/mindpaper-ai.git
cd mindpaper-ai

2. **Create a virtual environment**

conda create -n mindpaper python=3.11
conda activate mindpaper

3. ** Install dependencies**

pip install -r requirements.txt

4. **Install and pull Ollama models
**

ollama pull deepseek-llm:1.5b   
ollama pull llama3.2:latest    

5. **Run the application**

streamlit run app.py



