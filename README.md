🎓 AI Academic Agent — RAG-Powered Intelligent Tutoring System

An end-to-end Retrieval-Augmented Generation (RAG) based Academic Assistant designed for universities — enabling personalized learning, misconception detection, student mastery modeling, teacher analytics, and closed-loop improvement.

🚀 Project Overview

The AI Academic Agent is an intelligent education platform that ingests faculty materials (PDFs), performs semantic retrieval using FAISS + Sentence Transformers, and generates adaptive responses via an LLM.

Beyond simple Q&A, the system models student mastery, detects misconceptions, provides adaptive explanations, identifies at-risk students for teachers, enforces academic integrity, and continuously improves via closed-loop feedback.

This project was developed as part of a hackathon education problem statement and follows a full Machine Learning production workflow.

🧠 Core Capabilities
✅ RAG-Based Knowledge System

PDF ingestion + chunking

Vector embeddings (Sentence Transformers, GPU accelerated)

FAISS semantic search

Context-aware LLM responses

✅ Student Mastery Modeling

Tracks each student’s understanding per topic:

Mastered

Weak

Unknown

Stored in JSON for persistence.

✅ Misconception Detection

Automatically identifies conceptual misunderstandings using:

Topic mapping

Response analysis

Pattern detection

✅ Adaptive Tutoring

Generates explanations based on:

Student mastery level

Detected misconceptions

Difficulty adaptation

✅ Teacher Analytics Dashboard

Provides:

Students marked CRITICAL / AT RISK

Topic-wise misconceptions

Learning recommendations

✅ Academic Integrity Guardrails

Prevents:

Direct answer dumping

Cheating behavior

Exam-style exploitation

Promotes guided learning instead.

✅ Closed-Loop Learning System

Feedback from each interaction updates:

Student mastery

Misconceptions

Risk level

Recommendations

Creating a self-improving educational pipeline.

🏗️ System Architecture
PDFs
 ↓
Ingestion → Chunking → Embeddings → FAISS
 ↓
User Query
 ↓
Retriever
 ↓
LLM Generator
 ↓
Adaptive Explanation
 ↓
Misconception Detection
 ↓
Student Mastery Update
 ↓
Teacher Analytics
 ↓
Closed Loop Feedback


📂 Project Structure
AI-Academic-Agent/
│
├── ingestion.py                 # PDF ingestion + chunking + FAISS + embeddings
│
├── ragQuery/
│   └── ragQuery.py             # Main pipeline loop
│
├── models/
│   ├── adaptiveAnswer.py
│   ├── misconceptionDetector.py
│   ├── topicMapper.py
│   ├── studentModel.py
│   ├── teacherAnalytics.py
│   ├── integrityGuard.py
│   └── llm.py
│
├── data/
│   └── students.json           # Mastery tracking
│
├── requirements.txt
└── README.md

🛠️ Tech Stack
Programming Language

Python 3.11

Machine Learning & NLP

PyTorch

Sentence Transformers

Hugging Face Transformers

FAISS (Facebook AI Similarity Search)

LLM Integration

OpenRouter API

Data Processing

NumPy

Pandas

Vector Storage

FAISS Vector Database

Environment Management

Python Virtual Environment

1.Activate Virtual Environment
torch_gpu\Scripts\activate
2.Install Dependencies
pip install -r requirements.txt
3. Set API Key
Create .env:
OPENROUTER_API_KEY=your_key_here
4. Run Ingestion
python ingestion.py
5. Start Agent
python ragQuery/ragQuery.py
