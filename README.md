# 📰 NewsSense

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-black?logo=flask)](https://flask.palletsprojects.com/)
[![Transformers](https://img.shields.io/badge/HuggingFace-FinBERT-yellow?logo=huggingface)](https://huggingface.co/yiyanghkust/finbert-tone)
[![RAG](https://img.shields.io/badge/RAG-TF--IDF%20Pipeline-purple)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**NewsSense** is an AI-powered financial sentiment analysis and market intelligence dashboard with **RAG (Retrieval-Augmented Generation)** for Indian Equities (NSE/BSE). It fuses real-time financial news ingestion with domain-specific NLP (**FinBERT**), a RAG pipeline for contextual intelligence, and historical stock market data to generate directional sentiment signals and market insights.

---

## 🔑 Key Features

* **Real-Time News Ingestion:** Streaming headline extraction across Indian equities and market benchmarks via Google News RSS with strict 7-day temporal filtering.
* **Domain-Specific Financial NLP:** Employs `yiyanghkust/finbert-tone`, a transformer pre-trained on corporate filings and the Financial PhraseBank to resolve financial polysemy and nuances.
* **RAG Pipeline (Retrieval-Augmented Generation):** Indexes all ingested headlines into a TF-IDF vector store, performs semantic retrieval via cosine similarity, and generates contextual market intelligence summaries from top-K retrieved documents.
* **In-Memory Headless Visualization:** Generates publication-ready sentiment distribution charts via Matplotlib's `Agg` engine, piped to Base64 data URIs for zero-disk I/O and stateless concurrency.
* **Predictive Market Modeling:** Combines batch sentiment polarity with historical OHLCV candles to forecast intraday market direction (`Close > Open`) without lookahead bias.
* **Containerized Microservice:** Fully containerized with CPU-optimized PyTorch wheels, layer-cached Docker builds, and persistent volume bind mounts.
* **Regulatory Compliance:** Built-in SEBI regulatory disclaimer adhering to educational and technical research standards.

---

## 🏗️ System Architecture & RAG Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│                    USER INPUT (e.g. "RELIANCE")                      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Flask Controller (app.py)                            │
│              Orchestrates the 6-step pipeline                        │
└────┬──────────────┬──────────────┬──────────────┬───────────────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌─────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────────────┐
│  STEP 1 │  │  STEP 2   │  │ STEP 3   │  │ STEP 4              │
│ ingest  │  │ sentiment │  │ charts   │  │ train (ML Predict)   │
│ .py     │  │ .py       │  │ .py      │  │ .py                  │
│         │  │           │  │          │  │                      │
│ Google  │  │ FinBERT   │  │ Base64   │  │ Logistic Regression  │
│ News    │  │ Inference │  │ PNG      │  │ Inference            │
│ RSS     │  │ Pipeline  │  │ In-Mem   │  │ (predict_market)     │
└────┬────┘  └─────┬─────┘  └──────────┘  └──────────────────────┘
     │             │
     │             │
     ▼             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    STEP 5: RAG PIPELINE (rag.py)                     │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐   │
│  │  INDEXING    │    │   RETRIEVAL      │    │   GENERATION      │   │
│  │             │    │                  │    │                   │   │
│  │ News + NLP  │───▶│ Query → TF-IDF  │───▶│ Top-K Context     │   │
│  │ → TF-IDF    │    │ → Cosine Sim    │    │ → Market Intel    │   │
│  │ → Vector    │    │ → Top-K Docs    │    │ → Summary         │   │
│  │   Store     │    │                  │    │                   │   │
│  └─────────────┘    └──────────────────┘    └───────────────────┘   │
│                                                                      │
│  RAG Metadata: docs_indexed, docs_retrieved, avg_similarity,         │
│  vector_dimensions, dominant_sentiment, market_outlook                │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 STEP 6: Jinja2 HTML5 Dashboard                       │
│                                                                      │
│   ┌─────────────┐  ┌───────────────┐  ┌───────────────────────┐     │
│   │ Sentiment   │  │ ML Prediction │  │ 🧠 RAG Intelligence   │     │
│   │ Metrics +   │  │ Signal +      │  │ Panel: Summary +      │     │
│   │ Chart       │  │ Probabilities │  │ Retrieved Docs +      │     │
│   │             │  │               │  │ Pipeline Metadata     │     │
│   └─────────────┘  └───────────────┘  └───────────────────────┘     │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Headlines Table: All FinBERT-classified articles            │   │
│   └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Deep Dive

```
                    ┌────────────────────┐
                    │   Raw News Items   │
                    │   (from ingest.py) │
                    └────────┬───────────┘
                             │
                    ┌────────▼───────────┐
                    │  Sentiment-Enriched │
                    │  Documents          │
                    │  "headline [Sent:   │
                    │   Positive, 0.94]"  │
                    └────────┬───────────┘
                             │
               ┌─────────────▼─────────────┐
               │      TF-IDF Vectorizer     │
               │  • max_features=5000       │
               │  • ngram_range=(1,2)       │
               │  • sublinear_tf=True       │
               │  • English stop words      │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │   In-Memory Vector Store   │
               │   (DocumentStore class)    │
               │                            │
               │  doc_vectors: CSR Matrix   │
               │  metadata[]: parallel list │
               └─────────────┬─────────────┘
                             │
                   User Query│
                             │
               ┌─────────────▼─────────────┐
               │    Cosine Similarity       │
               │    Search (Top-K=5)        │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │  Context Assembly          │
               │  • Sentiment Distribution  │
               │  • Polarity Analysis       │
               │  • Market Outlook          │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │  Generated Summary         │
               │  + RAG Metadata            │
               │  → Rendered to Dashboard   │
               └───────────────────────────┘
```

---

## 📁 Repository Structure

```
NewsSense/
├── app.py                  # Main Flask application & MVC controller (6-step pipeline)
├── run_pipeline.py         # Offline ML training & dataset construction pipeline
├── requirements.txt        # Pinned Python package dependencies
├── Dockerfile              # Lightweight CPU-optimized container specification
├── docker-compose.yml      # Container orchestration with volume mounts
├── models/
│   └── sentiment_model.pkl # Serialized Logistic Regression artifact
├── src/
│   ├── __init__.py
│   ├── ingest.py           # Google News RSS ingestion & title sanitization
│   ├── sentiment.py        # FinBERT transformer inference & polarity scoring
│   ├── rag.py              # RAG pipeline: TF-IDF indexing, retrieval, generation
│   ├── dataset.py          # Yahoo Finance OHLCV ingestion & feature engineering
│   ├── train.py            # Model training & calibrated probability inference
│   └── charts.py           # Headless Matplotlib in-memory rasterization
└── templates/
    └── index.html          # Modern fintech-styled responsive dashboard with RAG panel
```

---

## 🚀 Quick Start Guide

### Option 1: Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/NewsSense.git
   cd NewsSense
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate       # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the predictive baseline model:**
   ```bash
   python run_pipeline.py
   ```

5. **Launch the Flask dashboard:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://localhost:5000`.

---

### Option 2: Running with Docker Compose

```bash
docker compose up --build
```
The application will be accessible at `http://localhost:5000` with hot code reloading enabled.

---

## 🧠 RAG Implementation Details

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document Indexing** | TF-IDF (scikit-learn) | Vectorize sentiment-enriched headlines into sparse feature vectors |
| **Vector Store** | In-memory CSR Matrix | Store and retrieve document embeddings without external DB |
| **Retrieval** | Cosine Similarity | Rank documents by semantic relevance to user query |
| **Generation** | Rule-based Synthesis | Produce market intelligence summaries from retrieved context |
| **Enrichment** | FinBERT Metadata | Augment each document with sentiment labels, scores, polarity |

### How the RAG Pipeline Works:

1. **Indexing Phase:** After FinBERT processes all headlines, each article is enriched with its sentiment label and confidence score, then vectorized using TF-IDF with bigram support.
2. **Retrieval Phase:** When a user searches for a stock/sector, the query is vectorized using the same TF-IDF model and compared against all indexed documents via cosine similarity.
3. **Generation Phase:** The top-K (default: 5) most relevant documents are assembled into context. Sentiment distribution, polarity analysis, and market outlook are computed and synthesized into a natural-language market intelligence summary.

---

## 🔧 Tech Stack & Tools

* **Backend & Web:** Python 3.10+, Flask, Jinja2
* **NLP & Deep Learning:** Hugging Face Transformers, PyTorch, FinBERT (`yiyanghkust/finbert-tone`)
* **RAG Pipeline:** TF-IDF Vectorization, Cosine Similarity Retrieval, Contextual Generation
* **Machine Learning & Analytics:** Scikit-Learn (Logistic Regression, TfidfVectorizer), Pandas, NumPy
* **Data Sources:** Google News RSS (`feedparser`), Yahoo Finance API (`yfinance`)
* **Visualization:** Matplotlib (`Agg` backend, Base64 encoding)
* **DevOps:** Docker, Docker Compose

---

## ⚠️ Regulatory Notice (SEBI Disclaimer)

This system is an academic and technical demonstration of NLP, RAG, and machine learning applied to financial datasets. The sentiment analysis, RAG-generated insights, and model predictions provided do not constitute registered investment or financial advice under the Securities and Exchange Board of India (SEBI) Investment Advisers Regulations, 2013.

---

## 👤 Author

**Deepak**  
AI / Data Science Engineer
