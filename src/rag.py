"""
NewsSense RAG (Retrieval-Augmented Generation) Engine
=====================================================
Implements a lightweight RAG pipeline for contextual financial intelligence:

1. INDEXING   – Vectorizes news headlines + sentiment metadata using TF-IDF
2. RETRIEVAL  – Cosine similarity search over the document vector store
3. GENERATION – Synthesizes market intelligence summaries from top-K retrieved docs

Architecture:
    Raw News → Chunking → TF-IDF Vectorization → Vector Store (in-memory)
                                                      ↓
    User Query → Query Embedding → Cosine Similarity → Top-K Retrieval
                                                      ↓
                                    Context Assembly → Summary Generation
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import logging

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    In-memory vector store backed by TF-IDF embeddings.
    Stores document texts alongside their metadata (sentiment, scores, links)
    and supports fast cosine-similarity retrieval.
    """

    def __init__(self):
        self.documents = []       # List of raw document strings
        self.metadata = []        # Parallel list of metadata dicts
        self.vectorizer = None    # Fitted TF-IDF vectorizer
        self.doc_vectors = None   # TF-IDF matrix (n_docs × n_features)

    def add_documents(self, texts, metadata_list=None):
        """
        Ingest documents into the store.

        Args:
            texts: List of document strings (headlines, summaries, etc.)
            metadata_list: Optional parallel list of metadata dicts per doc.
        """
        if not texts:
            return

        self.documents.extend(texts)
        if metadata_list:
            self.metadata.extend(metadata_list)
        else:
            self.metadata.extend([{} for _ in texts])

    def build_index(self):
        """
        Build TF-IDF vector index over all ingested documents.
        Uses sublinear TF scaling and English stop-word removal
        for better financial domain performance.
        """
        if not self.documents:
            logger.warning("No documents to index.")
            return

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),       # Unigrams + bigrams for phrase capture
            sublinear_tf=True,        # Logarithmic TF scaling
            min_df=1,
            max_df=0.95
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        logger.info(f"RAG index built: {len(self.documents)} docs, {self.doc_vectors.shape[1]} features")

    def retrieve(self, query, top_k=5):
        """
        Retrieve the top-K most relevant documents for a given query.

        Args:
            query: Natural language query string
            top_k: Number of documents to retrieve

        Returns:
            List of dicts with 'text', 'metadata', and 'similarity_score'
        """
        if self.vectorizer is None or self.doc_vectors is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        # Get top-K indices sorted by descending similarity
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.0:  # Only return non-zero matches
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "similarity_score": round(float(similarities[idx]), 4)
                })

        return results

    @property
    def size(self):
        return len(self.documents)


def build_rag_context(news_items, sentiment_df):
    """
    Build RAG document store from ingested news articles enriched with
    sentiment analysis results.

    Args:
        news_items: List of dicts from ingest.py (title, link, published)
        sentiment_df: DataFrame from sentiment.py with sentiment labels and scores

    Returns:
        DocumentStore instance ready for retrieval queries
    """
    store = DocumentStore()

    documents = []
    metadata_list = []

    for i, item in enumerate(news_items):
        # Build enriched document string combining headline + sentiment context
        title = item.get("title", "")
        sentiment_label = ""
        sentiment_score = 0.0
        polarity = 0.0

        if sentiment_df is not None and i < len(sentiment_df):
            row = sentiment_df.iloc[i]
            sentiment_label = str(row.get("sentiment", ""))
            sentiment_score = float(row.get("score", 0.0))
            polarity = float(row.get("polarity", 0.0))

        # Create a rich document representation for better retrieval
        doc_text = f"{title} [Sentiment: {sentiment_label}, Confidence: {sentiment_score:.2f}]"
        documents.append(doc_text)

        metadata_list.append({
            "title": title,
            "raw_title": item.get("raw_title", title),
            "link": item.get("link", ""),
            "published": item.get("published", ""),
            "sentiment": sentiment_label,
            "score": sentiment_score,
            "polarity": polarity
        })

    store.add_documents(documents, metadata_list)
    store.build_index()

    return store


def generate_rag_summary(store, query, metrics, top_k=5):
    """
    RAG Generation step: Retrieves relevant documents and synthesizes
    a contextual market intelligence summary.

    This implements the 'Generation' part of RAG without requiring
    an external LLM — it uses rule-based summarization over retrieved
    context to produce actionable market insights.

    Args:
        store: DocumentStore with indexed documents
        query: User's search query (e.g., "RELIANCE")
        metrics: Sentiment metrics dict from analyze_sentiment_batch
        top_k: Number of documents to retrieve for context

    Returns:
        Dict containing:
            - summary: Generated textual market intelligence summary
            - retrieved_docs: The actual retrieved documents used
            - rag_metadata: Pipeline metadata (retrieval stats, etc.)
    """
    if store is None or store.size == 0:
        return {
            "summary": "Insufficient data for RAG-based analysis.",
            "retrieved_docs": [],
            "rag_metadata": {"status": "no_documents"}
        }

    # RETRIEVAL: Fetch most relevant documents
    retrieved = store.retrieve(query, top_k=top_k)

    if not retrieved:
        return {
            "summary": f"No relevant context found for '{query}'.",
            "retrieved_docs": [],
            "rag_metadata": {"status": "no_matches"}
        }

    # GENERATION: Synthesize market intelligence from retrieved context
    # Analyze sentiment distribution in retrieved docs
    sentiments = [doc["metadata"].get("sentiment", "") for doc in retrieved]
    sentiment_counts = Counter(sentiments)
    dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "Neutral"

    avg_similarity = np.mean([doc["similarity_score"] for doc in retrieved])
    avg_confidence = np.mean([doc["metadata"].get("score", 0.5) for doc in retrieved])
    polarities = [doc["metadata"].get("polarity", 0.0) for doc in retrieved]
    avg_polarity = np.mean(polarities) if polarities else 0.0

    # Build contextual summary
    net_score = metrics.get("net_score", 0.5)
    total = metrics.get("total", 0)

    # Determine market outlook from retrieved context
    if net_score > 0.6:
        outlook = "strongly bullish"
        action_hint = "Positive momentum detected — sentiment is skewed towards optimism across retrieved headlines."
    elif net_score > 0.52:
        outlook = "mildly bullish"
        action_hint = "A slight positive bias exists in the news flow, though conviction is moderate."
    elif net_score < 0.4:
        outlook = "strongly bearish"
        action_hint = "Negative sentiment dominates the recent news cycle — caution is warranted."
    elif net_score < 0.48:
        outlook = "mildly bearish"
        action_hint = "Slight negative bias in the retrieved context. Monitor for trend reversal signals."
    else:
        outlook = "neutral"
        action_hint = "The market sentiment is balanced. No strong directional bias detected in recent headlines."

    # Extract key themes from top headlines
    key_headlines = [doc["metadata"].get("title", "") for doc in retrieved[:3]]
    themes_text = " | ".join(key_headlines) if key_headlines else "N/A"

    summary = (
        f"📊 RAG-Enhanced Market Intelligence for '{query}'\n\n"
        f"Based on retrieval of {len(retrieved)} most relevant articles from {total} indexed headlines, "
        f"the overall market sentiment is {outlook} (Net Polarity Index: {net_score:.4f}).\n\n"
        f"🔍 Key Retrieved Themes: {themes_text}\n\n"
        f"The dominant tone across retrieved context is '{dominant_sentiment}' "
        f"with average retrieval confidence of {avg_confidence:.1%} and "
        f"mean cosine similarity of {avg_similarity:.4f}.\n\n"
        f"💡 Insight: {action_hint}\n\n"
        f"⚠️ This analysis is generated via RAG (Retrieval-Augmented Generation) pipeline. "
        f"It retrieves the most semantically relevant headlines, enriches them with FinBERT "
        f"sentiment scores, and synthesizes directional market intelligence. "
        f"Not financial advice."
    )

    rag_metadata = {
        "status": "success",
        "query": query,
        "docs_indexed": store.size,
        "docs_retrieved": len(retrieved),
        "avg_similarity": round(avg_similarity, 4),
        "avg_confidence": round(avg_confidence, 4),
        "avg_polarity": round(avg_polarity, 4),
        "dominant_sentiment": dominant_sentiment,
        "outlook": outlook,
        "retrieval_method": "TF-IDF + Cosine Similarity",
        "generation_method": "Rule-Based Contextual Synthesis",
        "vector_dimensions": store.doc_vectors.shape[1] if store.doc_vectors is not None else 0
    }

    return {
        "summary": summary,
        "retrieved_docs": retrieved,
        "rag_metadata": rag_metadata
    }
