from flask import Flask, render_template, request
import pandas as pd
from src.ingest import fetch_news
from src.sentiment import analyze_sentiment_batch
from src.charts import generate_distribution_chart
from src.train import predict_market_trend
from src.rag import build_rag_context, generate_rag_summary
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route("/")
def index():
    query = request.args.get("q", "NIFTY 50")
    try:
        # Step 1: Ingest live financial news via Google News RSS
        news_items = fetch_news(query, max_results=20)
        if not news_items:
            return render_template(
                "index.html",
                query=query,
                error=f"No recent news articles found for query '{query}'. Try another ticker."
            )

        # Step 2: Run batch FinBERT transformer inference
        df, metrics = analyze_sentiment_batch(news_items)

        # Format publication timestamps for clean tabular display
        if 'published' in df.columns:
            df['published'] = pd.to_datetime(df['published'], errors='coerce').dt.strftime('%d %b %Y, %H:%M')
            df['published'] = df['published'].fillna('Recent')

        # Step 3: Generate headless in-memory distribution chart
        dist_chart_b64 = None
        try:
            dist_chart_b64 = generate_distribution_chart(df)
        except Exception as chart_err:
            app.logger.error(f"Visualization error: {chart_err}")

        # Step 4: Compute dynamic predictive ML signal using real sentiment polarity
        prediction = None
        try:
            prediction = predict_market_trend(sentiment_score=metrics["net_score"])
        except Exception as pred_err:
            app.logger.error(f"Inference prediction error: {pred_err}")

        # Step 5: RAG Pipeline — Retrieval-Augmented Generation for contextual insights
        rag_result = None
        try:
            rag_store = build_rag_context(news_items, df)
            rag_result = generate_rag_summary(rag_store, query, metrics, top_k=5)
            app.logger.info(
                f"RAG pipeline: {rag_result['rag_metadata'].get('docs_retrieved', 0)} docs retrieved, "
                f"outlook={rag_result['rag_metadata'].get('outlook', 'N/A')}"
            )
        except Exception as rag_err:
            app.logger.error(f"RAG pipeline error: {rag_err}")

        articles = df.to_dict(orient="records")

        # Step 6: Render View template with RAG-enhanced context
        return render_template(
            "index.html",
            query=query,
            articles=articles,
            metrics=metrics,
            dist_chart=dist_chart_b64,
            prediction=prediction,
            rag_result=rag_result
        )

    except Exception as e:
        app.logger.error(f"Unhandled server error: {e}")
        return render_template("index.html", query=query, error=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
