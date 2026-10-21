from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from src.ingest import fetch_news
from src.sentiment import analyze_sentiment_batch
from src.charts import generate_distribution_chart
from src.train import predict_market_trend
from src.rag import build_rag_context, generate_rag_summary
from src.database import register_or_login_user, init_db
import logging
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "newssense-production-secret-key-2026")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure SQLite database is initialized
init_db()

@app.route("/")
def home():
    """
    Landing Page & Dashboard Route:
    - If user visits root '/' without ?q=, serve the 3D aesthetic landing page.
    - If user provides ?q= (e.g. '/?q=RELIANCE'), render the full financial intelligence dashboard.
    """
    q_param = request.args.get("q")
    if not q_param:
        return render_template("landing.html")

    return render_dashboard(q_param)

@app.route("/dashboard")
def dashboard():
    """Direct dashboard endpoint (e.g. /dashboard?q=RELIANCE)"""
    query = request.args.get("q", "RELIANCE")
    return render_dashboard(query)

@app.route("/auth")
def auth_page():
    """Dedicated aesthetic authentication gateway (Guest or Email/Password)"""
    return render_template("auth.html")

@app.route("/auth/guest")
def auth_guest():
    """1-Click Guest sign in: initializes session and redirects to /?q=RELIANCE"""
    session["user"] = "Guest Trader"
    session["role"] = "Guest"
    session["is_authenticated"] = True
    app.logger.info("Guest session initialized. Redirecting to real-time dashboard.")
    return redirect(url_for("home", q="RELIANCE"))

@app.route("/auth/login", methods=["GET", "POST"])
def auth_login():
    """
    Sign up / Login with email & password:
    Persists real user credentials (hashed) in SQLite database data/users.db,
    then launches real-time dashboard.
    """
    if request.method == "GET":
        return redirect(url_for("auth_page"))

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()

    if not email or not password:
        return render_template("auth.html", error="Please enter both email and password.")

    success, user_data, msg = register_or_login_user(email, password)
    if not success:
        return render_template("auth.html", error=msg)

    # Set session for registered trader
    username = email.split("@")[0].replace(".", " ").title()
    session["user"] = username
    session["email"] = email
    session["role"] = "Registered Trader"
    session["is_authenticated"] = True
    app.logger.info(f"User {email} authenticated in DB. Launching real-time dashboard.")

    return redirect(url_for("home", q="RELIANCE"))

@app.route("/logout")
def logout():
    """Clears authentication session and returns to landing page"""
    session.clear()
    return redirect(url_for("home"))

def render_dashboard(query):
    """Core dashboard execution pipeline: Ingest -> FinBERT -> Chart -> ML Signal -> RAG"""
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

        # Step 3: Generate headless in-memory distribution chart (dark theme)
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
