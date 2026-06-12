from flask import Flask, render_template_string, request
import pandas as pd
from src.ingest import fetch_news
from src.sentiment import analyze_sentiment_batch
from src.charts import generate_distribution_chart
from src.train import load_model_and_predict
import os

app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NewsSense - Financial Sentiment Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background-color: #f8f9fa; color: #212529; }
        .container { max-width: 1100px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .disclaimer { background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 6px; border: 1px solid #ffeeba; margin-bottom: 25px; font-size: 0.9em; line-height: 1.5; }
        .search-box { margin-bottom: 30px; background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef; }
        .search-box input[type="text"] { padding: 10px; width: 300px; border: 1px solid #ced4da; border-radius: 4px; }
        .search-box button { padding: 10px 20px; background-color: #0d6efd; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .search-box button:hover { background-color: #0b5ed7; }
        
        .dashboard-grid { display: flex; gap: 25px; margin-bottom: 30px; }
        .panel { flex: 1; padding: 20px; border: 1px solid #e9ecef; border-radius: 8px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
        .panel h3 { margin-top: 0; color: #495057; }
        
        .prediction-positive { color: #0f5132; background-color: #d1e7dd; padding: 15px; border-radius: 6px; text-align: center; font-size: 1.2em; font-weight: bold; }
        .prediction-negative { color: #842029; background-color: #f8d7da; padding: 15px; border-radius: 6px; text-align: center; font-size: 1.2em; font-weight: bold; }
        .prediction-neutral { color: #41464b; background-color: #e2e3e5; padding: 15px; border-radius: 6px; text-align: center; font-size: 1.2em; font-weight: bold; }
        
        table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.95em; }
        th, td { padding: 15px 12px; text-align: left; border-bottom: 1px solid #dee2e6; }
        th { background-color: #f8f9fa; font-weight: 600; color: #495057; }
        tr:hover { background-color: #f8f9fa; }
        a { text-decoration: none; color: #0d6efd; }
        a:hover { text-decoration: underline; }
        
        .badge-positive { color: #198754; font-weight: bold; background: #d1e7dd; padding: 4px 8px; border-radius: 12px; font-size: 0.85em; }
        .badge-negative { color: #dc3545; font-weight: bold; background: #f8d7da; padding: 4px 8px; border-radius: 12px; font-size: 0.85em; }
        .badge-neutral { color: #6c757d; font-weight: bold; background: #e9ecef; padding: 4px 8px; border-radius: 12px; font-size: 0.85em; }
        
        .chart-container { text-align: center; }
        img { max-width: 100%; height: auto; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📰 NewsSense Indian Markets Dashboard</h1>

        <div class="disclaimer">
            <strong>SEBI Disclaimer:</strong> This tool is a technical demonstration for educational purposes only. The sentiment analysis and ML predictions provided do not constitute financial advice.
        </div>

        <div class="search-box">
            <form method="GET" action="/">
                <label style="margin-right: 10px; font-weight: 600;">Ticker / Topic:</label>
                <input type="text" name="q" value="{{ query }}" placeholder="e.g., RELIANCE or NIFTY 50">
                <button type="submit">Analyze News</button>
            </form>
        </div>

        {% if error %}
            <div style="color: #842029; background-color: #f8d7da; padding: 15px; border-radius: 6px;">
                <strong>Error:</strong> {{ error }}
            </div>
        {% elif articles %}
            <div class="dashboard-grid">
                <div class="panel chart-container">
                    <h3>Sentiment Distribution</h3>
                    {% if dist_chart %}
                        <img src="data:image/png;base64,{{ dist_chart }}" alt="Sentiment Chart">
                    {% else %}
                        <p style="color: #6c757d;">Chart generation failed.</p>
                    {% endif %}
                </div>
                
                <div class="panel">
                    <h3>Model Signal</h3>
                    {% if prediction %}
                        <p style="color: #6c757d; margin-bottom: 15px;">Based on historical sentiment-price correlation:</p>
                        <div class="prediction-{{ prediction['label'].lower() }}">
                            Predicted Trend: {{ prediction['label'] }}<br>
                            <span style="font-size: 0.75em; font-weight: normal;">(Confidence: {{ "%.1f"|format(prediction['confidence'] * 100) }}%)</span>
                        </div>
                    {% else %}
                        <div style="background: #e9ecef; padding: 20px; border-radius: 6px; text-align: center; color: #6c757d;">
                            <em>No trained ML model found. Run the training pipeline to enable predictions.</em>
                        </div>
                    {% endif %}
                </div>
            </div>

            <h3>Recent Headlines ({{ articles|length }})</h3>
            <table>
                <tr>
                    <th style="width: 15%;">Published</th>
                    <th style="width: 55%;">Headline</th>
                    <th style="width: 15%;">Sentiment</th>
                    <th style="width: 15%;">Score</th>
                </tr>
                {% for item in articles %}
                <tr>
                    <td style="color: #6c757d; font-size: 0.9em;">{{ item.published }}</td>
                    <td><a href="{{ item.link }}" target="_blank">{{ item.title }}</a></td>
                    <td><span class="badge-{{ item.sentiment|lower }}">{{ item.sentiment }}</span></td>
                    <td><strong>{{ "%.2f"|format(item.score) }}</strong></td>
                </tr>
                {% endfor %}
            </table>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    query = request.args.get("q", "NIFTY")
    try:
        news_items = fetch_news(query, max_results=20)
        if not news_items:
            return render_template_string(TEMPLATE, query=query, error="No recent news found for this query.")

        df = analyze_sentiment_batch(news_items)

        if 'published' in df.columns:
            df['published'] = pd.to_datetime(df['published'], errors='coerce').dt.strftime('%d %b %Y, %H:%M')
            df['published'] = df['published'].fillna('Unknown')

        dist_chart_b64 = None
        try:
            dist_chart_b64 = generate_distribution_chart(df)
        except Exception as e:
            print(f"Chart error: {e}")

        prediction = None
        if os.path.exists("models/sentiment_model.pkl"):
            try:
                prediction = load_model_and_predict()
            except Exception as e:
                print(f"Prediction error: {e}") 

        articles = df.to_dict(orient="records")

        return render_template_string(
            TEMPLATE,
            query=query,
            articles=articles,
            dist_chart=dist_chart_b64,
            prediction=prediction
        )

    except Exception as e:
        return render_template_string(TEMPLATE, query=query, error=str(e))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
