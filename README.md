# NewsSense

NewsSense is an experimental NIFTY 50 company-research workspace. The public landing page leads to guest or SQLite-backed authentication. The dashboard combines adjusted-price history, company facts, source-linked recent coverage, TF-IDF retrieval, and a selectable 1–10 year return estimate (default: 10 years).

## Run locally

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py
python app.py
```

Open `http://localhost:5000`. Training downloads the current NIFTY 50 constituents and maximum available Yahoo Finance price history for each company; the first run can take several minutes. Company details and the scan refresh from their sources every five minutes. Yahoo data can be delayed or incomplete.

## Dashboard features

- Three.js animated landing page with guest and account entry.
- Current NIFTY 50 company selector, latest available quote, and interactive 1-, 5-, and 10-year price chart.
- Historical returns and reported fundamentals, with growth/risk factors and company profile.
- Recent Google News RSS coverage with publisher links and publication dates.
- TF-IDF retrieval over company facts, historical performance, and news, with source citations.
- A chart and sortable 50-company table of experimental next-year return estimates and positive-return probabilities.
- Optional LLM synthesis using the OpenAI Chat Completions API. Set `OPENAI_API_KEY` in the local environment to enable it; without a key, the dashboard uses a deterministic retrieved-data summary. The LLM receives public research context only.

## Forecast methodology and limits

The Random Forest models use monthly price-derived momentum, volatility, and drawdown features. Separate models estimate adjusted-price return, gain/loss probability, and conditional loss severity at each available 1–10 year horizon. “Loss if negative” is conditional severity; “probability-weighted loss” multiplies it by the estimated loss probability. Chronological holdouts are purged by the selected horizon. Return error, probability Brier score, and conditional-loss error are each compared with simple baselines. The UI shows the probability-based model view (potential buy, hold, or avoid) separately from its holdout-validation status and exact failed checks. A potential-buy view is still unvalidated when the model fails its baseline tests. A company must have enough of its own history for the selected horizon. These estimates are not personalized investment recommendations and do not predict guaranteed profit.

Without `OPENAI_API_KEY`, the RAG panel uses TF-IDF retrieval and a deterministic extractive summary. With a key configured locally, retrieved source snippets are sent to the OpenAI Chat Completions API for a concise cited summary. No API key is required for price, company-fact, news, or forecast features.

Historical model evaluation uses today's constituents, which introduces survivorship bias. Financial facts may be missing or stale, company fundamentals are not all sector-normalized, and the price-only model does not incorporate news or filings. Backtest performance is not proof of future performance. Taxes, trading costs, liquidity, and individual risk tolerance are not modeled.

## Stack

Flask, pandas, NumPy, scikit-learn, yfinance, Three.js, and Chart.js. An optional Docker setup is included in `Dockerfile` and `docker-compose.yml`.
