from flask import Flask, jsonify, redirect, render_template, request, url_for, session
from src.database import register_or_login_user, init_db
import logging
import os
import time
from datetime import datetime
from src.research import (
    build_research_scan,
    download_nifty50_prices,
    fetch_company_news,
    fetch_company_research,
    load_nifty50_model,
    load_nifty50_universe,
    MODEL_PATH,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "newssense-production-secret-key-2026")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_research_scan_cache = {}
_company_cache = {}
_model_cache = {"mtime": None, "artifact": None}

# Ensure SQLite database is initialized
init_db()

@app.route("/")
def home():
    """Render the public 3D landing page."""
    return render_template("landing.html")

@app.route("/dashboard")
def dashboard():
    """Render the authenticated NIFTY 50 research workspace."""
    if not session.get("is_authenticated"):
        return redirect(url_for("auth_page"))
    return render_template("research.html", user=session.get("user", "Trader"))

@app.route("/nifty50")
def nifty50_research():
    """Compatibility route for the NIFTY 50 dashboard."""
    return redirect(url_for("dashboard"))


def _get_model_artifact():
    modified = MODEL_PATH.stat().st_mtime_ns
    if _model_cache["artifact"] is None or _model_cache["mtime"] != modified:
        _model_cache["artifact"] = load_nifty50_model()
        _model_cache["mtime"] = modified
    return _model_cache["artifact"]


@app.route("/api/universe")
def api_universe():
    try:
        return jsonify({"companies": load_nifty50_universe()})
    except Exception as error:
        app.logger.exception("Could not load NIFTY 50 constituents")
        return jsonify({"error": str(error)}), 503


@app.route("/api/scan")
def api_scan():
    try:
        try:
            horizon_years = int(request.args.get("horizon", "10"))
        except ValueError:
            return jsonify({"error": "Forecast horizon must be a whole number from 1 to 10."}), 400
        if not 1 <= horizon_years <= 10:
            return jsonify({"error": "Forecast horizon must be between 1 and 10 years."}), 400

        force_refresh = request.args.get("refresh", "").lower() in {"1", "true", "yes"}
        now = time.monotonic()
        cached = _research_scan_cache.get(horizon_years)
        if force_refresh or cached is None or now - cached["loaded_at"] >= 60:
            artifact = _get_model_artifact()
            _, live_closes = download_nifty50_prices(
                universe=artifact["universe"],
                period="5d",
            )
            horizon_entry = artifact["models_by_horizon"].get(str(horizon_years))
            cached = {
                "loaded_at": now,
                "rows": build_research_scan(artifact, live_closes, horizon_years),
                "metrics": horizon_entry["metrics"] if horizon_entry else None,
            }
            _research_scan_cache[horizon_years] = cached

        return jsonify(
            {
                "rows": cached["rows"],
                "metrics": cached["metrics"],
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "horizon_years": horizon_years,
            }
        )
    except FileNotFoundError as error:
        return jsonify({"error": str(error), "requires_training": True}), 503
    except Exception as error:
        app.logger.exception("NIFTY 50 research scan failed")
        return jsonify({"error": f"Research scan unavailable: {error}"}), 503


@app.route("/api/news/<symbol>")
def api_news(symbol):
    """Fetch fresh real-time financial headlines for a stock."""
    try:
        universe = load_nifty50_universe()
        company = next(
            (item for item in universe if item["symbol"].upper() == symbol.upper()),
            None,
        )
        if company is None:
            return jsonify({"error": "Choose a current NIFTY 50 constituent."}), 404

        news = fetch_company_news(company)
        return jsonify({
            "symbol": company["symbol"],
            "name": company["name"],
            "count": len(news),
            "news": news,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
    except Exception as error:
        app.logger.exception("Live news request failed for %s", symbol)
        return jsonify({"error": f"Live news unavailable: {error}"}), 503


@app.route("/api/company/<symbol>")
def api_company(symbol):
    try:
        try:
            horizon_years = int(request.args.get("horizon", "10"))
        except ValueError:
            return jsonify({"error": "Forecast horizon must be a whole number from 1 to 10."}), 400
        if not 1 <= horizon_years <= 10:
            return jsonify({"error": "Forecast horizon must be between 1 and 10 years."}), 400
        universe = load_nifty50_universe()
        company = next(
            (item for item in universe if item["symbol"].upper() == symbol.upper()),
            None,
        )
        if company is None:
            return jsonify({"error": "Choose a current NIFTY 50 constituent."}), 404

        force_refresh = request.args.get("refresh", "").lower() in {"1", "true", "yes"}
        now = time.monotonic()
        cache_key = (company["symbol"], horizon_years)
        cached = _company_cache.get(cache_key)
        if not force_refresh and cached and now - cached["loaded_at"] < 60:
            if cached["data"].get("prediction") is not None:
                return jsonify(cached["data"])
            try:
                _get_model_artifact()
            except FileNotFoundError:
                return jsonify(cached["data"])

        artifact = None
        try:
            artifact = _get_model_artifact()
            horizon_entry = artifact["models_by_horizon"].get(str(horizon_years))
            models = horizon_entry["models"] if horizon_entry else None
            metrics = horizon_entry["metrics"] if horizon_entry else None
        except FileNotFoundError:
            horizon_entry = None
            models = None
            metrics = None

        details = fetch_company_research(company, models, metrics, horizon_years)
        if models is None and not details.get("forecast_unavailable_reason"):
            details["forecast_unavailable_reason"] = (
                artifact.get("horizon_errors", {}).get(
                    str(horizon_years), "Not enough history to train this forecast horizon."
                )
                if artifact
                else "Train the price-history model before requesting a forecast."
            )
        _company_cache[cache_key] = {"loaded_at": now, "data": details}
        return jsonify(details)
    except Exception as error:
        app.logger.exception("Company research request failed for %s", symbol)
        return jsonify({"error": f"Company data unavailable: {error}"}), 503

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
    app.logger.info("Guest session initialized. Opening NIFTY 50 dashboard.")
    return redirect(url_for("dashboard"))

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

    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    """Clears authentication session and returns to landing page"""
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
