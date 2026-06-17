"""
Servidor Flask — PortfolioAI
Rutas:
  GET  /                      → UI principal
  GET  /api/stock-catalog     → Lista de acciones disponibles para autocomplete
  POST /api/optimize          → Optimización por perfil (Markowitz)
  POST /api/custom-optimize   → Optimización de selección personalizada (Solver)
  POST /api/analyze           → Análisis fundamental de acciones individuales
"""

from flask import Flask, render_template, request, jsonify
from optimizer import PortfolioOptimizer
from custom_optimizer import CustomPortfolioOptimizer
from analyzer import StockAnalyzer
from stock_universe import STOCK_UNIVERSE
from backtest import run_backtest
import traceback
import re

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stock-catalog")
def stock_catalog():
    """Devuelve el catálogo de acciones para el autocomplete del frontend."""
    catalog = []
    for ticker, data in STOCK_UNIVERSE.items():
        catalog.append({
            "ticker": ticker,
            "display": ticker.replace(".MX", ""),
            "name": data.get("name", ticker),
            "sector": data.get("sector", ""),
            "index": data.get("index", ""),
            "currency": data.get("currency", "USD"),
        })
    # Ordenar: S&P 500 primero, luego IPC
    order = {"ETF": 0, "S&P 500": 1, "IPC": 2, "Global": 3}
    catalog.sort(key=lambda x: (order.get(x["index"], 4), x["display"]))
    return jsonify(catalog)


@app.route("/api/optimize", methods=["POST"])
def optimize():
    try:
        data = request.json or {}
        profile = data.get("profile", "moderate")
        amount = float(data.get("amount", 100_000))
        ipc_pct = float(data.get("ipc_pct", 30))
        lookback = int(data.get("lookback", 2))

        if profile not in ("conservative", "moderate", "aggressive"):
            return jsonify({"error": "Perfil inválido"}), 400
        if amount < 1_000:
            return jsonify({"error": "El monto mínimo es $1,000"}), 400

        opt = PortfolioOptimizer(profile, amount, ipc_pct, lookback)
        result = opt.optimize()
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/custom-optimize", methods=["POST"])
def custom_optimize():
    """
    Recibe una lista de tickers seleccionados por el usuario y devuelve
    el portafolio óptimo calculado con el Solver de Markowitz.
    """
    try:
        data = request.json or {}
        tickers = data.get("tickers", [])
        if isinstance(tickers, str):
            tickers = [t.strip() for t in re.split(r"[,\s]+", tickers) if t.strip()]

        if len(tickers) < 2:
            return jsonify({"error": "Selecciona al menos 2 acciones"}), 400
        if len(tickers) > 15:
            return jsonify({"error": "Máximo 15 acciones por portafolio"}), 400

        objective  = data.get("objective", "max_sharpe")
        min_weight = float(data.get("min_weight", 2)) / 100
        max_weight = float(data.get("max_weight", 40)) / 100
        lookback   = int(data.get("lookback", 2))
        amount     = float(data.get("amount", 100_000))

        if objective not in ("max_sharpe", "min_variance", "equal_weight"):
            return jsonify({"error": "Objetivo inválido"}), 400
        if min_weight >= max_weight:
            return jsonify({"error": "El peso mínimo debe ser menor que el máximo"}), 400

        # Peso igual como caso especial (sin optimización de Solver)
        if objective == "equal_weight":
            n = len(tickers)
            objective = "min_variance"
            min_weight = max_weight = 1.0 / n

        opt = CustomPortfolioOptimizer(
            tickers=tickers,
            objective=objective,
            min_weight=min_weight,
            max_weight=max_weight,
            lookback_years=lookback,
            investment_amount=amount,
        )
        result = opt.optimize()
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json or {}
        raw = data.get("tickers", "")
        if isinstance(raw, list):
            tickers = [t.strip().upper() for t in raw if t.strip()]
        else:
            tickers = [t.upper() for t in re.split(r"[,\s]+", raw.strip()) if t.strip()]

        if not tickers:
            return jsonify({"error": "Ingresa al menos un ticker"}), 400
        if len(tickers) > 8:
            return jsonify({"error": "Máximo 8 tickers por consulta"}), 400

        results, errors = [], []
        for ticker in tickers:
            try:
                sa = StockAnalyzer(ticker)
                results.append(sa.analyze())
            except Exception as e:
                errors.append({"ticker": ticker, "error": str(e)})

        return jsonify({"results": results, "errors": errors})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest", methods=["POST"])
def backtest():
    """
    Recibe tickers + pesos y retorna la simulación histórica vs benchmark.
    Body: { tickers: [...], weights: [...], lookback: 3, benchmark: "^GSPC" }
    """
    try:
        data = request.json or {}
        tickers = data.get("tickers", [])
        weights = data.get("weights", [])
        lookback = int(data.get("lookback", 3))
        benchmark = data.get("benchmark", "^GSPC")

        if not tickers:
            return jsonify({"error": "Se necesita al menos un ticker"}), 400
        if len(tickers) != len(weights):
            return jsonify({"error": "Tickers y pesos deben tener la misma longitud"}), 400
        if lookback < 1 or lookback > 10:
            return jsonify({"error": "Lookback debe estar entre 1 y 10 años"}), 400

        result = run_backtest(tickers, weights, lookback, benchmark)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_ENV", "development") != "production"
    app.run(debug=debug, port=port, host="0.0.0.0")
