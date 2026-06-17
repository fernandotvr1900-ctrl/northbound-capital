"""
Motor de backtesting — simula el rendimiento histórico de un portafolio
con pesos fijos y lo compara contra el S&P 500 (o cualquier benchmark).
"""
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

TRADING_DAYS = 252
RF_RATE = 0.045


def run_backtest(
    tickers: List[str],
    weights: List[float],
    lookback_years: int = 3,
    benchmark: str = "^GSPC",
) -> Dict[str, Any]:
    end = datetime.today()
    start = end - timedelta(days=lookback_years * 365)

    # ── Descarga ────────────────────────────────────────────────────────────
    all_tickers = list(tickers) + [benchmark]
    raw = yf.download(all_tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    prices = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=all_tickers[0])

    # ── Separar benchmark y activos ─────────────────────────────────────────
    bench_col = benchmark if benchmark in prices.columns else None

    # Filtrar tickers con datos válidos (en el DataFrame descargado)
    valid = [t for t in tickers if t in prices.columns]
    if not valid:
        raise ValueError("No se encontraron precios para ningún ticker.")

    # Re-normalizar pesos a los tickers válidos
    original_idx = {t: i for i, t in enumerate(tickers)}
    raw_w = np.array([weights[original_idx[t]] for t in valid], dtype=float)
    raw_w /= raw_w.sum()

    # Alinear usando outer join + ffill para tolerar calendarios distintos (MX vs US)
    all_cols = valid + ([bench_col] if bench_col else [])
    aligned = prices[all_cols].copy()
    aligned = aligned.resample("B").last()   # re-indexar a días hábiles
    aligned.ffill(inplace=True)
    aligned.bfill(inplace=True)
    aligned.dropna(how="all", inplace=True)

    asset_prices = aligned[valid]
    bench_series = aligned[bench_col] if bench_col else None

    if len(asset_prices) < 20:
        raise ValueError("Datos insuficientes para el backtesting (< 20 días).")

    # ── Retornos diarios ────────────────────────────────────────────────────
    asset_ret = asset_prices.pct_change()
    port_ret  = (asset_ret * raw_w).sum(axis=1)

    if bench_series is not None:
        bench_ret = bench_series.pct_change()
        df = pd.DataFrame({"portfolio": port_ret, "benchmark": bench_ret})
    else:
        df = pd.DataFrame({"portfolio": port_ret, "benchmark": 0.0})

    df.dropna(inplace=True)
    if len(df) < 5:
        raise ValueError("Datos insuficientes después del alineamiento de calendarios.")

    # ── Cumulativas ─────────────────────────────────────────────────────────
    cum_p = (1 + df["portfolio"]).cumprod()
    cum_b = (1 + df["benchmark"]).cumprod()

    # Resamplear a semanal para un payload razonable
    cum_p_w = cum_p.resample("W").last()
    cum_b_w = cum_b.resample("W").last()

    dates_out = [d.strftime("%Y-%m-%d") for d in cum_p_w.index]
    port_cum = [(round((v - 1) * 100, 2)) for v in cum_p_w.values]
    bench_cum = [(round((v - 1) * 100, 2)) for v in cum_b_w.values]

    # ── Drawdown ────────────────────────────────────────────────────────────
    peak_p = cum_p.cummax()
    dd_p = (cum_p - peak_p) / peak_p
    dd_p_w = (dd_p.resample("W").last() * 100).round(2).tolist()

    peak_b = cum_b.cummax()
    dd_b = (cum_b - peak_b) / peak_b

    # ── Métricas globales ───────────────────────────────────────────────────
    n_years = len(df) / TRADING_DAYS
    total_p = float(cum_p.iloc[-1]) - 1
    total_b = float(cum_b.iloc[-1]) - 1

    cagr_p = (1 + total_p) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    cagr_b = (1 + total_b) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    vol_p = float(df["portfolio"].std() * np.sqrt(TRADING_DAYS))
    vol_b = float(df["benchmark"].std() * np.sqrt(TRADING_DAYS))

    sharpe_p = (cagr_p - RF_RATE) / vol_p if vol_p > 0 else 0.0
    sharpe_b = (cagr_b - RF_RATE) / vol_b if vol_b > 0 else 0.0

    max_dd_p = float(dd_p.min())
    max_dd_b = float(dd_b.min())

    calmar_p = cagr_p / abs(max_dd_p) if max_dd_p != 0 else 0.0
    calmar_b = cagr_b / abs(max_dd_b) if max_dd_b != 0 else 0.0

    # Sortino
    down_p = df["portfolio"][df["portfolio"] < 0].std() * np.sqrt(TRADING_DAYS)
    sortino_p = (cagr_p - RF_RATE) / down_p if down_p > 0 else 0.0

    # Alpha & Beta del portafolio vs benchmark
    if bench_col and len(df) >= 30:
        cv = df.cov()
        beta_val = float(cv.iloc[0, 1] / cv.iloc[1, 1])
        alpha_val = cagr_p - RF_RATE - beta_val * (cagr_b - RF_RATE)
    else:
        beta_val = float("nan")
        alpha_val = float("nan")

    # Ratio de victorias (días en que el portafolio supera al benchmark)
    win_rate = float((df["portfolio"] > df["benchmark"]).mean() * 100)

    # ── Retornos anuales ────────────────────────────────────────────────────
    annual_returns: Dict[str, Dict] = {}
    for year, grp in df.groupby(df.index.year):
        yp = float((1 + grp["portfolio"]).prod() - 1)
        yb = float((1 + grp["benchmark"]).prod() - 1)
        annual_returns[str(year)] = {
            "portfolio": round(yp * 100, 2),
            "benchmark": round(yb * 100, 2),
        }

    # ── Retornos rodantes de 12 meses ───────────────────────────────────────
    roll_p = df["portfolio"].rolling(TRADING_DAYS).apply(
        lambda x: float((1 + x).prod() - 1), raw=True
    ).dropna().resample("W").last()
    roll_b = df["benchmark"].rolling(TRADING_DAYS).apply(
        lambda x: float((1 + x).prod() - 1), raw=True
    ).dropna().resample("W").last()
    roll_dates = [d.strftime("%Y-%m-%d") for d in roll_p.index]
    roll_port = [round(v * 100, 2) for v in roll_p.values]
    roll_bench = [round(v * 100, 2) for v in roll_b.values]

    return {
        "dates": dates_out,
        "portfolio_cum": port_cum,
        "benchmark_cum": bench_cum,
        "drawdown_series": dd_p_w,
        "rolling_12m_dates": roll_dates,
        "rolling_12m_portfolio": roll_port,
        "rolling_12m_benchmark": roll_bench,
        "annual_returns": annual_returns,
        "metrics": {
            "portfolio": {
                "total_return":  round(total_p * 100, 2),
                "cagr":          round(cagr_p * 100, 2),
                "volatility":    round(vol_p * 100, 2),
                "sharpe":        round(sharpe_p, 3),
                "sortino":       round(sortino_p, 3),
                "calmar":        round(calmar_p, 3),
                "max_drawdown":  round(max_dd_p * 100, 2),
                "alpha":         round(alpha_val * 100, 2) if not (isinstance(alpha_val, float) and alpha_val != alpha_val) else None,
                "beta":          round(beta_val, 3) if not (isinstance(beta_val, float) and beta_val != beta_val) else None,
                "win_rate":      round(win_rate, 1),
            },
            "benchmark": {
                "total_return":  round(total_b * 100, 2),
                "cagr":          round(cagr_b * 100, 2),
                "volatility":    round(vol_b * 100, 2),
                "sharpe":        round(sharpe_b, 3),
                "calmar":        round(calmar_b, 3),
                "max_drawdown":  round(max_dd_b * 100, 2),
            },
        },
        "tickers_used": valid,
        "weights_used": {valid[i]: round(float(raw_w[i]) * 100, 2) for i in range(len(valid))},
        "benchmark_label": "S&P 500" if benchmark == "^GSPC" else benchmark,
        "period_years": round(n_years, 1),
        "start_date": df.index[0].strftime("%Y-%m-%d") if len(df) else None,
        "end_date":   df.index[-1].strftime("%Y-%m-%d") if len(df) else None,
    }
