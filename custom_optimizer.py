"""
Optimizador de portafolio personalizado — equivalente al Solver de Excel.

El proceso replica exactamente lo que Excel hace con Solver en un modelo Markowitz:
  1. Construye la matriz de covarianza anualizada (Σ)
  2. Calcula el vector de retornos esperados (μ)
  3. Minimiza σ²p = wᵀ Σ w  sujeto a  Σwᵢ = 1, wᵢ ∈ [min_w, max_w]
     — o bien maximiza el Sharpe Ratio (Rp - Rf) / σp
     — o bien alcanza un target de retorno con mínima varianza
  4. Devuelve pesos, frontera eficiente y métricas de riesgo completas.
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, LinearConstraint, Bounds
from datetime import datetime, timedelta

RF_RATE = 0.045   # Tasa libre de riesgo (T-bill EE.UU. ≈ 4.5%)
TRADING_DAYS = 252
TAX_RATE_US = 0.20   # LTCG federal EE.UU. (aprox)
TAX_RATE_MX = 0.10   # ISR México sobre ganancias de capital


def _download_prices(tickers: List[str], lookback_years: int,
                     include_benchmark: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    end = datetime.today()
    start = end - timedelta(days=lookback_years * 365)
    all_tickers = tickers + (["^GSPC"] if include_benchmark else [])
    raw = yf.download(all_tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    prices = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=all_tickers[0])
    prices.dropna(axis=1, how="all", inplace=True)
    prices.ffill(inplace=True)
    prices.dropna(how="all", inplace=True)
    # Separar benchmark
    bench = pd.DataFrame()
    if "^GSPC" in prices.columns:
        bench = prices[["^GSPC"]].copy()
        prices = prices.drop(columns=["^GSPC"])
    return prices, bench


def _max_drawdown(port_returns: pd.Series) -> float:
    """Máximo drawdown del portafolio (valor negativo)."""
    cum = (1 + port_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def _sortino_ratio(port_returns: pd.Series, rf_daily: float = RF_RATE / TRADING_DAYS) -> float:
    """Sortino = (Rp_anual - rf) / σ_downside_anual."""
    excess = port_returns - rf_daily
    downside = excess[excess < 0].std() * np.sqrt(TRADING_DAYS)
    ann_ret = float(port_returns.mean() * TRADING_DAYS)
    return (ann_ret - RF_RATE) / downside if downside > 1e-10 else 0.0


def _calmar_ratio(ann_ret: float, max_dd: float) -> float:
    """Calmar = retorno anualizado / |max drawdown|."""
    return ann_ret / abs(max_dd) if max_dd != 0 else 0.0


def _beta_portfolio(port_returns: pd.Series, bench: pd.DataFrame) -> Optional[float]:
    """Beta del portafolio vs ^GSPC."""
    if bench.empty:
        return None
    bench_ret = bench["^GSPC"].pct_change().dropna()
    aligned = pd.concat([port_returns, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return None
    cv = aligned.cov()
    return round(float(cv.iloc[0, 1] / cv.iloc[1, 1]), 3)


def _annual_metrics(prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Retorna (retornos anualizados, covarianza anualizada)."""
    daily_ret = prices.pct_change().dropna()
    mu = daily_ret.mean() * TRADING_DAYS
    cov = daily_ret.cov() * TRADING_DAYS
    return mu, cov


def _port_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> Tuple[float, float, float]:
    ret = float(w @ mu)
    var = float(w @ cov @ w)
    vol = float(np.sqrt(var))
    sharpe = (ret - RF_RATE) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _optimize(
    mu: np.ndarray, cov: np.ndarray,
    objective: str,           # "max_sharpe" | "min_variance" | "target_return"
    min_w: float, max_w: float,
    target_return: Optional[float] = None,
) -> np.ndarray:
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = Bounds(lb=min_w, ub=max_w)
    eq_constraint = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    if objective == "max_sharpe":
        def neg_sharpe(w):
            r, v, _ = _port_stats(w, mu, cov)
            return -((r - RF_RATE) / v) if v > 1e-10 else 1e10
        obj = neg_sharpe

    elif objective == "min_variance":
        def min_var(w):
            return float(w @ cov @ w)
        obj = min_var

    elif objective == "target_return":
        tr = target_return or float(mu.mean())
        ret_constraint = {"type": "eq", "fun": lambda w: float(w @ mu) - tr}
        def min_var_tr(w):
            return float(w @ cov @ w)
        res = minimize(min_var_tr, w0, method="SLSQP",
                       bounds=[(min_w, max_w)] * n,
                       constraints=[eq_constraint, ret_constraint],
                       options={"ftol": 1e-12, "maxiter": 2000})
        return res.x if res.success else w0

    else:
        obj = lambda w: float(w @ cov @ w)

    res = minimize(obj, w0, method="SLSQP",
                   bounds=[(min_w, max_w)] * n,
                   constraints=[eq_constraint],
                   options={"ftol": 1e-12, "maxiter": 2000})
    return res.x if res.success else w0


def _efficient_frontier(
    mu: np.ndarray, cov: np.ndarray, min_w: float, max_w: float,
    n_points: int = 40,
) -> List[Dict[str, float]]:
    """Construye la frontera eficiente variando el retorno objetivo."""
    ret_min = float(np.clip(mu.min(), -0.5, 5.0))
    ret_max = float(np.clip(mu.max(), -0.5, 5.0))
    n = len(mu)
    frontier = []
    for target in np.linspace(ret_min, ret_max, n_points):
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
        ]
        res = minimize(
            lambda w: float(w @ cov @ w),
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(min_w, max_w)] * n,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if res.success:
            vol = float(np.sqrt(res.x @ cov @ res.x))
            frontier.append({"return": round(target * 100, 2),
                             "volatility": round(vol * 100, 2)})
    return frontier


def _correlation_matrix(cov: pd.DataFrame) -> List[Dict]:
    """Devuelve la matriz de correlación como lista de filas para el frontend."""
    std = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    tickers = list(cov.columns)
    rows = []
    for i, row_t in enumerate(tickers):
        cells = []
        for j, col_t in enumerate(tickers):
            cells.append({
                "col": col_t.replace(".MX", ""),
                "value": round(float(corr[i, j]), 3),
            })
        rows.append({"ticker": row_t.replace(".MX", ""), "cells": cells})
    return rows


class CustomPortfolioOptimizer:
    def __init__(
        self,
        tickers: List[str],
        objective: str = "max_sharpe",
        min_weight: float = 0.02,
        max_weight: float = 0.40,
        lookback_years: int = 2,
        investment_amount: float = 100_000,
    ):
        self.tickers = [t.upper().strip() for t in tickers]
        self.objective = objective
        self.min_w = min_weight
        self.max_w = max_weight
        self.lookback = lookback_years
        self.amount = investment_amount

    def optimize(self) -> Dict[str, Any]:
        # ── Descarga de precios ────────────────────────────────────────────
        prices, bench = _download_prices(self.tickers, self.lookback)

        # Filtrar tickers sin datos suficientes
        valid = [c for c in prices.columns if prices[c].dropna().shape[0] > 60]
        if len(valid) < 2:
            raise ValueError("Se necesitan al menos 2 tickers con datos suficientes.")

        prices = prices[valid]
        mu, cov = _annual_metrics(prices)

        mu_arr = mu.values
        cov_arr = cov.values

        # ── Optimización (Solver Markowitz) ────────────────────────────────
        weights = _optimize(mu_arr, cov_arr, self.objective, self.min_w, self.max_w)
        ret, vol, sharpe = _port_stats(weights, mu_arr, cov_arr)

        # ── Frontera eficiente ─────────────────────────────────────────────
        frontier = _efficient_frontier(mu_arr, cov_arr, self.min_w, self.max_w)

        # ── Portafolio de mínima varianza (referencia) ────────────────────
        w_mv = _optimize(mu_arr, cov_arr, "min_variance", self.min_w, self.max_w)
        r_mv, v_mv, s_mv = _port_stats(w_mv, mu_arr, cov_arr)

        # ── Métricas individuales ──────────────────────────────────────────
        holdings = []
        for i, ticker in enumerate(valid):
            w = float(weights[i])
            ind_ret = float(mu_arr[i])
            ind_vol = float(np.sqrt(cov_arr[i, i]))
            ind_sharpe = (ind_ret - RF_RATE) / ind_vol if ind_vol > 0 else 0.0

            # Contribución al riesgo del portafolio (marginal)
            marginal = float(cov_arr[i, :] @ weights) / vol if vol > 0 else 0.0
            risk_contrib = float(w * marginal / vol) if vol > 0 else 0.0

            holdings.append({
                "ticker": ticker,
                "ticker_display": ticker.replace(".MX", ""),
                "weight": round(w * 100, 2),
                "amount": round(w * self.amount, 2),
                "expected_return": round(ind_ret * 100, 2),
                "volatility": round(ind_vol * 100, 2),
                "sharpe": round(ind_sharpe, 3),
                "risk_contribution": round(risk_contrib * 100, 2),
                "variance_contribution": round(float(w * marginal) * 100, 4),
            })

        holdings.sort(key=lambda x: x["weight"], reverse=True)

        # ── Matriz de correlación ──────────────────────────────────────────
        corr_matrix = _correlation_matrix(cov)

        # ── Covarianza como tabla para UI ──────────────────────────────────
        cov_table = []
        tickers_display = [t.replace(".MX", "") for t in valid]
        for i, row_t in enumerate(tickers_display):
            cells = []
            for j in range(len(valid)):
                cells.append({
                    "col": tickers_display[j],
                    "value": round(float(cov_arr[i, j]) * 100, 4),
                })
            cov_table.append({"ticker": row_t, "cells": cells})

        # ── Estadísticos individuales sin optimizar (pesos iguales) ────────
        w_eq = np.ones(len(valid)) / len(valid)
        r_eq, v_eq, s_eq = _port_stats(w_eq, mu_arr, cov_arr)

        # ── Métricas avanzadas del portafolio ──────────────────────────────
        daily_ret = prices.pct_change().dropna()
        port_daily = (daily_ret * weights).sum(axis=1)
        max_dd = _max_drawdown(port_daily)
        sortino = _sortino_ratio(port_daily)
        calmar = _calmar_ratio(ret, max_dd)
        beta_sp = _beta_portfolio(port_daily, bench)

        # Retorno ajustado por impuestos (capital gains)
        ret_after_tax_us = ret * (1 - TAX_RATE_US)
        ret_after_tax_mx = ret * (1 - TAX_RATE_MX)

        return {
            "tickers": [t.replace(".MX", "") for t in valid],
            "objective": self.objective,
            "investment_amount": self.amount,
            "portfolio": {
                "expected_return": round(ret * 100, 2),
                "volatility": round(vol * 100, 2),
                "variance": round((vol ** 2) * 100, 4),
                "sharpe_ratio": round(sharpe, 3),
                "sortino_ratio": round(sortino, 3),
                "calmar_ratio": round(calmar, 3),
                "max_drawdown": round(max_dd * 100, 2),
                "beta_sp500": beta_sp,
                "risk_free_rate": round(RF_RATE * 100, 1),
                "return_after_tax_us": round(ret_after_tax_us * 100, 2),
                "return_after_tax_mx": round(ret_after_tax_mx * 100, 2),
                "tax_rate_us": round(TAX_RATE_US * 100, 0),
                "tax_rate_mx": round(TAX_RATE_MX * 100, 0),
            },
            "benchmark_equal_weight": {
                "expected_return": round(r_eq * 100, 2),
                "volatility": round(v_eq * 100, 2),
                "sharpe_ratio": round(s_eq, 3),
            },
            "min_variance_portfolio": {
                "expected_return": round(r_mv * 100, 2),
                "volatility": round(v_mv * 100, 2),
                "sharpe_ratio": round(s_mv, 3),
                "weights": {valid[i].replace(".MX",""):round(float(w_mv[i])*100,2) for i in range(len(valid))},
            },
            "holdings": holdings,
            "frontier": frontier,
            "correlation_matrix": corr_matrix,
            "covariance_table": cov_table,
            "generated_at": datetime.now().isoformat(),
        }
