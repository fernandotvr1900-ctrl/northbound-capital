"""
Motor de optimización de portafolios basado en la Teoría Moderna de Portafolios (Markowitz).
Calcula la frontera eficiente, maximiza el Sharpe Ratio y devuelve métricas completas.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
from stock_universe import STOCK_UNIVERSE, PROFILE_CONFIG, get_tickers_for_profile

TRADING_DAYS = 252
TAX_RATE_US = 0.20
TAX_RATE_MX = 0.10

BENCHMARK = {
    "sp500": "^GSPC",
    "ipc": "^MXX",
}


def _max_drawdown(port_returns: pd.Series) -> float:
    cum = (1 + port_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def _sortino_ratio(port_returns: pd.Series, rf: float, trading_days: int = TRADING_DAYS) -> float:
    rf_daily = rf / trading_days
    excess = port_returns - rf_daily
    downside = excess[excess < 0].std() * np.sqrt(trading_days)
    ann_ret = float(port_returns.mean() * trading_days)
    return (ann_ret - rf) / downside if downside > 1e-10 else 0.0


def _calmar_ratio(ann_ret: float, max_dd: float) -> float:
    return ann_ret / abs(max_dd) if max_dd != 0 else 0.0


class PortfolioOptimizer:
    def __init__(self, profile: str, investment_amount: float = 100_000,
                 ipc_pct: float = 30.0, lookback_years: int = 2):
        self.profile = profile
        self.investment_amount = investment_amount
        self.ipc_pct = ipc_pct / 100.0   # porcentaje deseado de IPC en el portafolio
        self.lookback_years = lookback_years
        self.cfg = PROFILE_CONFIG[profile]
        self.rf = self.cfg["risk_free_rate"]

    # ─── Descarga de datos ────────────────────────────────────────────────────

    def _download(self, tickers: list[str], extra: list[str] = None) -> pd.DataFrame:
        all_tickers = tickers + (extra or [])
        end = datetime.today()
        start = end - timedelta(days=self.lookback_years * 365)
        raw = yf.download(all_tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        prices = raw["Close"] if "Close" in raw.columns else raw
        # Asegurar DataFrame aunque sea un solo ticker
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=all_tickers[0])
        prices.dropna(axis=1, how="all", inplace=True)
        prices.dropna(how="all", inplace=True)
        prices.ffill(inplace=True)
        return prices

    # ─── Selección inicial de acciones ───────────────────────────────────────

    def _select_candidates(self, prices: pd.DataFrame) -> list[str]:
        """
        Filtra los tickers con datos suficientes y selecciona los mejores
        por Sharpe individual para mantener el universo manejable.
        """
        cfg = self.cfg
        returns = prices.pct_change().dropna()
        sharpes = {}
        for col in returns.columns:
            mu = returns[col].mean() * 252
            sigma = returns[col].std() * np.sqrt(252)
            if sigma > 0:
                sharpes[col] = (mu - self.rf) / sigma

        sp500_tickers = sorted(
            [t for t in sharpes if not t.endswith(".MX")],
            key=lambda x: sharpes[x], reverse=True
        )
        ipc_tickers = sorted(
            [t for t in sharpes if t.endswith(".MX")],
            key=lambda x: sharpes[x], reverse=True
        )

        max_stocks = cfg["max_stocks"]
        n_ipc = max(1, round(max_stocks * self.ipc_pct))
        n_sp = max(1, max_stocks - n_ipc)

        selected = sp500_tickers[:n_sp] + ipc_tickers[:n_ipc]
        return selected

    # ─── Métricas de portafolio ───────────────────────────────────────────────

    def _port_stats(self, w: np.ndarray, mu: np.ndarray,
                    cov: np.ndarray) -> tuple[float, float, float]:
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        sharpe = (ret - self.rf) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    # ─── Optimización Markowitz ───────────────────────────────────────────────

    def _optimize_sharpe(self, mu: np.ndarray, cov: np.ndarray,
                         max_w: float) -> np.ndarray:
        n = len(mu)
        w0 = np.ones(n) / n
        bounds = [(0.02, max_w)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        def neg_sharpe(w):
            r, v, s = self._port_stats(w, mu, cov)
            return -s

        res = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 1000})
        return res.x if res.success else w0

    def _optimize_min_var(self, mu: np.ndarray, cov: np.ndarray,
                          max_w: float) -> np.ndarray:
        n = len(mu)
        w0 = np.ones(n) / n
        bounds = [(0.02, max_w)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        def portfolio_var(w):
            return float(w @ cov @ w)

        res = minimize(portfolio_var, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 1000})
        return res.x if res.success else w0

    def _efficient_frontier(self, mu: np.ndarray, cov: np.ndarray,
                            max_w: float, n_points: int = 30) -> list[dict]:
        ret_min = float(mu.min())
        ret_max = float(mu.max())
        targets = np.linspace(ret_min, ret_max, n_points)
        frontier = []
        n = len(mu)
        bounds = [(0.0, max_w)] * n

        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
            ]
            w0 = np.ones(n) / n
            res = minimize(lambda w: float(w @ cov @ w), w0,
                           method="SLSQP", bounds=bounds,
                           constraints=constraints,
                           options={"ftol": 1e-9, "maxiter": 500})
            if res.success:
                _, vol, _ = self._port_stats(res.x, mu, cov)
                frontier.append({"return": round(target * 100, 2),
                                  "volatility": round(vol * 100, 2)})
        return frontier

    # ─── Beta vs benchmark ────────────────────────────────────────────────────

    def _calc_beta(self, port_returns: pd.Series,
                   bench_prices: pd.DataFrame) -> dict[str, float]:
        betas = {}
        for name, col in BENCHMARK.items():
            if col in bench_prices.columns:
                bench_ret = bench_prices[col].pct_change().dropna()
                aligned = pd.concat([port_returns, bench_ret], axis=1,
                                    join="inner").dropna()
                if len(aligned) < 30:
                    continue
                cov_mat = aligned.cov()
                betas[name] = round(
                    cov_mat.iloc[0, 1] / cov_mat.iloc[1, 1], 4)
        return betas

    # ─── Punto de entrada principal ───────────────────────────────────────────

    def optimize(self) -> dict:
        universe_tickers = get_tickers_for_profile(self.profile)
        benchmarks = list(BENCHMARK.values())

        # Descarga precios
        all_prices = self._download(universe_tickers, benchmarks)

        bench_prices = all_prices[[c for c in benchmarks if c in all_prices.columns]]
        asset_prices = all_prices[[c for c in all_prices.columns if c not in benchmarks]]

        # Selección de candidatos
        selected = self._select_candidates(asset_prices)
        if len(selected) < 3:
            selected = list(asset_prices.columns[: max(3, len(asset_prices.columns))])

        prices_sel = asset_prices[selected].dropna()
        returns = prices_sel.pct_change().dropna()

        mu_annual = returns.mean().values * 252
        cov_annual = returns.cov().values * 252
        max_w = self.cfg["max_weight"]

        # Elegir estrategia según perfil
        if self.profile == "conservative":
            weights = self._optimize_min_var(mu_annual, cov_annual, max_w)
        else:
            weights = self._optimize_sharpe(mu_annual, cov_annual, max_w)

        ret, vol, sharpe = self._port_stats(weights, mu_annual, cov_annual)

        # Frontera eficiente
        frontier = self._efficient_frontier(mu_annual, cov_annual, max_w)

        # Retornos del portafolio para beta y métricas avanzadas
        port_returns = (returns * weights).sum(axis=1)
        betas = self._calc_beta(port_returns, bench_prices)

        # Métricas avanzadas
        max_dd = _max_drawdown(port_returns)
        sortino = _sortino_ratio(port_returns, self.rf)
        calmar = _calmar_ratio(ret, max_dd)
        ret_after_tax_us = ret * (1 - TAX_RATE_US)
        ret_after_tax_mx = ret * (1 - TAX_RATE_MX)

        # Métricas individuales de cada acción
        holdings = []
        for i, ticker in enumerate(selected):
            w = float(weights[i])
            if w < 0.005:
                continue
            ind_ret = float(mu_annual[i])
            ind_vol = float(np.sqrt(cov_annual[i, i]))
            ind_sharpe = (ind_ret - self.rf) / ind_vol if ind_vol > 0 else 0.0

            # Beta individual vs S&P 500
            bench_col = BENCHMARK["sp500"]
            ind_beta = None
            if bench_col in bench_prices.columns:
                s = pd.concat([returns[ticker],
                                bench_prices[bench_col].pct_change()],
                               axis=1, join="inner").dropna()
                if len(s) >= 30:
                    cv = s.cov()
                    ind_beta = round(cv.iloc[0, 1] / cv.iloc[1, 1], 3)

            meta = STOCK_UNIVERSE.get(ticker, {})
            holdings.append({
                "ticker": ticker,
                "name": meta.get("name", ticker),
                "sector": meta.get("sector", "N/A"),
                "index": meta.get("index", "N/A"),
                "currency": meta.get("currency", "USD"),
                "weight": round(w * 100, 2),
                "amount": round(w * self.investment_amount, 2),
                "expected_return": round(ind_ret * 100, 2),
                "volatility": round(ind_vol * 100, 2),
                "sharpe": round(ind_sharpe, 3),
                "beta": ind_beta,
                "thesis": meta.get("thesis", ""),
            })

        holdings.sort(key=lambda x: x["weight"], reverse=True)

        # Varianza del portafolio
        port_variance = vol ** 2

        return {
            "profile": self.profile,
            "profile_label": self.cfg["label"],
            "investment_amount": self.investment_amount,
            "portfolio": {
                "expected_return": round(ret * 100, 2),
                "volatility": round(vol * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "sortino_ratio": round(sortino, 3),
                "calmar_ratio": round(calmar, 3),
                "max_drawdown": round(max_dd * 100, 2),
                "variance": round(port_variance * 100, 4),
                "beta_sp500": betas.get("sp500"),
                "beta_ipc": betas.get("ipc"),
                "return_after_tax_us": round(ret_after_tax_us * 100, 2),
                "return_after_tax_mx": round(ret_after_tax_mx * 100, 2),
            },
            "holdings": holdings,
            "frontier": frontier,
            "generated_at": datetime.now().isoformat(),
        }
