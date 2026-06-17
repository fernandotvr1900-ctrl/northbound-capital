"""
Análisis fundamental y técnico de acciones individuales.
Emite señales de compra / mantener / venta y veredicto de valuación.
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# Medianas de P/E forward por sector (estimados consenso 2024-2025)
SECTOR_PE_MEDIANS: dict[str, float] = {
    "Technology": 27.0,
    "Healthcare": 20.0,
    "Financial Services": 12.5,
    "Financials": 12.5,
    "Consumer Staples": 21.0,
    "Consumer Defensive": 21.0,
    "Consumer Discretionary": 24.0,
    "Consumer Cyclical": 24.0,
    "Energy": 11.0,
    "Basic Materials": 14.0,
    "Materials": 14.0,
    "Industrials": 20.0,
    "Utilities": 16.0,
    "Real Estate": 32.0,
    "Communication Services": 18.0,
    "Telecommunications": 14.0,
    "default": 19.0,
}


def _fmt_pct(v: Optional[float], decimals: int = 1) -> Optional[str]:
    return f"{v * 100:.{decimals}f}%" if v is not None else None


class StockAnalyzer:
    def __init__(self, ticker: str):
        raw = ticker.strip().upper()
        # Si el usuario escribe sin .MX y tiene aspecto de ticker mexicano conocido,
        # no modificar — yfinance requiere el sufijo .MX para la BMV.
        self.ticker = raw
        self._yf = yf.Ticker(self.ticker)

    def analyze(self) -> dict:
        info = self._yf.info or {}
        if not info or info.get("quoteType") is None:
            raise ValueError(
                f"Ticker '{self.ticker}' no encontrado. "
                "Verifica el símbolo (IPC: añade .MX, ej. AMXL.MX)."
            )

        # ── Precios / rango ────────────────────────────────────────────────
        current = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("navPrice")
        )
        if current is None:
            raise ValueError(f"No se pudo obtener el precio actual para '{self.ticker}'.")

        hi52 = info.get("fiftyTwoWeekHigh")
        lo52 = info.get("fiftyTwoWeekLow")
        ma50 = info.get("fiftyDayAverage")
        ma200 = info.get("twoHundredDayAverage")

        pos52 = None
        if hi52 and lo52 and hi52 != lo52:
            pos52 = round((current - lo52) / (hi52 - lo52) * 100, 1)

        # ── Valuación ──────────────────────────────────────────────────────
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        pb = info.get("priceToBook")
        peg = info.get("pegRatio")
        ev_ebitda = info.get("enterpriseToEbitda")
        ps_ratio = info.get("priceToSalesTrailing12Months")

        # ── Analistas ─────────────────────────────────────────────────────
        target = info.get("targetMeanPrice")
        target_hi = info.get("targetHighPrice")
        target_lo = info.get("targetLowPrice")
        rec_mean = info.get("recommendationMean")
        rec_key = (info.get("recommendationKey") or "").lower()
        n_analysts = info.get("numberOfAnalystOpinions") or 0

        upside = round((target - current) / current * 100, 1) if target and current else None

        # ── Histórico (1 año) ─────────────────────────────────────────────
        hist = self._yf.history(period="1y")
        vol_annual = None
        ret_1y = None
        if len(hist) > 10:
            rets = hist["Close"].pct_change().dropna()
            vol_annual = round(float(rets.std() * np.sqrt(252) * 100), 1)
            ret_1y = round(
                float((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100), 1
            )

        # ── Scoring ────────────────────────────────────────────────────────
        sector = info.get("sector", "default")
        val_score, val_verdict, val_reasons = self._valuation_score(
            trailing_pe, forward_pe, pb, peg, current, target, sector
        )
        sig_score, signal, sig_reasons = self._signal_score(
            val_score, rec_mean, current, ma50, ma200, target,
            info.get("revenueGrowth"), info.get("earningsGrowth"),
        )

        # ── Dividendo ─────────────────────────────────────────────────────
        div_yield = info.get("dividendYield")
        div_yield_pct = round(div_yield * 100, 2) if div_yield else None

        return {
            "ticker": self.ticker,
            "name": info.get("longName") or info.get("shortName") or self.ticker,
            "sector": sector if sector != "default" else "N/D",
            "industry": info.get("industry", "N/D"),
            "exchange": info.get("exchange", "N/D"),
            "currency": info.get("currency", "USD"),
            "country": info.get("country", "N/D"),
            # Precio
            "current_price": round(current, 2),
            "week52_high": round(hi52, 2) if hi52 else None,
            "week52_low": round(lo52, 2) if lo52 else None,
            "week52_position": pos52,
            "ma50": round(ma50, 2) if ma50 else None,
            "ma200": round(ma200, 2) if ma200 else None,
            # Valuación
            "trailing_pe": round(trailing_pe, 2) if trailing_pe else None,
            "forward_pe": round(forward_pe, 2) if forward_pe else None,
            "price_to_book": round(pb, 2) if pb else None,
            "peg_ratio": round(peg, 2) if peg else None,
            "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda else None,
            "ps_ratio": round(ps_ratio, 2) if ps_ratio else None,
            # Analistas
            "target_price": round(target, 2) if target else None,
            "target_high": round(target_hi, 2) if target_hi else None,
            "target_low": round(target_lo, 2) if target_lo else None,
            "upside_pct": upside,
            "analyst_count": n_analysts,
            "recommendation": rec_key,
            "recommendation_mean": round(rec_mean, 2) if rec_mean else None,
            # Riesgo
            "beta": round(info.get("beta", 1.0), 2) if info.get("beta") else None,
            "volatility_annual": vol_annual,
            "return_1y": ret_1y,
            "dividend_yield": div_yield_pct,
            # Crecimiento
            "revenue_growth": _fmt_pct(info.get("revenueGrowth")),
            "earnings_growth": _fmt_pct(info.get("earningsGrowth")),
            # Veredicto
            "valuation_verdict": val_verdict,
            "valuation_score": val_score,
            "valuation_reasons": val_reasons,
            "signal": signal,
            "signal_score": sig_score,
            "signal_reasons": sig_reasons,
            # Cap
            "market_cap": info.get("marketCap"),
        }

    # ── Valuación ──────────────────────────────────────────────────────────────

    def _valuation_score(
        self, trailing_pe, forward_pe, pb, peg, current, target, sector
    ) -> Tuple[int, str, List[str]]:
        score = 0
        reasons: List[str] = []
        median_pe = SECTOR_PE_MEDIANS.get(sector, SECTOR_PE_MEDIANS["default"])

        if forward_pe and forward_pe > 0:
            ratio = forward_pe / median_pe
            if ratio < 0.75:
                score += 2
                reasons.append(
                    f"P/E forward {forward_pe:.1f}x está un {(1-ratio)*100:.0f}% "
                    f"por debajo de la mediana sectorial ({median_pe:.0f}x) — descuento significativo."
                )
            elif ratio < 0.95:
                score += 1
                reasons.append(
                    f"P/E forward {forward_pe:.1f}x ligeramente por debajo "
                    f"de la mediana sectorial ({median_pe:.0f}x)."
                )
            elif ratio > 1.6:
                score -= 2
                reasons.append(
                    f"P/E forward {forward_pe:.1f}x cotiza con una prima del "
                    f"{(ratio-1)*100:.0f}% sobre la mediana sectorial ({median_pe:.0f}x)."
                )
            elif ratio > 1.25:
                score -= 1
                reasons.append(
                    f"P/E forward {forward_pe:.1f}x por encima de la mediana "
                    f"sectorial ({median_pe:.0f}x)."
                )

        if peg and peg > 0:
            if peg < 1.0:
                score += 2
                reasons.append(
                    f"PEG {peg:.2f} — precio atractivo en relación al crecimiento esperado (umbral óptimo < 1)."
                )
            elif peg < 1.5:
                score += 1
                reasons.append(f"PEG {peg:.2f} — razonable para el crecimiento proyectado.")
            elif peg > 3.0:
                score -= 2
                reasons.append(
                    f"PEG {peg:.2f} — múltiplo de crecimiento elevado; "
                    "el precio incorpora expectativas muy optimistas."
                )
            elif peg > 2.0:
                score -= 1
                reasons.append(f"PEG {peg:.2f} — precio caro relativo al crecimiento.")

        if current and target:
            up = (target - current) / current
            if up > 0.20:
                score += 2
                reasons.append(
                    f"Consenso de analistas implica un upside de +{up*100:.1f}% "
                    f"al precio objetivo de ${target:.2f}."
                )
            elif up > 0.05:
                score += 1
                reasons.append(
                    f"Upside moderado al precio objetivo: +{up*100:.1f}%."
                )
            elif up < -0.10:
                score -= 2
                reasons.append(
                    f"El precio actual supera en {abs(up)*100:.1f}% el precio "
                    f"objetivo de analistas (${target:.2f}) — posible sobrevaluación."
                )
            elif up < 0:
                score -= 1
                reasons.append(
                    f"Precio ligeramente por encima del objetivo ({abs(up)*100:.1f}%)."
                )

        if pb and pb > 0:
            if pb < 1.0:
                score += 1
                reasons.append(
                    f"P/B {pb:.2f}x — cotiza por debajo de su valor en libros."
                )
            elif pb > 12:
                score -= 1
                reasons.append(f"P/B {pb:.2f}x elevado; refleja expectativas de crecimiento premium.")

        if score >= 3:
            verdict = "SUBVALUADA"
        elif score >= 0:
            verdict = "VALUACIÓN JUSTA"
        else:
            verdict = "SOBREVALUADA"

        return score, verdict, reasons

    # ── Señal ──────────────────────────────────────────────────────────────────

    def _signal_score(
        self, val_score, rec_mean, current, ma50, ma200, target,
        rev_growth, earn_growth,
    ) -> Tuple[int, str, List[str]]:
        score = val_score
        reasons: List[str] = []

        if rec_mean is not None:
            if rec_mean <= 1.5:
                score += 3
                reasons.append("Consenso analistas: Compra Fuerte.")
            elif rec_mean <= 2.2:
                score += 2
                reasons.append("Consenso analistas: Compra.")
            elif rec_mean <= 2.8:
                score += 1
                reasons.append("Consenso analistas: Mantener con sesgo positivo.")
            elif rec_mean <= 3.5:
                score -= 1
                reasons.append("Consenso analistas: Mantener.")
            elif rec_mean <= 4.2:
                score -= 2
                reasons.append("Consenso analistas: Venta.")
            else:
                score -= 3
                reasons.append("Consenso analistas: Venta Fuerte.")

        if current and ma200:
            diff = (current / ma200 - 1) * 100
            if diff > 5:
                score += 1
                reasons.append(
                    f"Precio un {diff:.1f}% por encima de la media móvil de 200 días — tendencia alcista confirmada."
                )
            elif diff < -5:
                score -= 1
                reasons.append(
                    f"Precio un {abs(diff):.1f}% por debajo de la media móvil de 200 días — tendencia bajista."
                )

        if ma50 and ma200:
            if ma50 > ma200:
                score += 1
                reasons.append("Golden Cross vigente: MA50 cruza al alza sobre MA200.")
            else:
                score -= 1
                reasons.append("Death Cross activo: MA50 por debajo de MA200.")

        if rev_growth is not None:
            rg = float(str(rev_growth).replace("%", "")) / 100 if isinstance(rev_growth, str) else rev_growth
            if rg > 0.15:
                score += 1
                reasons.append(f"Crecimiento de ingresos sólido: +{rg*100:.1f}% YoY.")
            elif rg < 0:
                score -= 1
                reasons.append(f"Contracción de ingresos: {rg*100:.1f}% YoY.")

        if earn_growth is not None:
            eg = float(str(earn_growth).replace("%", "")) / 100 if isinstance(earn_growth, str) else earn_growth
            if eg > 0.20:
                score += 1
                reasons.append(f"Crecimiento de utilidades destacado: +{eg*100:.1f}% YoY.")
            elif eg < -0.10:
                score -= 1
                reasons.append(f"Deterioro de utilidades: {eg*100:.1f}% YoY.")

        if score >= 5:
            signal = "COMPRAR"
        elif score >= 1:
            signal = "MANTENER"
        else:
            signal = "VENDER"

        return score, signal, reasons
