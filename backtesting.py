import numpy as np
import pandas as pd
import yfinance as yf
import logging
import os
from dataclasses import dataclass, field

from utils import get_month_offset
from momentum import compute_momentum_scores

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252

@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    portfolio_nav: pd.Series  # DatetimeIndex -> NAV
    benchmark_nav: pd.Series | None  # DatetimeIndex -> NAV (if benchmark provided)
    rebalance_log: list[dict]  # [{date, tickers, weights}, ...]
    metrics: dict  # CAGR, Sharpe, etc.
    benchmark_metrics: dict = field(default_factory=dict)

def _compute_metrics(nav: pd.Series) -> dict:
    """Compute portfolio performance metrics from a NAV series."""
    if nav.empty or len(nav) < 2:
        return {}

    total_return = (nav.iloc[-1] / nav.iloc[0]) - 1
    days = (nav.index[-1] - nav.index[0]).days
    years = days / 365.25

    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    daily_returns = nav.pct_change().dropna()
    if daily_returns.empty:
        return {"Total Return": total_return, "CAGR": cagr}

    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (cagr - 0.05) / ann_vol if ann_vol > 0 else 0.0  # Assume 5% risk-free

    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 0.0
    sortino = (cagr - 0.05) / downside_vol if downside_vol > 0 else 0.0

    # Max drawdown
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
    }

def _get_rebalance_dates(
    start_year: int, start_month: int,
    end_year: int, end_month: int,
    frequency_months: int,
) -> list[tuple[int, int]]:
    """Generate list of (year, month) rebalancing dates."""
    dates = []
    y, m = start_year, start_month
    end_total = end_year * 12 + end_month
    while y * 12 + m <= end_total:
        dates.append((y, m))
        y, m = get_month_offset(y, m, frequency_months)
    return dates

def run_backtest(
    stock_data: dict[str, pd.DataFrame],
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    frequency_months: int = 3,
    top_n: int = 50,
    weight_12m: float = 0.5,
    weight_3m: float = 0.0,
    absolute_momentum: bool = False,
    volatility_cap: float = 0.0,
    weighting_scheme: str = "equal",
    benchmark_ticker: str | None = None,
    cache_dir: str = "stock_data",
) -> BacktestResult:
    """
    Run a momentum backtest over multiple rebalancing periods.

    For each rebalancing date, selects top_n stocks using momentum scoring,
    weights them according to weighting_scheme, and tracks portfolio NAV.

    Parameters
    ----------
    weighting_scheme : "equal" for equal-weight, "inv_vol" for inverse-volatility weight
    """

    rebalance_dates = _get_rebalance_dates(
        start_year, start_month, end_year, end_month, frequency_months
    )

    if len(rebalance_dates) < 2:
        raise ValueError("Need at least 2 rebalancing dates for a backtest")

    # Build a combined price matrix (Adj Close) for all tickers
    price_col = "Adj Close"
    all_prices = {}
    for ticker, df in stock_data.items():
        if price_col in df.columns:
            all_prices[ticker] = df[price_col]
        elif "Close" in df.columns:
            all_prices[ticker] = df["Close"]

    if not all_prices:
        raise ValueError("No price data available for backtest")

    price_matrix = pd.DataFrame(all_prices)
    price_matrix = price_matrix.sort_index()

    nav_series = []
    rebalance_log = []
    current_nav = 1.0

    for idx in range(len(rebalance_dates) - 1):
        ry, rm = rebalance_dates[idx]
        next_ry, next_rm = rebalance_dates[idx + 1]

        # Score and select top stocks at this rebalance date
        results, _ = compute_momentum_scores(
            stock_data, ry, rm, top_n=top_n, weight_12m=weight_12m,
            weight_3m=weight_3m,
            absolute_momentum=absolute_momentum,
            volatility_cap=volatility_cap,
        )

        if results.empty:
            logger.warning("No eligible stocks for %d-%02d, carrying forward NAV", ry, rm)
            continue

        selected_tickers = results.head(top_n)["Ticker"].tolist()

        # Find actual holding period in price data
        # Start: first trading day of rebalance month
        # End: last trading day before next rebalance month
        hold_start = pd.Timestamp(year=ry, month=rm, day=1)
        y_end_hold, m_end_hold = get_month_offset(next_ry, next_rm, 0)
        hold_end = pd.Timestamp(year=y_end_hold, month=m_end_hold, day=1) - pd.Timedelta(days=1)

        # Get prices for selected tickers during holding period
        available = [t for t in selected_tickers if t in price_matrix.columns]
        if not available:
            logger.warning("None of the selected tickers have price data for %d-%02d", ry, rm)
            continue

        period_prices = price_matrix.loc[hold_start:hold_end, available].dropna(how="all")
        if period_prices.empty or len(period_prices) < 2:
            continue

        # Forward-fill missing prices within the period
        period_prices = period_prices.ffill()

        daily_returns = period_prices.pct_change()

        # Determine portfolio weights
        if weighting_scheme == "inv_vol" and len(available) > 1:
            # Inverse-volatility weighting: allocate more to lower-vol stocks
            vol_window = period_prices.iloc[:min(20, len(period_prices))]
            if len(vol_window) >= 2:
                stock_vols = vol_window.pct_change().std()
                stock_vols = stock_vols.replace(0, np.nan).dropna()
                if len(stock_vols) > 0:
                    inv_vol = 1.0 / stock_vols
                    weights = inv_vol / inv_vol.sum()
                else:
                    weights = pd.Series(1.0 / len(available), index=available)
            else:
                weights = pd.Series(1.0 / len(available), index=available)
            portfolio_daily = daily_returns[weights.index].mul(weights, axis=1).sum(axis=1).dropna()
        else:
            # Equal-weighting: define `weights` so later code can use it
            weights = pd.Series(1.0 / len(available), index=available)
            portfolio_daily = daily_returns[weights.index].mul(weights, axis=1).sum(axis=1).dropna()

        # Cash allocation: if fewer stocks passed filters than top_n,
        # scale equity returns proportionally (rest earns 0 = cash)
        actual_count = len(available)
        if actual_count < top_n:
            equity_fraction = actual_count / top_n
            portfolio_daily = portfolio_daily * equity_fraction
            stock_weights = (weights * equity_fraction).to_dict()
        else:
            stock_weights = weights.to_dict()

        # Build NAV for this period
        for date, ret in portfolio_daily.items():
            current_nav *= (1 + ret)
            nav_series.append((date, current_nav))

        rebalance_log.append({
            "date": f"{ry}-{rm:02d}",
            "tickers": available,
            "weights": stock_weights,
            "count": len(available),
        })

    if not nav_series:
        raise ValueError("Backtest produced no results – check data coverage")

    portfolio_nav = pd.Series(
        [v for _, v in nav_series],
        index=pd.DatetimeIndex([d for d, _ in nav_series]),
        name="Portfolio NAV",
    )

    # Remove duplicate index entries (if any) keeping last
    portfolio_nav = portfolio_nav[~portfolio_nav.index.duplicated(keep="last")]

    metrics = _compute_metrics(portfolio_nav)

    # Benchmark
    benchmark_nav = None
    benchmark_metrics = {}
    if benchmark_ticker:
        try:
            bench_start = portfolio_nav.index[0] - pd.Timedelta(days=5)
            bench_end = portfolio_nav.index[-1] + pd.Timedelta(days=1)
            bench_data = None
            bench_csv = os.path.join(cache_dir, f"{benchmark_ticker}.csv") if cache_dir else None

            # Try loading from cache
            if bench_csv and os.path.isfile(bench_csv):
                try:
                    cached = pd.read_csv(bench_csv, encoding="utf-8-sig", parse_dates=["Date"], index_col="Date")
                    if (not cached.empty
                            and cached.index.min() <= bench_start
                            and cached.index.max() >= bench_end - pd.Timedelta(days=5)):
                        bench_data = cached
                        logger.info("Loaded benchmark %s from cache", benchmark_ticker)
                except Exception:
                    pass  # fall through to download

            # Download if not cached or cache insufficient
            if bench_data is None:
                bench_data = yf.download(
                    benchmark_ticker,
                    start=bench_start.strftime("%Y-%m-%d"),
                    end=bench_end.strftime("%Y-%m-%d"),
                    auto_adjust=False,
                    progress=False,
                )
                # Save to cache for future runs
                if not bench_data.empty and cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                    bench_data.to_csv(bench_csv)
                    logger.info("Saved benchmark %s to cache", benchmark_ticker)
            if not bench_data.empty and "Adj Close" in bench_data.columns:
                bench_prices = bench_data["Adj Close"].dropna()
                # Normalize to start at 1.0 on the portfolio start date
                bench_prices = bench_prices.reindex(portfolio_nav.index, method="ffill")
                bench_prices = bench_prices.dropna()
                if len(bench_prices) > 1:
                    benchmark_nav = bench_prices / bench_prices.iloc[0]
                    benchmark_nav.name = "Benchmark NAV"
                    benchmark_metrics = _compute_metrics(benchmark_nav)
        except Exception as e:
            logger.warning("Failed to download benchmark data for %s: %s", benchmark_ticker, e)

    return BacktestResult(
        portfolio_nav=portfolio_nav,
        benchmark_nav=benchmark_nav,
        rebalance_log=rebalance_log,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
    )