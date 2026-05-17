import numpy as np
import pandas as pd
import logging

from utils import get_last_trading_day, get_month_offset

logger = logging.getLogger(__name__)

# Number of trading days in a year (for annualizing volatility)
TRADING_DAYS_PER_YEAR = 252

def compute_momentum_scores(
    stock_data: dict[str, pd.DataFrame],
    rebal_year: int,
    rebal_month: int,
    top_n: int = 50,
    weight_12m: float = 0.5,
    weight_3m: float = 0.0,
    absolute_momentum: bool = False,
    volatility_cap: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute Normalized Momentum Scores for all eligible stocks and return
    the ranked DataFrame plus a list of excluded tickers with reasons.

    Parameters
    ----------
    stock_data : dict mapping ticker -> DataFrame with DatetimeIndex and 'Adj Close'/'Close' column
    rebal_year : the rebalancing year (M)
    rebal_month : the rebalancing month (M)
    top_n : number of top stocks to select (default 50)
    weight_12m : weight for 12-month Z-score
    weight_3m : weight for 3-month Z-score (default 0 = disabled).
                6-month weight becomes 1 - weight_12m - weight_3m.
    absolute_momentum : if True, exclude stocks with negative 12M return
    volatility_cap : if > 0, exclude stocks with annualized vol above this value (e.g. 0.80 = 80%)

    Returns
    -------
    (results_df, excluded) where:
        results_df : DataFrame with all eligible stocks, scored and ranked
        excluded   : list of strings describing excluded tickers and reasons
    """

    weight_6m = 1.0 - weight_12m - weight_3m

    # Determine required months relative to rebal month M
    # M-1 month
    y_m1, m_m1 = get_month_offset(rebal_year, rebal_month, -1)
    # M-4 month (for 3-month return)
    y_m4, m_m4 = get_month_offset(rebal_year, rebal_month, -4)
    # M-7 month (for 6-month return)
    y_m7, m_m7 = get_month_offset(rebal_year, rebal_month, -7)
    # M-13 month (for 12-month return)
    y_m13, m_m13 = get_month_offset(rebal_year, rebal_month, -13)

    records = []
    excluded = []

    for ticker, df in stock_data.items():
        # Use Adj Close for split/dividend-adjusted returns; fall back to Close
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        
        if price_col == "Close" and "Adj Close" not in df.columns:
            logger.debug("%s: 'Adj Close' not available, using 'Close'", ticker)

        # --- Step A: Get required prices ---
        date_m1 = get_last_trading_day(df, y_m1, m_m1)
        date_m4 = get_last_trading_day(df, y_m4, m_m4)
        date_m7 = get_last_trading_day(df, y_m7, m_m7)
        date_m13 = get_last_trading_day(df, y_m13, m_m13)

        if date_m1 is None:
            excluded.append(f"{ticker}: No price data for {y_m1}-{m_m1:02d} (M-1)")
            continue
        if date_m7 is None:
            excluded.append(f"{ticker}: No price data for {y_m7}-{m_m7:02d} (M-7)")
            continue
        if date_m13 is None:
            excluded.append(f"{ticker}: No price data for {y_m13}-{m_m13:02d} (M-13)")
            continue

        # 3-month price is optional - only required when weight_3m > 0
        use_3m = weight_3m > 0
        if use_3m and date_m4 is None:
            excluded.append(f"{ticker}: No price data for {y_m4}-{m_m4:02d} (M-4)")
            continue

        price_m1 = df.loc[date_m1, price_col]
        price_m7 = df.loc[date_m7, price_col]
        price_m13 = df.loc[date_m13, price_col]
        price_m4 = df.loc[date_m4, price_col] if use_3m else None

        # Handle cases where Close might be a Series (duplicate dates)
        if isinstance(price_m1, pd.Series):
            price_m1 = price_m1.iloc[-1]
        if isinstance(price_m7, pd.Series):
            price_m7 = price_m7.iloc[-1]
        if isinstance(price_m13, pd.Series):
            price_m13 = price_m13.iloc[-1]
        if isinstance(price_m4, pd.Series):
            price_m4 = price_m4.iloc[-1]

        # Validate prices are positive
        if price_m1 <= 0 or price_m7 <= 0 or price_m13 <= 0:
            excluded.append(f"{ticker}: Non-positive price encountered")
            continue
        
        if use_3m and (price_m4 is None or price_m4 <= 0):
            excluded.append(f"{ticker}: Non-positive 3M price encountered")
            continue

        # 12-month price return
        ret_12m = (price_m1 / price_m13) - 1
        # 6-month price return
        ret_6m = (price_m1 / price_m7) - 1
        # 3-month price return
        ret_3m = (price_m1 / price_m4) - 1 if use_3m else None

        # --- Absolute Momentum Filter ---
        if absolute_momentum and ret_12m < 0:
            excluded.append(f"{ticker}: Negative 12M return ({ret_12m:.2%}) – filtered by absolute momentum")
            continue

        # --- Step B: Annualized Volatility ---
        # 1-year window of daily data ending at last trading day of M-1
        vol_start = date_m1 - pd.DateOffset(years=1)
        vol_data = df.loc[vol_start:date_m1, price_col]

        if len(vol_data) < 20:  # Need minimum data points
            excluded.append(f"{ticker}: Insufficient data for volatility ({len(vol_data)} days)")
            continue

        # Daily log returns
        log_returns = np.log(vol_data / vol_data.shift(1)).dropna()

        if len(log_returns) < 20:
            excluded.append(f"{ticker}: Insufficient log returns ({len(log_returns)} days)")
            continue

        sigma_daily = log_returns.std()
        sigma_p = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

        if sigma_p <= 0 or np.isnan(sigma_p) or np.isinf(sigma_p):
            excluded.append(f"{ticker}: Invalid volatility (σ_p={sigma_p})")
            continue

        # --- Volatility Cap Filter ---
        if volatility_cap > 0 and sigma_p > volatility_cap:
            excluded.append(f"{ticker}: Volatility {sigma_p:.2%} exceeds cap {volatility_cap:.2%}")
            continue

        # --- Step C: Momentum Ratios ---
        mr_12 = ret_12m / sigma_p
        mr_6 = ret_6m / sigma_p
        mr_3 = ret_3m / sigma_p if use_3m else None

        record = {
            "Ticker": ticker,
            "Price (M-1)": round(price_m1, 2),
            "Price (M-7)": round(price_m7, 2),
            "Price (M-13)": round(price_m13, 2),
            "12M Return": ret_12m,
            "6M Return": ret_6m,
            "σ_p (Ann. Vol)": sigma_p,
            "MR12": mr_12,
            "MR6": mr_6,
        }
        
        if use_3m:
            record["Price (M-4)"] = round(price_m4, 2)
            record["3M Return"] = ret_3m
            record["MR3"] = mr_3
        
        records.append(record)

    if not records:
        logger.warning("No eligible stocks found after filtering")
        return pd.DataFrame(), excluded

    results = pd.DataFrame(records)

    # --- Step D: Z-Scores across the eligible universe ---
    mu_mr12 = results["MR12"].mean()
    sigma_mr12 = results["MR12"].std()
    mu_mr6 = results["MR6"].mean()
    sigma_mr6 = results["MR6"].std()

    if sigma_mr12 == 0 or sigma_mr6 == 0:
        logger.warning("Zero standard deviation in momentum ratios – cannot compute Z-scores")
        return pd.DataFrame(), excluded

    results["Z12"] = (results["MR12"] - mu_mr12) / sigma_mr12
    results["Z6"] = (results["MR6"] - mu_mr6) / sigma_mr6

    # --- Step E: Weighted Average Z Score ---
    if weight_3m > 0 and "MR3" in results.columns:
        mu_mr3 = results["MR3"].mean()
        sigma_mr3 = results["MR3"].std()
        if sigma_mr3 == 0:
            logger.warning("Zero std deviation in 3M momentum ratios – falling back to 2-signal")
            results["Wgt Avg Z"] = weight_12m * results["Z12"] + weight_6m * results["Z6"]
        else:
            results["Z3"] = (results["MR3"] - mu_mr3) / sigma_mr3
            results["Wgt Avg Z"] = (
                weight_12m * results["Z12"]
                + weight_6m * results["Z6"]
                + weight_3m * results["Z3"]
            )
    else:
        results["Wgt Avg Z"] = weight_12m * results["Z12"] + weight_6m * results["Z6"]

    # --- Step F: Normalized Momentum Score ---
    results["Normalized Score"] = results["Wgt Avg Z"].apply(
        lambda z: (1 + z) if z >= 0 else (1 - z) ** (-1)
    )

    # --- Step G: Rank and select top N ---
    results = results.sort_values("Normalized Score", ascending=False).reset_index(drop=True)
    results["Rank"] = range(1, len(results) + 1)

    # Reorder columns for clarity
    col_order = [
        "Rank", "Ticker",
        "Price (M-1)", "Price (M-7)", "Price (M-13)",
        "12M Return", "6M Return", "σ_p (Ann. Vol)",
        "MR12", "MR6", "Z12", "Z6",
    ]
    
    if weight_3m > 0 and "MR3" in results.columns:
        col_order.insert(col_order.index("Price (M-13)") + 1, "Price (M-4)")
        col_order.insert(col_order.index("6M Return") + 1, "3M Return")
        col_order.insert(col_order.index("MR6") + 1, "MR3")
        if "Z3" in results.columns:
            col_order.insert(col_order.index("Z6") + 1, "Z3")
            
    col_order.extend(["Wgt Avg Z", "Normalized Score"])
    results = results[col_order]

    # Format percentage columns
    for col in ["12M Return", "6M Return"]:
        results[col] = results[col].astype(float)

    logger.info(
        "Computed scores for %d eligible stocks. Top %d selected.",
        len(results), min(top_n, len(results))
    )

    return results, excluded