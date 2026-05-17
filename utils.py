import os
import io
import time
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# Canonical column name mapping (lowercase -> title-case)
_COLUMN_MAP = {
    "close": "Close",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "volume": "Volume",
    "adj close": "Adj Close",
    "adjclose": "Adj Close",
}


def _parse_stock_csv(df: pd.DataFrame, source_name: str) -> pd.DataFrame | None:
    """
    Shared CSV parsing: normalize columns, set DatetimeIndex, validate
    required columns. Returns cleaned DataFrame or None on failure.
    """
    df.columns = df.columns.str.strip()

    # Normalize column names to canonical casing
    rename = {}
    for col in df.columns:
        canonical = _COLUMN_MAP.get(col.lower())
        if canonical and col != canonical:
            rename[col] = canonical

    if rename:
        df = df.rename(columns=rename)

    # Identify the date column
    date_col = None
    for col in df.columns:
        if col.lower() in ("date", "timestamp", "trade date"):
            date_col = col
            break

    if date_col is None:
        logger.warning("No date column found in %s – skipping", source_name)
        return None

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    df.index.name = "Date"

    # Ensure Close column exists
    if "Close" not in df.columns:
        logger.warning("No 'Close' column in %s – skipping", source_name)
        return None

    return df


def _read_csv_with_fallback(filepath_or_buffer, source_name: str) -> pd.DataFrame | None:
    """Read CSV trying utf-8-sig first, falling back to latin-1 on decode error."""
    try:
        return pd.read_csv(filepath_or_buffer, encoding="utf-8-sig")
    except UnicodeDecodeError:
        logger.debug("UTF-8 decode failed for %s, retrying with latin-1", source_name)
        if hasattr(filepath_or_buffer, "seek"):
            filepath_or_buffer.seek(0)
        return pd.read_csv(filepath_or_buffer, encoding="latin-1")


def load_stock_data(folder_path: str) -> dict[str, pd.DataFrame]:
    """
    Scan a folder for .csv files and load each as a stock price DataFrame.
    Each CSV is expected to have Yahoo-style columns:
      Date, Open, High, Low, Close, Adj Close, Volume
    The ticker name is derived from the filename (without .csv extension).
    Returns a dict mapping ticker -> DataFrame (sorted by Date, DatetimeIndex).
    """
    stock_data: dict[str, pd.DataFrame] = {}

    if not os.path.isdir(folder_path):
        logger.error("Folder not found: %s", folder_path)
        return stock_data

    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]

    if not csv_files:
        logger.warning("No CSV files found in: %s", folder_path)
        return stock_data

    for filename in csv_files:
        ticker = os.path.splitext(filename)[0]
        filepath = os.path.join(folder_path, filename)
        try:
            df = _read_csv_with_fallback(filepath, filename)
            if df is None or df.empty:
                logger.warning("Empty or unreadable file: %s – skipping", filename)
                continue

            parsed = _parse_stock_csv(df, filename)
            if parsed is not None:
                stock_data[ticker] = parsed
                logger.debug("Loaded %s: %d rows", ticker, len(parsed))

        except Exception as e:
            logger.warning("Failed to load %s: %s", filename, e)
            continue

    logger.info("Loaded %d stocks from %s", len(stock_data), folder_path)
    return stock_data


def get_last_trading_day(df: pd.DataFrame, year: int, month: int) -> pd.Timestamp | None:
    """
    Return the last available trading date in the given year/month
    from the stock's DatetimeIndex. Returns None if no data exists for that month.
    """
    mask = (df.index.year == year) & (df.index.month == month)
    filtered = df.index[mask]
    if filtered.empty:
        return None
    return filtered.max()


def get_month_offset(year: int, month: int, offset: int) -> tuple[int, int]:
    """
    Given a (year, month) and an integer offset (can be negative),
    return the (new_year, new_month) after applying the offset.
    E.g. get_month_offset(2026, 1, -1) -> (2025, 12)
    """
    # Convert to 0-based month for arithmetic
    total_months = year * 12 + (month - 1) + offset
    new_year = total_months // 12
    new_month = total_months % 12 + 1
    return new_year, new_month


def download_stock_data(
    tickers: list[str],
    rebal_year: int,
    rebal_month: int,
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    """
    Download daily price data from Yahoo Finance for a list of tickers.
    Date range is auto-calculated: 15 months before rebalancing month to M-1.
    Returns dict[str, DataFrame] compatible with compute_momentum_scores.
    """
    y_start, m_start = get_month_offset(rebal_year, rebal_month, -15)
    start_date = f"{y_start}-{m_start:02d}-01"
    y_end, m_end = get_month_offset(rebal_year, rebal_month, 0)
    end_date = f"{y_end}-{m_end:02d}-01"

    stock_data: dict[str, pd.DataFrame] = {}

    # Process in chunks for progress reporting
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        if progress_callback:
            progress_callback(
                min(i, len(tickers)),
                len(tickers),
                f"Downloading batch {i // chunk_size + 1}...",
            )

        try:
            raw = yf.download(
                tickers=chunk,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as e:
            logger.warning("Batch download failed (tickers %d-%d): %s", i, i + len(chunk), e)
            continue

        if raw.empty:
            logger.warning("No data returned for batch %d", i // chunk_size + 1)
            continue

        # yf.download returns MultiIndex columns for multiple tickers
        if len(chunk) == 1:
            ticker = chunk[0]
            ticker_df = raw.dropna(how="all")
            if not ticker_df.empty:
                ticker_df.index.name = "Date"
                stock_data[ticker] = ticker_df
                logger.debug("Downloaded %s: %d rows", ticker, len(ticker_df))
        else:
            for ticker in chunk:
                try:
                    ticker_df = raw.xs(ticker, level="Ticker", axis=1).dropna(how="all")
                    if not ticker_df.empty:
                        ticker_df.index.name = "Date"
                        stock_data[ticker] = ticker_df
                        logger.debug("Downloaded %s: %d rows", ticker, len(ticker_df))
                except KeyError:
                    logger.warning("No data returned for %s", ticker)

        # Rate-limit pause between batches to avoid hitting Yahoo Finance limits
        if i + chunk_size < len(tickers):
            time.sleep(2)

    if progress_callback:
        progress_callback(len(tickers), len(tickers), "Download complete")

    logger.info("Downloaded %d / %d tickers from Yahoo Finance", len(stock_data), len(tickers))
    return stock_data

def save_stock_data(stock_data: dict[str, pd.DataFrame], output_dir: str) -> int:
    """
    Save stock data DataFrames to CSV files in the specified folder.
    Each ticker's DataFrame is saved as TICKER.csv with Yahoo-style columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for ticker, df in stock_data.items():
        filename = f"{ticker}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath)
        saved += 1
    logger.info("Saved %d stocks to %s", saved, output_dir)
    return saved

def load_cached_if_valid(
    folder: str,
    tickers: list[str],
    required_start: str,
    required_end: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Load cached CSVs for tickers whose date range covers [required_start, required_end].

    Returns (cached_data, missing_tickers) where missing_tickers need downloading.
    A ticker is considered "missing" if its CSV doesn't exist, can't be parsed,
    or doesn't cover the required date range.
    """

    cached: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    req_start = pd.Timestamp(required_start)
    req_end = pd.Timestamp(required_end) - pd.Timedelta(days=5)

    if not os.path.isdir(folder):
        return {}, list(tickers)

    for ticker in tickers:
        path = os.path.join(folder, f"{ticker}.csv")
        if not os.path.isfile(path):
            missing.append(ticker)
            continue
        try:
            raw_df = _read_csv_with_fallback(path, ticker)
            if raw_df is None or raw_df.empty:
                missing.append(ticker)
                continue
            parsed = _parse_stock_csv(raw_df, ticker)
            if parsed is None:
                missing.append(ticker)
                continue
            # Check date coverage
            if parsed.index.min() <= req_start and parsed.index.max() >= req_end:
                cached[ticker] = parsed
            else:
                missing.append(ticker)
        except Exception as e:
            logger.warning("Failed to read cached %s: %s", ticker, e)
            missing.append(ticker)

    logger.info(
        "Cache check: %d cached, %d need download (folder: %s)",
        len(cached), len(missing), folder,
    )
    return cached, missing

def load_stock_data_from_uploads(uploaded_files) -> dict[str, pd.DataFrame]:
    """
    Load stock data from Streamlit UploadedFile objects.
    """
    stock_data: dict[str, pd.DataFrame] = {}

    for uploaded_file in uploaded_files:
        ticker = os.path.splitext(uploaded_file.name)[0]
        try:
            content = uploaded_file.read()
            df = _read_csv_with_fallback(io.BytesIO(content), uploaded_file.name)
            if df is None or df.empty:
                logger.warning("Empty or unreadable upload: %s – skipping", uploaded_file.name)
                continue

            parsed = _parse_stock_csv(df, uploaded_file.name)
            if parsed is not None:
                stock_data[ticker] = parsed
                logger.debug("Loaded upload %s: %d rows", ticker, len(parsed))

        except Exception as e:
            logger.warning("Failed to load uploaded %s: %s", uploaded_file.name, e)
            continue

    logger.info("Loaded %d stocks from uploaded files", len(stock_data))
    return stock_data