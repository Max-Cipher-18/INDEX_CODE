import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
import calendar
from datetime import datetime

from utils import load_stock_data, download_stock_data, load_stock_data_from_uploads, save_stock_data, load_cached_if_valid
from momentum import compute_momentum_scores
from backtesting import run_backtest

# Configure logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Super Index – Momentum 50",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Super Index – Momentum 50 Stock Selector")
st.markdown(
    "Select the **top 50 momentum stocks** from your universe based on "
    "6-month and 12-month risk-adjusted momentum Z-scores."
)

# ——— Sidebar: Inputs ————————————————————————————————————————————————————————————

st.sidebar.header("Configuration")

data_source = st.sidebar.radio(
    "📁 Data Source",
    ["Download from Yahoo Finance", "Upload CSV Files", "Load from Folder"],
    index=0,
    help="Choose how to provide stock price data.",
)

if data_source == "Download from Yahoo Finance":
    ticker_input = st.sidebar.text_area(
        "🏷️ Ticker Symbols",
        value="",
        height=120,
        help=(
            "Enter ticker symbols separated by commas or newlines.\n"
            "For NSE stocks, append .NS (e.g., RELIANCE.NS, TCS.NS).\n"
            "For BSE stocks, append .BO (e.g., RELIANCE.BO)."
        ),
    )
    save_to_disk = st.sidebar.checkbox(
        "💾 Save downloaded data to disk",
        value=False,
        help="Saves each stock's data as a CSV in the 'stock_data' folder for faster loading next time.",
    )
    if save_to_disk:
        save_folder = st.sidebar.text_input(
            "📂 Save Folder",
            value="stock_data",
            help="Path to the folder where downloaded data will be saved.",
        )
elif data_source == "Upload CSV Files":
    uploaded_files = st.sidebar.file_uploader(
        "📄 Upload stock CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one CSV per stock (Yahoo-style OHLCV). File name = ticker.",
    )
else:  # Load from Folder
    folder_path = st.sidebar.text_input(
        "📂 Path to stock CSV folder",
        value="",
        help="Absolute path to the folder containing one CSV per stock (Yahoo-style OHLCV).",
    )

current_year = datetime.now().year
years = list(range(current_year - 5, current_year + 2))
months = list(range(1, 13))

rebal_year = st.sidebar.selectbox(
    "🗓️ Rebalancing Year",
    years,
    index=years.index(current_year),
)

rebal_month = st.sidebar.selectbox(
    "🗓️ Rebalancing Month",
    months,
    index=4,  # Default to May
    format_func=lambda m: calendar.month_name[m],
)

top_n = st.sidebar.number_input(
    "🏆 Number of top stocks to select",
    min_value=1,
    max_value=500,
    value=50,
)

weight_12m = st.sidebar.slider(
    "⚖️ 12-Month Z Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Weight for 12-month momentum Z-score.",
)

weight_3m = st.sidebar.slider(
    "⚖️ 3-Month Z Weight",
    min_value=0.0,
    max_value=1.0 - weight_12m,
    value=0.0,
    step=0.05,
    help="Weight for 3-month momentum Z-score. 6-month gets the remainder (1 - W12 - W3). " \
    "Set to 0 to disable.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Risk Filters")

absolute_momentum = st.sidebar.checkbox(
    "Absolute Momentum Filter",
    value=False,
    help="Exclude stocks with negative 12-month return. Protects against bear market losses.",
)

volatility_cap = st.sidebar.number_input(
    "Volatility Cap (0 = off)",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.05,
    format="%.2f",
    help="Maximum annualized volatility allowed (e.g., 0.80 = 80%). Set to 0 to disable.",
)

st.sidebar.markdown("---")
compute_btn = st.sidebar.button("🚀 Compute Index", type="primary", use_container_width=True)

# ——— Sidebar: Backtesting ——————————————————————————————————————————————————————

st.sidebar.markdown("---")
st.sidebar.header("📈 Backtesting")

enable_backtest = st.sidebar.checkbox("Enable Backtesting", value=False)

if enable_backtest:
    bt_start_year = st.sidebar.selectbox(
        "Backtest Start Year",
        list(range(current_year - 10, current_year)),
        index=7,  # ~3 years ago
        key="bt_start_year",
    )
    bt_start_month = st.sidebar.selectbox(
        "Backtest Start Month",
        months,
        index=0,
        format_func=lambda m: calendar.month_name[m],
        key="bt_start_month",
    )
    bt_end_year = st.sidebar.selectbox(
        "Backtest End Year",
        list(range(bt_start_year, current_year + 1)),
        index=min(len(list(range(bt_start_year, current_year + 1))) - 1, 10),
        key="bt_end_year",
    )
    bt_end_month = st.sidebar.selectbox(
        "Backtest End Month",
        months,
        index=min(datetime.now().month - 1, 11),
        format_func=lambda m: calendar.month_name[m],
        key="bt_end_month",
    )
    bt_frequency = st.sidebar.selectbox(
        "Rebalancing Frequency",
        [1, 3, 6, 12],
        index=1,
        format_func=lambda f: {1: "Monthly", 3: "Quarterly", 6: "Semi-Annual", 12: "Annual"}[f],
    )
    bt_benchmark = st.sidebar.text_input(
        "Benchmark Ticker",
        value="^NSEI",
        help="Yahoo Finance ticker for benchmark (e.g., ^NSEI for Nifty 50, ^GSPC for S&P 500).",
    )
    bt_weighting = st.sidebar.selectbox(
        "Portfolio Weighting",
        ["equal", "inv_vol"],
        index=0,
        format_func=lambda w: {"equal": "Equal Weight", "inv_vol": "Inverse Volatility"}[w],
        help="Equal weight gives each stock the same allocation. Inverse volatility allocates more to lower-risk stocks.",
    )
    backtest_btn = st.sidebar.button("📊 Run Backtest", use_container_width=True)
else:
    backtest_btn = False

# ——— Main Area ————————————————————————————————————————————————————————————————

if compute_btn:
    stock_data = {}

    if data_source == "Download from Yahoo Finance":
        raw_tickers = ticker_input.strip()
        if not raw_tickers:
            st.error("Please enter at least one ticker symbol.")
            st.stop()
        tickers = [t.strip() for t in raw_tickers.replace("\n", ",").split(",") if t.strip()]
        if not tickers:
            st.error("No valid ticker symbols found. Separate tickers with commas or newlines.")
            st.stop()

        progress_bar = st.progress(0, text="Downloading stock data from Yahoo Finance...")

        def update_progress(done, total, msg):
            progress_bar.progress(done / total, text=f"{msg} ({done}/{total} tickers)")

        stock_data = download_stock_data(tickers, rebal_year, rebal_month, progress_callback=update_progress)
        progress_bar.empty()

        if not stock_data:
            st.error("Failed to download data for any of the provided tickers. Check the symbols and try again.")
            st.stop()

        st.info(f"Downloaded **{len(stock_data)}** / {len(tickers)} tickers from Yahoo Finance.")

        if save_to_disk:
            n = save_stock_data(stock_data, save_folder)
            st.success(f"Saved {n} stocks to '{save_folder}' for future use.")

    elif data_source == "Upload CSV Files":
        if not uploaded_files:
            st.error("Please upload at least one CSV file.")
            st.stop()

        with st.spinner("Loading uploaded CSV files..."):
            stock_data = load_stock_data_from_uploads(uploaded_files)

        if not stock_data:
            st.error(
                "No valid stock data found in uploaded files. "
                "Ensure CSVs have columns: Date, Open, High, Low, Close, Adj Close, Volume."
            )
            st.stop()

        st.info(f"Loaded **{len(stock_data)}** stocks from uploaded files.")

    else:  # Load from Folder
        if not folder_path.strip():
            st.error("Please enter a valid folder path in the sidebar.")
            st.stop()

        with st.spinner("Loading stock CSV files..."):
            stock_data = load_stock_data(folder_path.strip())

        if not stock_data:
            st.error(
                f"No valid stock CSVs found in **{folder_path}**. "
                "Ensure the folder contains .csv files with columns: Date, Open, High, Low, Close, Adj Close, Volume."
            )
            st.stop()

        st.info(f"Loaded **{len(stock_data)}** stocks from the folder.")

    # Compute scores
    with st.spinner("Computing momentum scores..."):
        results, excluded = compute_momentum_scores(
            stock_data, rebal_year, rebal_month, top_n=top_n, weight_12m=weight_12m,
            weight_3m=weight_3m,
            absolute_momentum=absolute_momentum,
            volatility_cap=volatility_cap,
        )

    if results.empty:
        st.error("No eligible stocks found after applying filters. Check excluded stocks below.")
        if excluded:
            with st.expander(f"❌ Excluded Stocks ({len(excluded)})"):
                for reason in excluded:
                    st.text(reason)
        st.stop()

    # ——— Summary Metrics ——————————————————————————————————————————————————————————

    total_loaded = len(stock_data)
    total_eligible = len(results)
    total_excluded = len(excluded)
    selected_count = min(top_n, total_eligible)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks Loaded", total_loaded)
    col2.metric("Eligible Stocks", total_eligible)
    col3.metric("Excluded", total_excluded)
    col4.metric("Selected (Top N)", selected_count)

    st.markdown(
        f"**Rebalancing Month:** {calendar.month_name[rebal_month]} {rebal_year}"
    )

    # ——— Top N Table ——————————————————————————————————————————————————————————————

    st.subheader(f"🏆 Top {selected_count} Momentum Stocks")

    top_df = results.head(top_n).copy()

    # Format for display
    display_df = top_df.copy()
    display_df["12M Return"] = display_df["12M Return"].map("{:.2%}".format)
    display_df["6M Return"] = display_df["6M Return"].map("{:.2%}".format)
    if "3M Return" in display_df.columns:
        display_df["3M Return"] = display_df["3M Return"].map("{:.2%}".format)
    
    fmt_cols = ["σ_p (Ann. Vol)", "MR12", "MR6", "Z12", "Z6", "Wgt Avg Z", "Normalized Score"]
    if "MR3" in display_df.columns:
        fmt_cols.extend(["MR3"])
    if "Z3" in display_df.columns:
        fmt_cols.extend(["Z3"])
        
    for col in fmt_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map("{:.4f}".format)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ——— Download CSV —————————————————————————————————————————————————————————————

    csv_data = top_df.to_csv(index=False)
    st.download_button(
        label=f"📥 Download Top {selected_count} as CSV",
        data=csv_data,
        file_name=f"Super_Index_Top{selected_count}_{calendar.month_abbr[rebal_month]}_{rebal_year}.csv",
        mime="text/csv",
    )

    # ——— Charts ———————————————————————————————————————————————————————————————————

    st.subheader("📊 Charts")

    tab1, tab2, tab3 = st.tabs([
        "Bar Chart – Normalized Score",
        "Scatter – Z12 vs Z6",
        "Returns Distribution",
    ])

    with tab1:
        fig_bar = px.bar(
            top_df,
            x="Ticker",
            y="Normalized Score",
            title=f"Top {selected_count} Stocks by Normalized Momentum Score",
            color="Normalized Score",
            color_continuous_scale="Viridis",
        )
        fig_bar.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        scatter_df = results.copy()
        scatter_df["Selected"] = scatter_df["Rank"] <= top_n

        fig_scatter = px.scatter(
            scatter_df,
            x="Z12",
            y="Z6",
            color="Selected",
            color_discrete_map={True: "#2ecc71", False: "#95a5a6"},
            hover_data=["Ticker", "Normalized Score", "Wgt Avg Z"],
            title="12-Month vs 6-Month Momentum Z-Scores (Green = Selected)",
            labels={"Z12": "12-Month Momentum Z-Score", "Z6": "6-Month Momentum Z-Score"},
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=results["12M Return"],
            name="12M Return",
            opacity=0.7,
            marker_color="#3498db",
        ))
        fig_ret.add_trace(go.Histogram(
            x=results["6M Return"],
            name="6M Return",
            opacity=0.7,
            marker_color="#e74c3c",
        ))
        fig_ret.update_layout(
            barmode="overlay",
            title="Distribution of Returns (Eligible Universe)",
            xaxis_title="Return",
            yaxis_title="Count",
            height=500,
        )

    # ——— Excluded Stocks ——————————————————————————————————————————————————————————

    if excluded:
        with st.expander(f"❌ Excluded Stocks ({total_excluded})"):
            for reason in excluded:
                st.text(reason)

    # ——— Full Universe Table ——————————————————————————————————————————————————————

    with st.expander(f"📄 Full Eligible Universe ({total_eligible} stocks)"):
        full_display = results.copy()
        full_display["12M Return"] = full_display["12M Return"].map("{:.2%}".format)
        full_display["6M Return"] = full_display["6M Return"].map("{:.2%}".format)
        if "3M Return" in full_display.columns:
            full_display["3M Return"] = full_display["3M Return"].map("{:.2%}".format)
        
        full_fmt_cols = ["σ_p (Ann. Vol)", "MR12", "MR6", "Z12", "Z6", "Wgt Avg Z", "Normalized Score"]
        if "MR3" in full_display.columns:
            full_fmt_cols.append("MR3")
        if "Z3" in full_display.columns:
            full_fmt_cols.append("Z3")
            
        for col in full_fmt_cols:
            if col in full_display.columns:
                full_display[col] = full_display[col].map("{:.4f}".format)
        
        st.dataframe(full_display, use_container_width=True, hide_index=True)

    full_csv = results.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Full Universe as CSV",
        data=full_csv,
        file_name=f"Super_Index_FullUniverse_{calendar.month_abbr[rebal_month]}_{rebal_year}.csv",
        mime="text/csv",
    )

else:
    st.markdown("---")
    st.markdown(
        """
        ### How to use
        
        1. **Choose a data source** in the sidebar.
        2. **Select the rebalancing month and year**.
        3. **Click "Compute Index"**.
        
        ### Methodology Summary
        | Step | Description |
        |------|-------------|
        | 1 | Compute 12-month, 6-month, and (optional) 3-month **price returns** |
        | 2 | Compute **annualized volatility** (σ_p) |
        | 3 | **Absolute Momentum Filter** |
        | 4 | **Volatility Cap** |
        | 5 | Calculate **Momentum Ratios**: MR = Return / σ_p |
        | 6 | Compute **Z-Scores** |
        | 7 | **Weighted Average Z** |
        | 8 | **Normalized Score** |
        | 9 | **Rank** and select top N |
        """
    )

# ——— Backtesting Section ——————————————————————————————————————————————————————

if enable_backtest and backtest_btn:
    st.markdown("---")
    st.header("📊 Backtest Results")
    
    # Load data for backtest (same source selection as above)
    bt_stock_data = {}
    
    if data_source == "Download from Yahoo Finance":
        raw_tickers = ticker_input.strip()
        if not raw_tickers:
            st.error("Please enter at least one ticker symbol for backtesting.")
            st.stop()
        tickers = [t.strip() for t in raw_tickers.replace("\n", ",").split(",") if t.strip()]
        if not tickers:
            st.error("No valid ticker symbols found.")
            st.stop()
            
        # Download extended date range for backtest
        from utils import get_month_offset as _gmo
        y_s, m_s = _gmo(bt_start_year, bt_start_month, -15)
        start_dl = f"{y_s}-{m_s:02d}-01"
        y_e, m_e = _gmo(bt_end_year, bt_end_month, 1)
        end_dl = f"{y_e}-{m_e:02d}-01"
        
        # Use cache-aware downloader: load cached files first, then download missing tickers
        cache_folder = save_folder if save_to_disk else "stock_data"

        # Try loading from disk cache first
        cached_data, needed_tickers = load_cached_if_valid(
            cache_folder, tickers, start_dl, end_dl,
        )
        bt_stock_data.update(cached_data)

        if needed_tickers:
            import yfinance as yf_dl
            progress_bar = st.progress(0, text=f"Downloading {len(needed_tickers)} tickers (cached {len(cached_data)})...")
            
            try:
                raw = yf_dl.download(
                    tickers=needed_tickers,
                    start=start_dl,
                    end=end_dl,
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                )
                progress_bar.progress(0.8, text="Processing downloaded data...")
                newly_downloaded: dict[str, pd.DataFrame] = {}
                if not raw.empty:
                    if len(needed_tickers) == 1:
                        ticker = needed_tickers[0]
                        ticker_df = raw.dropna(how="all")
                        if not ticker_df.empty:
                            ticker_df.index.name = "Date"
                            newly_downloaded[ticker] = ticker_df
                    else:
                        for ticker in needed_tickers:
                            try:
                                ticker_df = raw.xs(ticker, level="Ticker", axis=1).dropna(how="all")
                                if not ticker_df.empty:
                                    ticker_df.index.name = "Date"
                                    newly_downloaded[ticker] = ticker_df
                            except KeyError:
                                pass

                bt_stock_data.update(newly_downloaded)

                # Save newly downloaded data to cache
                if newly_downloaded:
                    save_stock_data(newly_downloaded, cache_folder)

            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()

            progress_bar.progress(1.0, text="Download complete")
            progress_bar.empty()

            st.info(
                f"Backtest data: **{len(bt_stock_data)}** / {len(tickers)} tickers "
                f"({len(cached_data)} from cache, {len(bt_stock_data) - len(cached_data)} downloaded)."
            )
        
    elif data_source == "Upload CSV Files":
        if not uploaded_files:
            st.error("Please upload CSV files for backtesting.")
            st.stop()
        bt_stock_data = load_stock_data_from_uploads(uploaded_files)
    else:
        if not folder_path.strip():
            st.error("Please enter a folder path for backtesting.")
            st.stop()
        bt_stock_data = load_stock_data(folder_path.strip())
        
    if not bt_stock_data:
        st.error("No stock data available for backtesting.")
        st.stop()

    # Determine cache directory for benchmark data
    if data_source == "Download from Yahoo Finance":
        bt_cache_dir = save_folder if save_to_disk else "stock_data"
    elif data_source == "Load from Folder":
        bt_cache_dir = folder_path.strip()
    else:
        bt_cache_dir = "stock_data"
    
    # Run the backtest
    with st.spinner("Running backtest... This may take a while for large universes."):
        try:
            bt_result = run_backtest(
                stock_data=bt_stock_data,
                start_year=bt_start_year,
                start_month=bt_start_month,
                end_year=bt_end_year,
                end_month=bt_end_month,
                frequency_months=bt_frequency,
                top_n=top_n,
                weight_12m=weight_12m,
                weight_3m=weight_3m,
                absolute_momentum=absolute_momentum,
                volatility_cap=volatility_cap,
                weighting_scheme=bt_weighting,
                benchmark_ticker=bt_benchmark.strip() if bt_benchmark.strip() else None,
                cache_dir=bt_cache_dir,
            )
        except ValueError as e:
            st.error(f"Backtest error: {e}")
            st.stop()

    # —— Backtest Metrics ———————————————————————————————————————————————————————

    st.subheader("📋 Performance Metrics")

    metric_cols = st.columns(4)
    m = bt_result.metrics
    metric_cols[0].metric("CAGR", f"{m.get('CAGR', 0):.2%}")
    metric_cols[1].metric("Sharpe Ratio", f"{m.get('Sharpe Ratio', 0):.2f}")
    metric_cols[2].metric("Max Drawdown", f"{m.get('Max Drawdown', 0):.2%}")
    metric_cols[3].metric("Total Return", f"{m.get('Total Return', 0):.2%}")

    if bt_result.benchmark_metrics:
        st.markdown("**Benchmark:**")
        bm_cols = st.columns(4)
        bm = bt_result.benchmark_metrics
        bm_cols[0].metric("CAGR", f"{bm.get('CAGR', 0):.2%}")
        bm_cols[1].metric("Sharpe Ratio", f"{bm.get('Sharpe Ratio', 0):.2f}")
        bm_cols[2].metric("Max Drawdown", f"{bm.get('Max Drawdown', 0):.2%}")
        bm_cols[3].metric("Total Return", f"{bm.get('Total Return', 0):.2%}")

    # Additional metrics table
    all_metrics = {"Metric": [], "Portfolio": []}
    if bt_result.benchmark_metrics:
        all_metrics["Benchmark"] = []

    for key in ["Total Return", "CAGR", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio"]:
        all_metrics["Metric"].append(key)
        val = m.get(key, 0)
        if key in ("Total Return", "CAGR", "Ann. Volatility", "Max Drawdown"):
            all_metrics["Portfolio"].append(f"{val:.2%}")
        else:
            all_metrics["Portfolio"].append(f"{val:.2f}")
            
        if bt_result.benchmark_metrics:
            bval = bm.get(key, 0)
            if key in ("Total Return", "CAGR", "Ann. Volatility", "Max Drawdown"):
                all_metrics["Benchmark"].append(f"{bval:.2%}")
            else:
                all_metrics["Benchmark"].append(f"{bval:.2f}")

    st.dataframe(pd.DataFrame(all_metrics), use_container_width=True, hide_index=True)

    # —— NAV Chart ——————————————————————————————————————————————————————————————

    st.subheader("📈 Portfolio NAV")

    fig_nav = go.Figure()
    fig_nav.add_trace(go.Scatter(
        x=bt_result.portfolio_nav.index,
        y=bt_result.portfolio_nav.values,
        name="Momentum Portfolio",
        line=dict(color="#2ecc71", width=2),
    ))

    if bt_result.benchmark_nav is not None:
        fig_nav.add_trace(go.Scatter(
            x=bt_result.benchmark_nav.index,
            y=bt_result.benchmark_nav.values,
            name=f"Benchmark ({bt_benchmark})",
            line=dict(color="#95a5a6", width=2, dash="dash"),
        ))

    fig_nav.update_layout(
        title="Portfolio NAV vs Benchmark (Starting NAV = 1.0)",
        xaxis_title="Date",
        yaxis_title="NAV",
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig_nav, use_container_width=True)

    # —— Drawdown Chart —————————————————————————————————————————————————————————

    st.subheader("📉 Drawdown")

    nav = bt_result.portfolio_nav
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#e74c3c"),
        fillcolor="rgba(231, 76, 60, 0.3)",
    ))

    fig_dd.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=400,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # —— Rebalance Log ————————————————————————————————————————————————————————————

    st.subheader("🔄 Rebalance Portfolio Holdings")

    rebal_log = bt_result.rebalance_log

    if rebal_log:
        tab_start, tab_end, tab_all = st.tabs([
            f"🟢 Start ({rebal_log[0]['date']})",
            f"🔴 End ({rebal_log[-1]['date']})",
            f"📋 All Periods ({len(rebal_log)})",
        ])

        def _render_holdings(entry):
            """Render a single rebalance entry as a styled DataFrame."""
            weights = entry.get("weights", {})
            if weights:
                holdings_df = pd.DataFrame([
                    {"Ticker": t, "Weight": w}
                    for t, w in sorted(weights.items(), key=lambda x: -x[1])
                ])
                holdings_df["Weight %"] = holdings_df["Weight"].map("{:.2%}".format)
                st.dataframe(
                    holdings_df[["Ticker", "Weight %"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.text(", ".join(entry["tickers"]))

        with tab_start:
            st.markdown(f"**{rebal_log[0]['date']}** – {rebal_log[0]['count']} stocks")
            _render_holdings(rebal_log[0])

        with tab_end:
            st.markdown(f"**{rebal_log[-1]['date']}** – {rebal_log[-1]['count']} stocks")
            _render_holdings(rebal_log[-1])

        with tab_all:
            for i, entry in enumerate(rebal_log):
                label = entry['date']
                if i == 0:
                    label += " (Start)"
                elif i == len(rebal_log) - 1:
                    label += " (End)"
                with st.expander(f"**{label}** – {entry['count']} stocks"):
                    _render_holdings(entry)

        # Download all holdings as CSV
        all_holdings_rows = []
        for entry in rebal_log:
            weights = entry.get("weights", {})
            for t, w in weights.items():
                all_holdings_rows.append({
                    "Rebalance Date": entry["date"],
                    "Ticker": t,
                    "Weight": w,
                })
                
        if all_holdings_rows:
            holdings_csv = pd.DataFrame(all_holdings_rows).to_csv(index=False)
            st.download_button(
                label="📥 Download All Holdings as CSV",
                data=holdings_csv,
                file_name=f"Backtest_Holdings_{bt_start_year}-{bt_end_year}.csv",
                mime="text/csv",
            )
            
    # —— Download Backtest Data —————————————————————————————————————————————————

    bt_nav_df = pd.DataFrame({"Date": bt_result.portfolio_nav.index, "Portfolio NAV": bt_result.portfolio_nav.values})

    if bt_result.benchmark_nav is not None:
        bt_nav_df["Benchmark NAV"] = bt_result.benchmark_nav.reindex(bt_result.portfolio_nav.index).values
    bt_csv = bt_nav_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Backtest NAV as CSV",
        data=bt_csv,
        file_name=f"Backtest_NAV_{bt_start_year}-{bt_end_year}.csv",
        mime="text/csv",
    )