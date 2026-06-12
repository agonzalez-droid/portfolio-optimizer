import html as _html
import warnings
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")

# ── Color palette ──────────────────────────────────────────────────────────────
NAV  = "#1a2744"
STL  = "#7c9cbf"
SGE  = "#4a7c3f"
GLD  = "#8a6c1a"
RED  = "#8a3028"
BG   = "#f7f6f2"
CARD = "#eeece8"
MID  = "#8b93a7"
LGT  = "#c0bdb8"

PORT_COLORS = {"Max Sharpe": NAV, "Min Volatility": SGE, "Max Sortino": GLD, "HRP": STL}
SPY_COLOR = RED

# ── CSS ────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    .stApp { background-color: #f7f6f2; color: #1a2744; }
    .stApp * { font-family: "Arial","Helvetica Neue",sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap:0; border-bottom:2px solid #e2e0db; background:transparent; }
    .stTabs [data-baseweb="tab"] {
        font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
        padding:8px 18px; color:#8b93a7; border-bottom:2px solid transparent;
        background:transparent; border-radius:0;
    }
    .stTabs [aria-selected="true"] {
        color:#1a2744 !important; border-bottom:2px solid #1a2744 !important;
        background:transparent !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color:#eeece8; }
    section[data-testid="stSidebar"] label { color:#1a2744 !important; font-size:11px; font-weight:600; }
    section[data-testid="stSidebar"] .stButton button {
        background-color:#1a2744; color:#fff; border:none;
        font-size:11px; font-weight:700; letter-spacing:0.08em;
        text-transform:uppercase; border-radius:4px; padding:10px;
    }
    section[data-testid="stSidebar"] .stButton button:hover { background-color:#253660; }
    .stSlider [role="slider"] { background-color:#1a2744 !important; border-color:#1a2744 !important; }

    /* Metric cards */
    .metric-card {
        background-color:#eeece8; border-radius:6px;
        padding:16px 18px; text-align:center; margin-bottom:4px;
    }
    .metric-label {
        font-size:9px; font-weight:700; letter-spacing:0.12em;
        text-transform:uppercase; color:#8b93a7; margin-bottom:5px;
    }
    .metric-value { font-size:20px; font-weight:700; color:#1a2744; }
    .metric-sub { font-size:10px; color:#b0b8c6; margin-top:2px; }

    /* Section labels */
    .section-label {
        font-size:9px; font-weight:700; letter-spacing:0.14em;
        text-transform:uppercase; color:#8b93a7;
        margin-top:24px; margin-bottom:8px;
        border-bottom:1px solid #e2e0db; padding-bottom:5px;
    }

    /* Tables */
    .mtable-wrap { overflow-x:auto; margin-top:4px; }
    .mtable { width:100%; border-collapse:collapse; font-size:11px; }
    .mtable th {
        background-color:#1a2744; color:#fff; padding:7px 12px;
        text-align:center; font-size:10px; font-weight:700; letter-spacing:0.06em;
    }
    .mtable th.lh { text-align:left; }
    .mtable td { padding:5px 12px; text-align:center; color:#1a2744; }
    .mtable tr:nth-child(even) td { background-color:#f3f1ed; }
    .mtable tr:hover td { background-color:#eae8e4; }
    .mtable td.lh { text-align:left; font-weight:600; color:#8b93a7; }

    /* Error / warning */
    .error-box {
        background-color:#fef2f2; border-left:3px solid #8a3028;
        border-radius:0 6px 6px 0; padding:12px 16px; color:#8a3028; font-size:12px;
    }
    .warn-box {
        background-color:#fffbeb; border-left:3px solid #8a6c1a;
        border-radius:0 6px 6px 0; padding:10px 14px; color:#6b5214;
        font-size:11px; margin:6px 0;
    }

    /* Landing */
    .landing {
        text-align:center; margin-top:100px; color:#b0b8c6;
        font-size:12px; font-weight:700; letter-spacing:0.14em; text-transform:uppercase;
    }

    /* Tear sheet */
    .ts-row { padding:7px 0; border-bottom:1px solid #e2e0db; display:flex; justify-content:space-between; }
    .ts-label { font-size:10px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#8b93a7; }
    .ts-value { font-size:12px; font-weight:700; color:#1a2744; }
</style>
"""

DARK_CSS = """
<style>
    .stApp { background-color: #0d1117 !important; color: #e2e4ed !important; }
    section[data-testid="stSidebar"] { background-color: #0b0f1a !important; }
    section[data-testid="stSidebar"] label { color: #c8cad4 !important; }
    section[data-testid="stSidebar"] p { color: #8b93a7 !important; }
    .stTabs [data-baseweb="tab-list"] { border-bottom-color: #1e2538 !important; }
    .stTabs [data-baseweb="tab"] { color: #8b93a7 !important; }
    .stTabs [aria-selected="true"] { color: #e2e4ed !important; border-bottom-color: #7c9cbf !important; }
    .metric-card { background-color: #161b27 !important; }
    .metric-label { color: #8b93a7 !important; }
    .metric-value { color: #e2e4ed !important; }
    .metric-sub { color: #8b93a7 !important; }
    .section-label { color: #8b93a7 !important; border-bottom-color: #1e2538 !important; }
    .mtable td { color: #e2e4ed !important; }
    .mtable tr:nth-child(even) td { background-color: #1a1f2e !important; }
    .mtable tr:hover td { background-color: #202640 !important; }
    .mtable td.lh { color: #8b93a7 !important; }
    .error-box { background-color: #1f0a0a !important; color: #f87171 !important; }
    .warn-box { background-color: #201a09 !important; color: #fbbf24 !important; }
    .landing { color: #4a5568 !important; }
    .ts-row { border-bottom-color: #1e2538 !important; }
    .ts-label { color: #8b93a7 !important; }
    .ts-value { color: #e2e4ed !important; }
    div[data-testid="stMarkdownContainer"] p { color: #c8cad4; }
    .stTextInput input { background-color: #161b27 !important; color: #e2e4ed !important; }
    .stSelectbox div[data-baseweb="select"] { background-color: #161b27 !important; }
</style>
"""


# ── Data fetching ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: tuple, period: str) -> pd.DataFrame:
    raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_spy(period: str) -> pd.Series:
    raw = yf.download("SPY", period=period, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    c = raw["Close"]
    return (c.squeeze() if isinstance(c, pd.DataFrame) else c).dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_current_prices(tickers: tuple) -> dict:
    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            p = info.get("currentPrice") or info.get("regularMarketPrice")
            out[t] = float(p) if p else None
        except Exception:
            out[t] = None
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_income_history(tickers: tuple) -> dict:
    out = {}
    for tk in tickers:
        try:
            inc = yf.Ticker(tk).income_stmt
            if inc is None or inc.empty:
                out[tk] = {}
                continue
            inc.index = [str(i).lower().strip() for i in inc.index]
            rev_series = ni_series = None
            for idx in inc.index:
                if rev_series is None and ("total revenue" in idx or idx == "revenue"):
                    rev_series = inc.loc[idx]
                if ni_series is None and "net income" in idx:
                    ni_series = inc.loc[idx]
            result = {"revenue_history": [], "net_income_history": [], "income_years": []}
            if rev_series is not None:
                rev_clean = rev_series.dropna().sort_index()
                result["revenue_history"] = list(rev_clean.values.astype(float))
                result["income_years"] = [str(d.year) for d in rev_clean.index]
            if ni_series is not None:
                ni_clean = ni_series.dropna().sort_index()
                result["net_income_history"] = list(ni_clean.values.astype(float))
                if not result["income_years"]:
                    result["income_years"] = [str(d.year) for d in ni_clean.index]
            out[tk] = result
        except Exception:
            out[tk] = {}
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_analyst_data(tickers: tuple) -> dict:
    out = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            out[tk] = {
                "target_mean":   info.get("targetMeanPrice"),
                "target_high":   info.get("targetHighPrice"),
                "target_low":    info.get("targetLowPrice"),
                "rec_key":       (info.get("recommendationKey") or "").lower(),
                "num_analysts":  info.get("numberOfAnalystOpinions"),
            }
        except Exception:
            out[tk] = {}
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sectors(tickers: tuple) -> dict:
    out = {}
    for t in tickers:
        try:
            out[t] = yf.Ticker(t).info.get("sector") or "Unknown"
        except Exception:
            out[t] = "Unknown"
    return out


# ── Optimization ───────────────────────────────────────────────────────────────
def optimize_max_sharpe(mu, S, rf, min_w, max_w) -> dict:
    ef = EfficientFrontier(mu, S, weight_bounds=(min_w, max_w))
    ef.max_sharpe(risk_free_rate=rf)
    return dict(ef.clean_weights())


def optimize_min_vol(mu, S, min_w, max_w) -> dict:
    ef = EfficientFrontier(mu, S, weight_bounds=(min_w, max_w))
    ef.min_volatility()
    return dict(ef.clean_weights())


def optimize_max_sortino(daily_returns: pd.DataFrame, rf: float, min_w: float, max_w: float) -> dict:
    tickers = daily_returns.columns.tolist()
    n = len(tickers)
    daily_rf = rf / 252

    def neg_sortino(w):
        port = daily_returns.values @ w
        ann_ret = port.mean() * 252
        dd = port[port < daily_rf]
        if len(dd) < 5:
            return 10.0
        dstd = np.sqrt((dd ** 2).mean() * 252)
        return -(ann_ret - rf) / (dstd + 1e-9)

    bounds = [(min_w, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    best, best_val = None, np.inf
    rng = np.random.default_rng(42)
    for _ in range(8):
        x0 = rng.dirichlet(np.ones(n))
        x0 = np.clip(x0, min_w, max_w)
        x0 /= x0.sum()
        res = minimize(neg_sortino, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-9})
        if res.success and res.fun < best_val:
            best_val, best = res.fun, res.x
    if best is None:
        best = np.full(n, 1 / n)
    w = {t: max(float(v), 0.0) for t, v in zip(tickers, best)}
    total = sum(w.values())
    return {t: v / total for t, v in w.items()} if total > 0 else w


def optimize_hrp(daily_returns: pd.DataFrame, min_w: float, max_w: float) -> dict:
    hrp = HRPOpt(daily_returns)
    hrp.optimize()
    w = dict(hrp.clean_weights())
    w = {t: float(np.clip(v, min_w, max_w)) for t, v in w.items()}
    total = sum(w.values())
    return {t: v / total for t, v in w.items()} if total > 0 else w


# ── Metrics ────────────────────────────────────────────────────────────────────
def calc_metrics(weights: dict, daily_ret: pd.DataFrame, spy_prices: pd.Series, rf: float) -> dict:
    w = np.array([weights.get(t, 0.0) for t in daily_ret.columns])
    port = daily_ret.values @ w

    cum = np.cumprod(1 + port)
    years = len(port) / 252
    cagr = float(cum[-1] ** (1 / max(years, 0.1)) - 1)
    vol = float(port.std() * np.sqrt(252))
    sharpe = (cagr - rf) / (vol + 1e-9)

    daily_rf = rf / 252
    dd = port[port < daily_rf]
    sortino_denom = float(np.sqrt((dd ** 2).mean() * 252)) if len(dd) >= 5 else vol
    sortino = (cagr - rf) / (sortino_denom + 1e-9)

    cum_s = pd.Series(cum)
    max_dd = float((cum_s / cum_s.cummax() - 1).min())
    calmar = cagr / (abs(max_dd) + 1e-9)

    var_95 = float(np.percentile(port, 5))
    cvar_95 = float(port[port <= var_95].mean()) if (port <= var_95).any() else var_95

    spy_ret = spy_prices.pct_change().dropna()
    port_s = pd.Series(port, index=daily_ret.index)
    common = port_s.index.intersection(spy_ret.index)
    p_al, s_al = port_s.loc[common].values, spy_ret.loc[common].values
    if len(p_al) > 20:
        cov_mat = np.cov(p_al, s_al)
        beta = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-9)
        spy_cagr = float(np.prod(1 + s_al) ** (252 / len(s_al)) - 1)
        alpha = cagr - (rf + beta * (spy_cagr - rf))
    else:
        beta, alpha = 1.0, 0.0

    return dict(
        CAGR=cagr, Annual_Return=cagr, Annual_Volatility=vol,
        Sharpe=sharpe, Sortino=sortino, Calmar=calmar,
        Max_Drawdown=max_dd, VaR=var_95, CVaR=cvar_95,
        Alpha=alpha, Beta=beta,
    )


# ── Monte Carlo ────────────────────────────────────────────────────────────────
def mc_frontier(daily_ret: pd.DataFrame, n: int, rf: float) -> pd.DataFrame:
    mu = (daily_ret.mean() * 252).values
    cov = (daily_ret.cov() * 252).values
    k = len(mu)
    rows = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        w = rng.dirichlet(np.ones(k))
        r = float(mu @ w)
        v = float(np.sqrt(w @ cov @ w))
        rows.append({"ret": r, "vol": v, "sharpe": (r - rf) / (v + 1e-9)})
    return pd.DataFrame(rows)


def mc_future(weights: dict, daily_ret: pd.DataFrame, n_paths: int = 1000) -> pd.DataFrame:
    w = np.array([weights.get(t, 0.0) for t in daily_ret.columns])
    hist = daily_ret.values @ w
    rng = np.random.default_rng(1)
    paths = np.zeros((n_paths, 252))
    for i in range(n_paths):
        samp = rng.choice(hist, size=252, replace=True)
        paths[i] = np.cumprod(1 + samp) - 1
    return pd.DataFrame(paths)


# ── Chart helpers ──────────────────────────────────────────────────────────────
def _layout(h=300, t=None):
    bg   = t["bg"]   if t else BG
    tc   = t["tc"]   if t else NAV
    grid = t["grid"] if t else "#e2e0db"
    return dict(
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(family="Arial,sans-serif", size=11, color=tc),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=grid),
        yaxis=dict(showgrid=True, gridcolor=grid, zeroline=False),
        margin=dict(l=40, r=10, t=30, b=30),
        height=h,
    )


def chart_frontier(mc_df, all_metrics, spy_prices, rf, t=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mc_df["vol"], y=mc_df["ret"], mode="markers",
        marker=dict(size=3, color=mc_df["sharpe"], colorscale="Blues", opacity=0.45,
                    showscale=True, colorbar=dict(title="Sharpe", thickness=10, len=0.5)),
        hovertemplate="Vol:%{x:.1%} Ret:%{y:.1%}<extra>Simulated</extra>", name="Simulated",
    ))
    spy_ret_s = spy_prices.pct_change().dropna()
    spy_r = float(np.prod(1 + spy_ret_s) ** (252 / len(spy_ret_s)) - 1)
    spy_v = float(spy_ret_s.std() * np.sqrt(252))
    fig.add_trace(go.Scatter(
        x=[spy_v], y=[spy_r], mode="markers+text",
        marker=dict(size=13, color=SPY_COLOR, symbol="diamond"),
        text=["SPY"], textposition="top center", textfont=dict(size=9, color=SPY_COLOR),
        name="SPY", hovertemplate=f"SPY Vol:{spy_v:.1%} Ret:{spy_r:.1%}<extra></extra>",
    ))
    syms = {"Max Sharpe": "star", "Min Volatility": "circle", "Max Sortino": "square", "HRP": "triangle-up"}
    for name, m in all_metrics.items():
        c = PORT_COLORS.get(name, MID)
        fig.add_trace(go.Scatter(
            x=[m["Annual_Volatility"]], y=[m["CAGR"]], mode="markers+text",
            marker=dict(size=14, color=c, symbol=syms.get(name, "circle"),
                        line=dict(width=1, color="white")),
            text=[name], textposition="top center", textfont=dict(size=9, color=c),
            name=name,
        ))
    fig.update_layout(**_layout(400, t))
    fig.update_xaxes(tickformat=".0%", title=dict(text="Annual Volatility", font=dict(size=10)))
    fig.update_yaxes(tickformat=".0%", title=dict(text="Annual Return", font=dict(size=10)))
    return fig


def chart_corr(daily_ret, t=None) -> go.Figure:
    corr = daily_ret.corr().round(2)
    tks = corr.columns.tolist()
    mid_bg = t["bg"] if t else BG
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=tks, y=tks,
        colorscale=[[0, "#faf0ef"], [0.5, mid_bg], [1, NAV]], zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y}/%{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**_layout(max(280, 48 * len(tks)), t))
    return fig


def chart_asset_scatter(daily_ret, t=None) -> go.Figure:
    rets = daily_ret.mean() * 252
    vols = daily_ret.std() * np.sqrt(252)
    fig = go.Figure()
    for t in daily_ret.columns:
        fig.add_trace(go.Scatter(
            x=[vols[t]], y=[rets[t]], mode="markers+text",
            marker=dict(size=10, color=SGE if rets[t] > 0 else RED, opacity=0.85),
            text=[t], textposition="top center", textfont=dict(size=9), name=t,
        ))
    fig.update_layout(**_layout(300, t), showlegend=False)
    fig.update_xaxes(tickformat=".0%", title=dict(text="Annual Volatility", font=dict(size=10)))
    fig.update_yaxes(tickformat=".0%", title=dict(text="Annual Return", font=dict(size=10)))
    return fig


def chart_sector_bar(weights_dict, sectors, t=None) -> go.Figure:
    all_sec = sorted({s for s in sectors.values()})
    fig = go.Figure()
    colors = [NAV, SGE, GLD, STL]
    for i, (name, weights) in enumerate(weights_dict.items()):
        sw = {}
        for t, w in weights.items():
            s = sectors.get(t, "Unknown")
            sw[s] = sw.get(s, 0) + w
        fig.add_trace(go.Bar(
            name=name, x=all_sec, y=[sw.get(s, 0) for s in all_sec],
            marker_color=colors[i % 4], opacity=0.85,
        ))
    fig.update_layout(**_layout(280, t), barmode="group")
    fig.update_yaxes(tickformat=".0%")
    return fig


def chart_var(all_metrics, t=None) -> go.Figure:
    names = list(all_metrics.keys())
    colors = [PORT_COLORS.get(n, MID) for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="VaR (95%)", x=names,
                         y=[all_metrics[n]["VaR"] for n in names],
                         marker_color=colors, opacity=0.75))
    fig.add_trace(go.Bar(name="CVaR (95%)", x=names,
                         y=[all_metrics[n]["CVaR"] for n in names],
                         marker_color=colors, opacity=0.40))
    fig.update_layout(**_layout(260, t), barmode="group")
    fig.update_yaxes(tickformat=".2%")
    return fig


def chart_rolling_vol(daily_ret, weights_dict, window=30, t=None) -> go.Figure:
    fig = go.Figure()
    for name, weights in weights_dict.items():
        w = np.array([weights.get(t, 0.0) for t in daily_ret.columns])
        port = pd.Series(daily_ret.values @ w, index=daily_ret.index)
        rv = port.rolling(window).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(x=rv.index, y=rv.values, name=name, mode="lines",
                                 line=dict(color=PORT_COLORS.get(name, MID), width=1.5)))
    fig.update_layout(**_layout(260, t))
    fig.update_yaxes(tickformat=".0%", title=dict(text=f"{window}d Rolling Vol", font=dict(size=10)))
    return fig


def chart_cumulative(daily_ret, weights_dict, spy_prices, t=None) -> go.Figure:
    fig = go.Figure()
    spy_r = spy_prices.pct_change().dropna()
    cum_spy = (1 + spy_r).cumprod() - 1
    fig.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy.values, name="SPY", mode="lines",
                             line=dict(color=SPY_COLOR, width=1.5, dash="dot")))
    for name, weights in weights_dict.items():
        w = np.array([weights.get(t, 0.0) for t in daily_ret.columns])
        port = pd.Series(daily_ret.values @ w, index=daily_ret.index)
        cum = (1 + port).cumprod() - 1
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name=name, mode="lines",
                                 line=dict(color=PORT_COLORS.get(name, MID), width=2)))
    fig.update_layout(**_layout(340, t))
    fig.update_yaxes(tickformat=".0%", title=dict(text="Cumulative Return", font=dict(size=10)))
    return fig


def chart_rolling_sharpe(daily_ret, weights_dict, rf, window=90, t=None) -> go.Figure:
    fig = go.Figure()
    for name, weights in weights_dict.items():
        w = np.array([weights.get(t, 0.0) for t in daily_ret.columns])
        port = pd.Series(daily_ret.values @ w, index=daily_ret.index)
        rs = (port.rolling(window).mean() * 252 - rf) / (port.rolling(window).std() * np.sqrt(252) + 1e-9)
        fig.add_trace(go.Scatter(x=rs.index, y=rs.values, name=name, mode="lines",
                                 line=dict(color=PORT_COLORS.get(name, MID), width=1.5)))
    fig.add_hline(y=0, line=dict(color=LGT, width=1, dash="dash"))
    fig.update_layout(**_layout(260, t))
    fig.update_yaxes(title=dict(text=f"{window}d Rolling Sharpe", font=dict(size=10)))
    return fig


def chart_drawdown(daily_ret, weights_dict, t=None) -> go.Figure:
    fig = go.Figure()
    for name, weights in weights_dict.items():
        w = np.array([weights.get(t, 0.0) for t in daily_ret.columns])
        port = pd.Series(daily_ret.values @ w, index=daily_ret.index)
        cum = (1 + port).cumprod()
        dd = cum / cum.cummax() - 1
        c = PORT_COLORS.get(name, MID)
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name, mode="lines",
                                 fill="tozeroy", line=dict(color=c, width=1),
                                 fillcolor=_hex_rgba(c, 0.15)))
    fig.update_layout(**_layout(260, t))
    fig.update_yaxes(tickformat=".0%", title=dict(text="Drawdown", font=dict(size=10)))
    return fig


def chart_mc_fan(weights, daily_ret, t=None) -> go.Figure:
    paths = mc_future(weights, daily_ret, n_paths=1000)
    days = list(range(252))
    b = {p: paths.quantile(p / 100, axis=0).values for p in [5, 25, 50, 75, 95]}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days + days[::-1], y=list(b[95]) + list(b[5])[::-1],
        fill="toself", fillcolor=_hex_rgba(STL, 0.13), line=dict(color="rgba(0,0,0,0)"), name="5–95th pct",
    ))
    fig.add_trace(go.Scatter(
        x=days + days[::-1], y=list(b[75]) + list(b[25])[::-1],
        fill="toself", fillcolor=_hex_rgba(STL, 0.27), line=dict(color="rgba(0,0,0,0)"), name="25–75th pct",
    ))
    fig.add_trace(go.Scatter(x=days, y=b[50], name="Median",
                             line=dict(color=NAV, width=2)))
    fig.add_hline(y=0, line=dict(color=LGT, width=1, dash="dash"))
    fig.update_layout(**_layout(300, t))
    fig.update_yaxes(tickformat=".0%", title=dict(text="Cumulative Return (1-yr fwd)", font=dict(size=10)))
    fig.update_xaxes(title=dict(text="Trading Days Forward", font=dict(size=10)))
    return fig


def chart_financials(revenue_history, net_income_history, income_years, title="", t=None) -> go.Figure:
    bg   = t["bg"]   if t else BG
    tc   = t["tc"]   if t else NAV
    grid = t["grid"] if t else "#e2e0db"
    rev_color = STL if t else NAV

    all_vals = [abs(v) for v in revenue_history + net_income_history]
    scale, unit = (1e9, "B") if max(all_vals, default=1) >= 1e9 else (1e6, "M")

    fig = go.Figure()
    if revenue_history:
        fig.add_trace(go.Bar(
            x=income_years, y=[v / scale for v in revenue_history],
            name="Revenue", marker_color=rev_color, opacity=0.85,
            hovertemplate=f"$%{{y:.2f}}{unit}<extra>Revenue</extra>",
        ))
    if net_income_history:
        ni_colors = [SGE if v >= 0 else RED for v in net_income_history]
        fig.add_trace(go.Bar(
            x=income_years, y=[v / scale for v in net_income_history],
            name="Net Income", marker_color=ni_colors, opacity=0.82,
            hovertemplate=f"$%{{y:.2f}}{unit}<extra>Net Income</extra>",
        ))
    fig.update_layout(**_layout(210, t), barmode="group",
                      title=dict(text=title, font=dict(size=11, color=tc), x=0.5) if title else {})
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=".1f", tickprefix="$", ticksuffix=unit,
                     zeroline=True, zerolinecolor=grid, zerolinewidth=1)
    return fig


def chart_price_history(prices: pd.DataFrame, t=None) -> go.Figure:
    palette = [NAV, SGE, GLD, STL, RED, MID, "#6b4fbb", "#c87941", "#2d8a8a", "#8a4a6b"]
    fig = go.Figure()
    for i, col in enumerate(prices.columns):
        norm = (prices[col] / prices[col].iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, name=col, mode="lines",
            line=dict(color=palette[i % len(palette)], width=1.5),
            hovertemplate=f"{col}: %{{y:+.1f}}%<extra>%{{x|%b %d, %Y}}</extra>",
        ))
    grid = t["grid"] if t else "#e2e0db"
    fig.add_hline(y=0, line=dict(color=grid, width=1, dash="dash"))
    fig.update_layout(**_layout(300, t))
    fig.update_yaxes(ticksuffix="%", tickformat="+.0f",
                     title=dict(text="Return from Start", font=dict(size=10)))
    return fig


def pie_chart(weights, title, color, t=None) -> go.Figure:
    labels = [tk for tk, w in weights.items() if w > 0.001]
    values = [weights[tk] for tk in labels]
    bg = t["bg"] if t else BG
    tc = t["tc"] if t else NAV
    palette = [color, LGT, CARD, STL, SGE, GLD, MID, NAV, RED, BG]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.38,
        marker=dict(colors=palette[:len(labels)], line=dict(color=bg, width=2)),
        textfont=dict(size=10),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(family="Arial,sans-serif", size=10, color=tc),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=36, b=10), height=240,
        title=dict(text=title, font=dict(size=10, color=color), x=0.5),
    )
    return fig


# ── HTML table helpers ─────────────────────────────────────────────────────────
def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _pct(v):    return f"{v:.1%}" if v is not None else "—"
def _num(v, d=2): return f"{v:.{d}f}" if v is not None else "—"

def _colored(v, fmt):
    if v is None: return "—"
    c = SGE if v > 0 else (RED if v < 0 else MID)
    return f'<span style="color:{c};font-weight:700;">{fmt(v)}</span>'


def metrics_table(all_metrics) -> str:
    strats = list(all_metrics.keys())
    ROWS = [
        ("Annual Return",     "Annual_Return",     lambda v: _colored(v, lambda x: f"{x:+.1%}")),
        ("Annual Volatility", "Annual_Volatility", lambda v: _pct(v)),
        ("Sharpe Ratio",      "Sharpe",            lambda v: _colored(v, lambda x: f"{x:+.2f}")),
        ("Sortino Ratio",     "Sortino",           lambda v: _colored(v, lambda x: f"{x:+.2f}")),
        ("Calmar Ratio",      "Calmar",            lambda v: _colored(v, lambda x: f"{x:+.2f}")),
        ("Max Drawdown",      "Max_Drawdown",      lambda v: _colored(v, lambda x: f"{x:+.1%}")),
        ("VaR (95%)",         "VaR",               lambda v: _pct(v)),
        ("CVaR (95%)",        "CVaR",              lambda v: _pct(v)),
        ("Alpha vs SPY",      "Alpha",             lambda v: _colored(v, lambda x: f"{x:+.1%}")),
        ("Beta vs SPY",       "Beta",              lambda v: _num(v)),
        ("CAGR",              "CAGR",              lambda v: _colored(v, lambda x: f"{x:+.1%}")),
    ]
    h = '<div class="mtable-wrap"><table class="mtable"><tr><th class="lh">Metric</th>'
    for s in strats:
        c = PORT_COLORS.get(s, NAV)
        h += f'<th style="border-top:3px solid {c};">{_html.escape(s)}</th>'
    h += "</tr>"
    for label, key, fmt in ROWS:
        h += f'<tr><td class="lh">{label}</td>'
        for s in strats:
            h += f"<td>{fmt(all_metrics[s].get(key))}</td>"
        h += "</tr>"
    return h + "</table></div>"


def weights_table(weights_dict) -> str:
    strats = list(weights_dict.keys())
    tickers = sorted({t for w in weights_dict.values() for t, v in w.items() if v > 0.001})
    h = '<div class="mtable-wrap"><table class="mtable"><tr><th class="lh">Ticker</th>'
    for s in strats:
        c = PORT_COLORS.get(s, NAV)
        h += f'<th style="border-top:3px solid {c};">{_html.escape(s)}</th>'
    h += "</tr>"
    for t in tickers:
        h += f'<tr><td class="lh">{_html.escape(t)}</td>'
        for s in strats:
            v = weights_dict[s].get(t, 0.0)
            h += f"<td>{'—' if v < 0.001 else f'{v:.1%}'}</td>"
        h += "</tr>"
    return h + "</table></div>"


def rebalance_table(weights_dict, cur_prices, portfolio_value) -> str:
    strats = list(weights_dict.keys())
    tickers = sorted({t for w in weights_dict.values() for t, v in w.items() if v > 0.001})
    h = '<div class="mtable-wrap"><table class="mtable"><tr><th class="lh">Ticker</th>'
    for s in strats:
        c = PORT_COLORS.get(s, NAV)
        h += f'<th colspan="2" style="border-top:3px solid {c};">{_html.escape(s)}</th>'
    h += '</tr><tr><th class="lh"></th>'
    for _ in strats:
        h += "<th>$ Amt</th><th>Shares</th>"
    h += "</tr>"
    for t in tickers:
        price = cur_prices.get(t)
        h += f'<tr><td class="lh">{_html.escape(t)}</td>'
        for s in strats:
            v = weights_dict[s].get(t, 0.0)
            dollar = portfolio_value * v
            shares = (dollar / price) if price and price > 0 else None
            h += f"<td>${dollar:,.0f}</td><td>{'—' if shares is None else f'{shares:.1f}'}</td>"
        h += "</tr>"
    return h + "</table></div>"


def analyst_table(tickers, analyst_data, cur_prices) -> str:
    REC_COLOR = {
        "strong_buy": SGE, "buy": SGE, "hold": GLD,
        "underperform": RED, "sell": RED,
    }
    REC_LABEL = {
        "strong_buy": "Strong Buy", "buy": "Buy", "hold": "Hold",
        "underperform": "Underperform", "sell": "Sell",
    }
    h = ('<div class="mtable-wrap"><table class="mtable">'
         '<tr><th class="lh">Ticker</th><th>Price</th>'
         '<th>Low Target</th><th>Mean Target</th><th>High Target</th>'
         '<th>Upside</th><th>Rating</th><th>Analysts</th></tr>')
    for tk in tickers:
        d = analyst_data.get(tk, {})
        price = cur_prices.get(tk)
        mean  = d.get("target_mean")
        low   = d.get("target_low")
        high  = d.get("target_high")
        rec   = d.get("rec_key", "")
        n     = d.get("num_analysts")

        upside_html = "—"
        if mean and price:
            pct = (mean - price) / price * 100
            col = SGE if pct >= 0 else RED
            upside_html = f'<span style="color:{col};font-weight:700;">{pct:+.1f}%</span>'

        rec_col   = REC_COLOR.get(rec, MID)
        rec_label = REC_LABEL.get(rec, rec.replace("_", " ").title() if rec else "—")
        rec_html  = f'<span style="color:{rec_col};font-weight:700;">{_html.escape(rec_label)}</span>'

        h += (f'<tr><td class="lh">{_html.escape(tk)}</td>'
              f'<td>{f"${price:,.2f}" if price else "—"}</td>'
              f'<td>{f"${low:,.2f}" if low else "—"}</td>'
              f'<td>{f"${mean:,.2f}" if mean else "—"}</td>'
              f'<td>{f"${high:,.2f}" if high else "—"}</td>'
              f'<td>{upside_html}</td>'
              f'<td>{rec_html}</td>'
              f'<td>{n if n else "—"}</td></tr>')
    return h + "</table></div>"


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    dark_mode = st.session_state.get("dark_mode", False)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    if dark_mode:
        st.markdown(DARK_CSS, unsafe_allow_html=True)
    T = {"bg": "#0d1117", "tc": "#e2e4ed", "grid": "#1e2538"} if dark_mode else None

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.toggle("Dark Mode", key="dark_mode")
        st.markdown(
            '<p style="font-size:16px;font-weight:700;color:#1a2744;margin-bottom:2px;">Portfolio Optimizer</p>'
            '<p style="font-size:10px;color:#8b93a7;margin:0 0 16px;">MPT · HRP · Monte Carlo</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Tickers (up to 10)</div>', unsafe_allow_html=True)
        defaults = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "", "", "", "", ""]
        col_a, col_b = st.columns(2)
        raw = []
        for i in range(10):
            with (col_a if i % 2 == 0 else col_b):
                raw.append(
                    st.text_input(f"#{i+1}", value=defaults[i],
                                  label_visibility="collapsed", key=f"tk{i}").strip().upper()
                )
        tickers = [t for t in raw if t]

        st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
        period_label = st.selectbox("Date Range", ["1 Year", "3 Years", "5 Years"], index=1)
        period = {"1 Year": "1y", "3 Years": "3y", "5 Years": "5y"}[period_label]
        rf_pct = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 4.5, 0.1)
        rf = rf_pct / 100
        n_sims = st.slider("Monte Carlo Simulations", 1000, 10000, 5000, step=1000)

        st.markdown('<div class="section-label">Weight Constraints</div>', unsafe_allow_html=True)
        min_w = st.slider("Min Weight per Asset (%)", 0, 20, 0, 1) / 100
        max_w = st.slider("Max Weight per Asset (%)", 10, 100, 40, 5) / 100
        port_val = st.number_input("Portfolio Value ($)", 100, 10_000_000, 10_000, 1000)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Optimize", use_container_width=True)
        st.markdown(
            '<p style="font-size:9px;color:#b0b8c6;line-height:1.6;margin-top:18px;">'
            "Data from Yahoo Finance · Educational use only · Not financial advice.</p>",
            unsafe_allow_html=True,
        )

    if not run_btn:
        st.markdown('<div class="landing">Enter tickers in the sidebar and click Optimize</div>',
                    unsafe_allow_html=True)
        return

    # ── Validation ─────────────────────────────────────────────────────────────
    if len(tickers) < 2:
        st.markdown('<div class="error-box"><strong>Error:</strong> Enter at least 2 tickers.</div>',
                    unsafe_allow_html=True)
        return
    if min_w * len(tickers) > 1.0:
        st.markdown(
            f'<div class="warn-box">Min weight × tickers exceeds 100% — resetting min weight to 0%.</div>',
            unsafe_allow_html=True)
        min_w = 0.0

    # ── Fetch ──────────────────────────────────────────────────────────────────
    with st.spinner("Fetching price data…"):
        prices_raw = fetch_prices(tuple(tickers), period)
        spy_prices = fetch_spy(period)

    if prices_raw.empty or spy_prices.empty:
        st.markdown('<div class="error-box"><strong>Data Error:</strong> Could not fetch price data. Verify tickers.</div>',
                    unsafe_allow_html=True)
        return

    valid = [c for c in prices_raw.columns if prices_raw[c].isna().mean() < 0.2]
    dropped = [t for t in tickers if t not in valid]
    if dropped:
        st.markdown(f'<div class="warn-box">Dropped (insufficient data): {", ".join(dropped)}</div>',
                    unsafe_allow_html=True)

    prices = prices_raw[valid].dropna()
    if len(prices.columns) < 2:
        st.markdown('<div class="error-box"><strong>Error:</strong> Need at least 2 valid tickers.</div>',
                    unsafe_allow_html=True)
        return
    if len(prices) < 60:
        st.markdown('<div class="warn-box">Less than 60 trading days of data — results may be unreliable.</div>',
                    unsafe_allow_html=True)

    active = tuple(prices.columns.tolist())
    daily_ret = prices.pct_change().dropna()

    # ── Optimize ───────────────────────────────────────────────────────────────
    with st.spinner("Running optimization…"):
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        weights_dict, opt_errors = {}, []
        for name, fn in [
            ("Max Sharpe",     lambda: optimize_max_sharpe(mu, S, rf, min_w, max_w)),
            ("Min Volatility", lambda: optimize_min_vol(mu, S, min_w, max_w)),
            ("Max Sortino",    lambda: optimize_max_sortino(daily_ret, rf, min_w, max_w)),
            ("HRP",            lambda: optimize_hrp(daily_ret, min_w, max_w)),
        ]:
            try:
                weights_dict[name] = fn()
            except Exception as e:
                opt_errors.append(f"{name}: {e}")
        for err in opt_errors:
            st.markdown(f'<div class="warn-box">Optimization warning — {_html.escape(str(err))}</div>',
                        unsafe_allow_html=True)
    if not weights_dict:
        st.markdown('<div class="error-box"><strong>Error:</strong> All optimizations failed.</div>',
                    unsafe_allow_html=True)
        return

    # ── Compute everything ─────────────────────────────────────────────────────
    with st.spinner("Computing metrics & running Monte Carlo…"):
        all_metrics = {n: calc_metrics(w, daily_ret, spy_prices, rf)
                       for n, w in weights_dict.items()}
        mc_df = mc_frontier(daily_ret, n_sims, rf)

    with st.spinner("Fetching supplementary data…"):
        cur_prices     = fetch_current_prices(active)
        sectors        = fetch_sectors(active)
        analyst_data   = fetch_analyst_data(active)
        income_history = fetch_income_history(active)

    # ══ TABS ══════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Allocations", "Risk Analysis", "Performance", "Tear Sheet"]
    )

    # ── Tab 1: Overview ────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-label">Portfolio Highlights</div>', unsafe_allow_html=True)
        highlights = [
            ("Max Sharpe",     "Highest Sharpe",    lambda m: f'{m["Sharpe"]:.2f}',    "Sharpe Ratio"),
            ("Min Volatility", "Lowest Volatility", lambda m: f'{m["Annual_Volatility"]:.1%}', "Annual Vol"),
            ("Max Sortino",    "Highest Sortino",   lambda m: f'{m["Sortino"]:.2f}',   "Sortino Ratio"),
            ("HRP",            "HRP",               lambda m: f'{m["CAGR"]:.1%}',      "CAGR"),
        ]
        cols = st.columns(4)
        for col, (strat, label, val_fn, sub) in zip(cols, highlights):
            if strat not in all_metrics:
                continue
            c = PORT_COLORS[strat]
            with col:
                st.markdown(
                    f'<div class="metric-card" style="border-top:3px solid {c};">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value" style="color:{c};">{val_fn(all_metrics[strat])}</div>'
                    f'<div class="metric-sub">{sub}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-label">Full Metrics Comparison</div>', unsafe_allow_html=True)
        st.markdown(metrics_table(all_metrics), unsafe_allow_html=True)

        st.markdown('<div class="section-label">Efficient Frontier</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_frontier(mc_df, all_metrics, spy_prices, rf, t=T),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-label">Individual Asset Price History</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_price_history(prices, t=T),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-label">Revenue &amp; Net Income by Ticker</div>', unsafe_allow_html=True)
        fin_cols = st.columns(2)
        for i, tk in enumerate(active):
            d = income_history.get(tk, {})
            if d.get("revenue_history") or d.get("net_income_history"):
                with fin_cols[i % 2]:
                    st.plotly_chart(
                        chart_financials(
                            d.get("revenue_history", []),
                            d.get("net_income_history", []),
                            d.get("income_years", []),
                            title=tk, t=T,
                        ),
                        use_container_width=True, config={"displayModeBar": False},
                    )

    # ── Tab 2: Allocations ─────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-label">Portfolio Weights</div>', unsafe_allow_html=True)
        st.markdown(weights_table(weights_dict), unsafe_allow_html=True)

        st.markdown('<div class="section-label">Allocation Breakdown</div>', unsafe_allow_html=True)
        pie_cols = st.columns(len(weights_dict))
        for col, (name, w) in zip(pie_cols, weights_dict.items()):
            with col:
                st.plotly_chart(pie_chart(w, name, PORT_COLORS.get(name, NAV), t=T),
                                use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-label">Analyst Price Targets</div>', unsafe_allow_html=True)
        st.markdown(analyst_table(list(active), analyst_data, cur_prices), unsafe_allow_html=True)

        st.markdown('<div class="section-label">Rebalancing Guide</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:10px;color:{MID};margin-top:-4px;">'
            f'Based on ${port_val:,.0f} · Share counts are approximate</p>',
            unsafe_allow_html=True,
        )
        st.markdown(rebalance_table(weights_dict, cur_prices, port_val), unsafe_allow_html=True)

    # ── Tab 3: Risk Analysis ────────────────────────────────────────────────────
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">Correlation Heatmap</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_corr(daily_ret, t=T), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            st.markdown('<div class="section-label">Asset Risk / Return</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_asset_scatter(daily_ret, t=T), use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown('<div class="section-label">Sector Allocation by Portfolio</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_sector_bar(weights_dict, sectors, t=T),
                        use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="section-label">VaR &amp; CVaR Comparison</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_var(all_metrics, t=T), use_container_width=True,
                            config={"displayModeBar": False})
        with c4:
            st.markdown('<div class="section-label">Rolling 30-Day Volatility</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rolling_vol(daily_ret, weights_dict, t=T),
                            use_container_width=True, config={"displayModeBar": False})

    # ── Tab 4: Performance ─────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-label">Cumulative Returns vs SPY</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_cumulative(daily_ret, weights_dict, spy_prices, t=T),
                        use_container_width=True, config={"displayModeBar": False})

        c5, c6 = st.columns(2)
        with c5:
            st.markdown('<div class="section-label">Rolling 90-Day Sharpe</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_rolling_sharpe(daily_ret, weights_dict, rf, t=T),
                            use_container_width=True, config={"displayModeBar": False})
        with c6:
            st.markdown('<div class="section-label">Drawdown</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_drawdown(daily_ret, weights_dict, t=T),
                            use_container_width=True, config={"displayModeBar": False})

        if "Max Sharpe" in weights_dict:
            st.markdown(
                '<div class="section-label">Monte Carlo Fan — Max Sharpe (1-Year Forward, 1,000 Paths)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="font-size:9px;color:{MID};margin-top:-4px;">'
                "Bootstrap from historical daily returns · Not a forecast.</p>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(chart_mc_fan(weights_dict["Max Sharpe"], daily_ret, t=T),
                            use_container_width=True, config={"displayModeBar": False})

    # ── Tab 5: Tear Sheet ──────────────────────────────────────────────────────
    with tab5:
        if "Max Sharpe" not in all_metrics:
            st.markdown('<div class="warn-box">Max Sharpe portfolio unavailable.</div>',
                        unsafe_allow_html=True)
        else:
            m = all_metrics["Max Sharpe"]
            w = weights_dict["Max Sharpe"]
            tc1, tc2 = st.columns([1, 2])

            with tc1:
                st.markdown(
                    f'<p style="font-size:14px;font-weight:700;color:{NAV};margin-bottom:2px;">'
                    f'Max Sharpe Portfolio</p>'
                    f'<p style="font-size:9px;color:{MID};text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin-top:0;">Fact Sheet · {period_label}</p>',
                    unsafe_allow_html=True,
                )
                stats = [
                    ("Annual Return",     _pct(m["Annual_Return"])),
                    ("Annual Volatility", _pct(m["Annual_Volatility"])),
                    ("Sharpe Ratio",      _num(m["Sharpe"])),
                    ("Sortino Ratio",     _num(m["Sortino"])),
                    ("Calmar Ratio",      _num(m["Calmar"])),
                    ("Max Drawdown",      _pct(m["Max_Drawdown"])),
                    ("VaR (95%)",         _pct(m["VaR"])),
                    ("CVaR (95%)",        _pct(m["CVaR"])),
                    ("Alpha vs SPY",      _pct(m["Alpha"])),
                    ("Beta vs SPY",       _num(m["Beta"])),
                    ("CAGR",              _pct(m["CAGR"])),
                ]
                st.markdown(
                    "".join(f'<div class="ts-row"><span class="ts-label">{lbl}</span>'
                            f'<span class="ts-value">{val}</span></div>' for lbl, val in stats),
                    unsafe_allow_html=True,
                )
                quality = "strong" if m["Sharpe"] > 1 else ("moderate" if m["Sharpe"] > 0.5 else "weak")
                alpha_word = "outperformed" if m["Alpha"] > 0 else "underperformed"
                st.markdown(
                    f'<p style="font-size:11px;color:{MID};line-height:1.75;margin-top:16px;">'
                    f'The Max Sharpe portfolio delivered a CAGR of <strong style="color:{NAV};">'
                    f'{_pct(m["CAGR"])}</strong> at annualized volatility of <strong style="color:{NAV};">'
                    f'{_pct(m["Annual_Volatility"])}</strong>, a <strong style="color:{NAV};">{quality}'
                    f'</strong> risk-adjusted profile (Sharpe {_num(m["Sharpe"])}). '
                    f'Worst drawdown was <strong style="color:{RED};">{_pct(m["Max_Drawdown"])}</strong>. '
                    f'It {alpha_word} SPY by <strong style="color:{NAV};">{_pct(abs(m["Alpha"]))}</strong> '
                    f'with a beta of <strong style="color:{NAV};">{_num(m["Beta"])}</strong>.</p>',
                    unsafe_allow_html=True,
                )

            with tc2:
                st.plotly_chart(pie_chart(w, "Allocation", NAV, t=T),
                                use_container_width=True, config={"displayModeBar": False})
                pw = np.array([w.get(tk, 0.0) for tk in daily_ret.columns])
                cum_p = (1 + pd.Series(daily_ret.values @ pw, index=daily_ret.index)).cumprod() - 1
                cum_spy = (1 + spy_prices.pct_change().dropna()).cumprod() - 1
                mini = go.Figure()
                mini.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy.values, name="SPY",
                                          line=dict(color=SPY_COLOR, width=1.5, dash="dot")))
                mini.add_trace(go.Scatter(x=cum_p.index, y=cum_p.values, name="Max Sharpe",
                                          line=dict(color=NAV, width=2)))
                mini.update_layout(**_layout(190, T))
                mini.update_yaxes(tickformat=".0%")
                st.plotly_chart(mini, use_container_width=True, config={"displayModeBar": False})

                rs = chart_rolling_sharpe(daily_ret, {"Max Sharpe": w}, rf, t=T)
                rs.update_layout(height=165, margin=dict(l=40, r=10, t=10, b=30))
                st.plotly_chart(rs, use_container_width=True, config={"displayModeBar": False})


if __name__ == "__main__":
    main()
