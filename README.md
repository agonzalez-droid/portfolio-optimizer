# Portfolio Optimizer

A professional-grade portfolio optimization tool built with Streamlit, using live data from Yahoo Finance.

---

<!-- Screenshot placeholder -->
<!-- ![App screenshot](screenshot.png) -->

## How to run locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Methodology

**Modern Portfolio Theory (MPT)** — The Max Sharpe and Min Volatility portfolios are solved via quadratic programming on the mean-variance frontier (PyPortfolioOpt). Expected returns and the covariance matrix are estimated from historical daily returns.

**Max Sortino** — Directly maximizes the Sortino ratio (return / downside deviation) using multi-start SLSQP optimization via SciPy. Only downside semi-deviation is penalized, making it more suitable for asymmetric return distributions.

**Hierarchical Risk Parity (HRP)** — Clusters assets by correlation using hierarchical linkage, then allocates risk inversely proportional to cluster volatility. Produces diversified portfolios without requiring expected return estimates, which reduces estimation error.

**Monte Carlo (efficient frontier)** — Randomly samples thousands of weight combinations to map the feasible portfolio space. Each point is colored by its Sharpe ratio.

**Monte Carlo (forward simulation)** — Bootstraps 1,000 one-year forward paths by sampling with replacement from historical daily returns. Shows 5th/25th/50th/75th/95th percentile bands.

**Backtesting note** — All performance metrics are computed in-sample using the same historical data used for optimization. This overstates expected out-of-sample performance and is for educational illustration only.

## Disclaimer

This tool is for **educational purposes only**. It is not financial advice and should not be used as the sole basis for any investment decision. All results are highly sensitive to the historical period selected and input assumptions.
