"""
Monte Carlo Option Pricer
=========================
Prices a European call option by simulating thousands of possible future
share prices, then checks the answer against the exact Black-Scholes formula.

Also compares plain Monte Carlo with "antithetic variates", a simple trick
that makes the estimate more accurate for the same number of simulations.

Run:  python monte_carlo_options.py
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Inputs (change these and re-run to see how the price moves)
# ---------------------------------------------------------------------------
S0 = 100.0     # share price today (£)
K = 105.0      # strike price: the price you have the right to buy at (£)
T = 1.0        # time to expiry in years
r = 0.04       # risk-free interest rate (4% a year)
sigma = 0.20   # volatility: how much the share price swings (20% a year)
SEED = 42      # fixes the random numbers so results are reproducible


# ---------------------------------------------------------------------------
# 2. Exact answer: the Black-Scholes formula
# ---------------------------------------------------------------------------
def black_scholes_call(S0, K, T, r, sigma):
    """Closed-form price of a European call option."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# 3. Monte Carlo estimate
# ---------------------------------------------------------------------------
def monte_carlo_call(S0, K, T, r, sigma, n_sims, antithetic=False, rng=None):
    """
    Simulate end-of-period share prices using geometric Brownian motion:
        S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)
    The option pays max(S_T - K, 0). Average the payoffs, then discount
    back to today. Returns (price, standard error).
    """
    rng = rng or np.random.default_rng(SEED)

    if antithetic:
        # Draw half the random numbers, then use their mirror image (-Z).
        # Pairs of opposite shocks cancel out some of the noise.
        Z = rng.standard_normal(n_sims // 2)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(n_sims)

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.exp(-r * T) * np.maximum(ST - K, 0.0)

    if antithetic:
        # Average each pair first; the pairs are the independent samples
        half = n_sims // 2
        pair_means = 0.5 * (payoffs[:half] + payoffs[half:])
        return pair_means.mean(), pair_means.std(ddof=1) / np.sqrt(half)

    return payoffs.mean(), payoffs.std(ddof=1) / np.sqrt(n_sims)


# ---------------------------------------------------------------------------
# 4. Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exact = black_scholes_call(S0, K, T, r, sigma)
    print(f"Black-Scholes price:            £{exact:.4f}\n")

    n = 100_000
    plain, se_plain = monte_carlo_call(S0, K, T, r, sigma, n)
    anti, se_anti = monte_carlo_call(S0, K, T, r, sigma, n, antithetic=True)

    print(f"Monte Carlo ({n:,} sims):     £{plain:.4f}  (std error {se_plain:.4f})")
    print(f"With antithetic variates:       £{anti:.4f}  (std error {se_anti:.4f})")
    print(f"Error reduction from antithetic: {100 * (1 - se_anti / se_plain):.1f}%\n")

    # --- Chart 1: does the estimate converge to the true price? ---
    sizes = np.unique(np.logspace(2, 5.5, 30).astype(int) // 2 * 2)
    plain_est, anti_est = [], []
    for m in sizes:
        plain_est.append(monte_carlo_call(S0, K, T, r, sigma, m)[0])
        anti_est.append(monte_carlo_call(S0, K, T, r, sigma, m, antithetic=True)[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(sizes, plain_est, "o-", label="Plain Monte Carlo", alpha=0.8)
    ax.plot(sizes, anti_est, "s-", label="Antithetic variates", alpha=0.8)
    ax.axhline(exact, color="black", ls="--", label=f"Black-Scholes (£{exact:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulations")
    ax.set_ylabel("Option price (£)")
    ax.set_title("Monte Carlo estimate converges to Black-Scholes")
    ax.legend()

    # --- Chart 2: a sample of simulated share price paths ---
    rng = np.random.default_rng(SEED)
    steps, n_paths = 252, 50                     # 252 trading days in a year
    dt = T / steps
    shocks = rng.standard_normal((n_paths, steps))
    log_paths = np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))

    ax = axes[1]
    ax.plot(np.linspace(0, T, steps + 1), paths.T, lw=0.8, alpha=0.6)
    ax.axhline(K, color="black", ls="--", label=f"Strike (£{K:.0f})")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Share price (£)")
    ax.set_title(f"{n_paths} simulated share price paths")
    ax.legend()

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("Chart saved to results.png")
