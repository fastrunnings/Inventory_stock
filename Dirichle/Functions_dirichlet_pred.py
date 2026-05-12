# -*- coding: utf-8 -*-

import numpy as np


# ============================================================
# Helper functions for Dirichlet stock prediction
# ============================================================

def round_and_clip_nonnegative(x):
    x = np.asarray(x)
    return np.clip(np.rint(x).astype(int), 0, None)


def make_states_from_data(samples, tail_buffer=5):
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]

    if len(samples) == 0:
        max_val = 1
    else:
        max_val = int(np.max(round_and_clip_nonnegative(samples)))

    return np.arange(max_val + tail_buffer + 1)


def decaying_tail_prior(states, lam=1.2):
    prior = np.exp(-lam * states)
    return prior / prior.sum()


def initialize_alpha_from_samples(
    samples,
    states,
    prior_strength=2.0,
    lam=1.2,
    min_alpha=1e-6
):
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    samples = round_and_clip_nonnegative(samples)

    counts = np.bincount(samples, minlength=len(states))[:len(states)].astype(float)

    prior = decaying_tail_prior(states, lam=lam)
    alpha = counts + prior_strength * prior

    return np.maximum(alpha, min_alpha)


def update_alpha_dirichlet(
    alpha,
    obs,
    states,
    decay=0.995,
    obs_strength=1.0,
    min_alpha=1e-6
):
    """
    Bayesian evolving Dirichlet update:

        alpha[t+1] = decay * alpha[t] + evidence[t]

    Here evidence is one observed epsilon / demand value.
    """

    alpha = decay * alpha

    if not np.isnan(obs):
        obs_int = int(round_and_clip_nonnegative([obs])[0])

        if obs_int < len(states):
            alpha[obs_int] += obs_strength
        else:
            # If observation is outside known states, add it to last bin
            alpha[-1] += obs_strength

    return np.maximum(alpha, min_alpha)


def compute_epsilon(stock, sold, Idx_pred):
    """
    Same idea as your PMF version:
        s[t+1] = s[t] + delivery[t] + epsilon[t]

    Here epsilon is approximated from stock change.

    Negative jumps and unrealistically large jumps are filtered.
    You can adapt this exactly to your pmf.update_epsilon logic if needed.
    """

    stock_arr = np.asarray(stock, dtype=float)

    epsilon = stock_arr - np.roll(stock_arr, -1)

    # Ignore final artificial roll value
    epsilon[-1] = np.nan

    # Basic cleaning:
    # negative means stock increased, likely delivery/correction
    epsilon = np.where(epsilon < 0, np.nan, epsilon)

    # Optional: remove very large values
    epsilon = np.where(epsilon > 8, np.nan, epsilon)

    # Historical/base epsilon
    base_epsilon = epsilon[:Idx_pred]

    return epsilon, base_epsilon


def simulate_stock_paths_dirichlet(
    start_stock,
    alpha,
    states,
    horizon=200,
    n_sims=1000,
    rng=None
):
    """
    Simulate future stock paths using:
        p ~ Dirichlet(alpha)
        demand_j ~ Categorical(p)

    Returns:
        paths with shape (n_sims, horizon + 1)
    """

    if rng is None:
        rng = np.random.default_rng()

    paths = np.zeros((n_sims, horizon + 1))
    paths[:, 0] = start_stock

    for s in range(n_sims):
        p = rng.dirichlet(alpha)
        future_draws = rng.choice(states, size=horizon, p=p)
        paths[s, 1:] = start_stock - np.cumsum(future_draws)

    return paths


# ============================================================
# Main Dirichlet prediction function
# ============================================================

def stock_prediction_dirichlet(
        t,
        Idx_pred,
        stock,
        sold,
        delivery_amount,
        rng,
        pred_t,
        avg_delivery_time,
        window,
        lag,
        decay,
        alpha_decay,
        obs_strength,
        alpha,
        states,
        alpha_base,
        history,
        draws,
        multi_draws,
        alpha_history,
        pred_q10,
        pred_q50,
        pred_q90,
        manual_check=False,
        reset_on_manual_check=True,
        horizon=200,
        n_sims=1000,
        order_quantile="q10",
        prior_strength=2.0,
        lam=1.2
):
    """
    Dirichlet version of stock_prediction.

    Parameters
    ----------
    t : int
        Current timestep.

    Idx_pred : int
        Timestep where prediction starts / prior is initialized.

    stock : array-like
        Stock time series.

    sold : array-like
        Sold time series. Included for compatibility and possible epsilon cleaning.

    delivery_amount : array-like
        Delivery amount time series.

    rng : np.random.Generator
        Random generator.

    pred_t : list of lists
        Stores median stock prediction path after delivery.

    avg_delivery_time : int
        Average delivery lead time.

    window : int
        Recent evidence window length.

    lag : int
        Lag for slow evidence update.

    decay : float
        Recency decay for recent observations.

    alpha_decay : float
        Dirichlet memory decay.
        Example: 0.995 means slow forgetting.

    obs_strength : float
        Strength of one observed evidence update.

    alpha : np.ndarray or None
        Current Dirichlet alpha vector.

    states : np.ndarray or None
        Possible epsilon/demand states.

    alpha_base : np.ndarray or None
        Baseline alpha prior.

    history : list
        Stores posterior mean PMF alpha / sum(alpha).

    draws : list
        Stores one-step sampled draws if desired.

    multi_draws : list
        Stores simulated future demand draws or paths.

    alpha_history : list
        Stores alpha over time.

    pred_q10, pred_q50, pred_q90 : list of lists
        Stores 10%, 50%, and 90% prediction trajectories.

    manual_check : bool
        True if Nulltelling / Voorraad correctie / Dag rest correctie happened.

    reset_on_manual_check : bool
        If True, reset alpha to alpha_base when manual stock check happens.

    horizon : int
        Number of future timesteps to simulate.

    n_sims : int
        Number of Monte Carlo simulations.

    order_quantile : str
        Which quantile to use for order decision:
        "q10" = pessimistic
        "q50" = median
        "q90" = optimistic

    Returns
    -------
    alpha, states, alpha_base, history, draws, multi_draws,
    alpha_history, pred_q10, pred_q50, pred_q90, time_to_order
    """

    time_to_order = np.nan

    # ========================================================
    # 1. Initialize Dirichlet prior at prediction start
    # ========================================================

    if t == Idx_pred:

        epsilon, base_epsilon = compute_epsilon(stock, sold, Idx_pred)

        states = make_states_from_data(base_epsilon, tail_buffer=5)

        alpha_base = initialize_alpha_from_samples(
            base_epsilon,
            states=states,
            prior_strength=prior_strength,
            lam=lam
        )

        alpha = alpha_base.copy()

        history.append(alpha / alpha.sum())
        alpha_history.append(alpha.copy())

    # ========================================================
    # 2. Prediction phase
    # ========================================================

    elif t > Idx_pred:

        epsilon, base_epsilon = compute_epsilon(stock, sold, Idx_pred)

        # ----------------------------------------------------
        # Manual stock check: reset uncertainty
        # ----------------------------------------------------
        if reset_on_manual_check and manual_check:
            alpha = alpha_base.copy()

        # ----------------------------------------------------
        # Recent evidence update
        # ----------------------------------------------------
        recent_data = epsilon[max(0, len(epsilon) - window):]

        # Apply recent observations with recency weights
        recent_data = np.asarray(recent_data, dtype=float)
        recent_data = recent_data[~np.isnan(recent_data)]

        if len(recent_data) > 0:
            recency_weights = decay ** np.arange(len(recent_data) - 1, -1, -1)
            recency_weights = recency_weights[::-1]

            for obs, w in zip(recent_data, recency_weights):
                alpha = update_alpha_dirichlet(
                    alpha=alpha,
                    obs=obs,
                    states=states,
                    decay=1.0,
                    obs_strength=obs_strength * w
                )

        # Save current predictive mean PMF
        pmf_mean_t = alpha / alpha.sum()
        history.append(pmf_mean_t)
        alpha_history.append(alpha.copy())

        # Optional one-step draw
        draw = rng.choice(states, p=pmf_mean_t)
        draws.append(draw)

        # ----------------------------------------------------
        # If delivery arrives, simulate future stock paths
        # ----------------------------------------------------
        if manual_check:

            start_val = stock[t]

            paths = simulate_stock_paths_dirichlet(
                start_stock=start_val,
                alpha=alpha,
                states=states,
                horizon=horizon,
                n_sims=n_sims,
                rng=rng
            )

            q10 = np.percentile(paths, 10, axis=0)
            q50 = np.percentile(paths, 50, axis=0)
            q90 = np.percentile(paths, 90, axis=0)

            pred_q10[t].append(q10)
            pred_q50[t].append(q50)
            pred_q90[t].append(q90)

            # For compatibility with your old plotting:
            pred_t[t].append(q50)

            multi_draws.append(paths)

            # ------------------------------------------------
            # Determine time_to_order from selected quantile
            # ------------------------------------------------
            if order_quantile == "q10":
                order_path = q10
            elif order_quantile == "q50":
                order_path = q50
            elif order_quantile == "q90":
                order_path = q90
            else:
                raise ValueError("order_quantile must be 'q10', 'q50', or 'q90'.")

            for j_pred, pred_stock in enumerate(order_path):
                if pred_stock <= 0:
                    time_to_order = j_pred - avg_delivery_time - 1
                    break

        # ----------------------------------------------------
        # No delivery: keep placeholder
        # ----------------------------------------------------
        else:
            multi_draws.append(np.array([]))

        # ----------------------------------------------------
        # Slow lagged baseline update
        # ----------------------------------------------------
        lag_index = t - lag

        if 0 <= lag_index < len(base_epsilon):
            obs_lagged = base_epsilon[lag_index]

            if not np.isnan(obs_lagged):
                alpha_base = update_alpha_dirichlet(
                    alpha=alpha_base,
                    obs=obs_lagged,
                    states=states,
                    decay=alpha_decay,
                    obs_strength=obs_strength
                )

    return (
        alpha,
        states,
        alpha_base,
        history,
        draws,
        multi_draws,
        alpha_history,
        pred_q10,
        pred_q50,
        pred_q90,
        time_to_order
    )