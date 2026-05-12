# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:00:55 2026

@author: Elkri
"""

# -*- coding: utf-8 -*-

import numpy as np
from pmf_stock_pred import Functions_pmf_pred as pmf


def stock_prediction(
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
        rho,
        sigma,
        base_prior_pmf,
        states,
        history,
        draws,
        multi_draws,
        base_history,
        pred_q10,
        pred_q50,
        pred_q90,
        manual_check=False,
        reset_on_manual_check=True,
        n_sims=1000,
        horizon=200,
        order_quantile="q10"
        ):

    time_to_order = np.nan

    if t == Idx_pred:

        epsilon = np.asarray(stock, dtype=int) - np.roll(np.asarray(stock, dtype=int), -1)
        epsilon, base_pmf_epsilon = pmf.update_epsilon(epsilon, stock, sold, Idx_pred)

        states = pmf.make_states_from_data(epsilon, tail_buffer=5)

        base_prior_pmf = pmf.pmf_from_samples_with_tail_prior(
            epsilon,
            states=states,
            prior_strength=2.0,
            lam=1.2
        )

    elif t > Idx_pred:

        epsilon = np.asarray(stock, dtype=int) - np.roll(np.asarray(stock, dtype=int), -1)
        epsilon, base_pmf_epsilon = pmf.update_epsilon(epsilon, stock, sold, Idx_pred)

        recent_data = epsilon[-window:]

        pmf_t = pmf.predictive_pmf_from_base_and_recent(
            base_prior_pmf=base_prior_pmf,
            recent_data=recent_data,
            states=states,
            decay=decay
        )

        # ----------------------------------------------------
        # Manual stock check: reset uncertainty like Dirichlet
        # ----------------------------------------------------
        if reset_on_manual_check and manual_check:
            pmf_t = base_prior_pmf.copy()

        history.append(pmf_t)
        base_history.append(base_prior_pmf.copy())

        if delivery_amount[t] > 0:
            start_val = stock[t]

            future_draws = rng.choice(
                states,
                size=(n_sims, horizon),
                p=pmf_t
            )

            stock_paths = start_val - np.concatenate(
                [
                    np.zeros((n_sims, 1)),
                    np.cumsum(future_draws, axis=1)
                ],
                axis=1
            )

            q10 = np.percentile(stock_paths, 10, axis=0)
            q50 = np.percentile(stock_paths, 50, axis=0)
            q90 = np.percentile(stock_paths, 90, axis=0)

            pred_q10[t].append(q10)
            pred_q50[t].append(q50)
            pred_q90[t].append(q90)

            pred_t[t].append(q50)

            draws = future_draws[0]
            multi_draws.append(stock_paths)

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

        else:
            multi_draws.append(np.array([]))

        lag_index = t - lag

        if 0 <= lag_index < len(base_pmf_epsilon):
            if not np.isnan(base_pmf_epsilon[lag_index]):
                base_prior_pmf = pmf.update_base_prior_lagged(
                    base_prior_pmf=base_prior_pmf,
                    obs=base_pmf_epsilon[lag_index],
                    states=states,
                    rho=rho,
                    sigma=sigma
                )

    return (
        base_prior_pmf,
        states,
        history,
        draws,
        multi_draws,
        base_history,
        pred_q10,
        pred_q50,
        pred_q90,
        time_to_order
    )