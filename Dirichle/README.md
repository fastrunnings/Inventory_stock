# Bayesian Dirichlet Inventory Prediction

## Overview

This project implements a Bayesian inventory forecasting framework based on an evolving Dirichlet distribution. The method estimates future stock depletion using historical inventory observations and uncertainty-aware Monte Carlo simulation.

The approach consists of:

1. Estimating a prior demand distribution from historical stock observations.
2. Updating the demand distribution over time using Bayesian evidence accumulation.
3. Simulating many future demand scenarios.
4. Constructing uncertainty bands using the 10th, 50th and 90th percentiles.
5. Determining future ordering dates from the predicted stock trajectories.

---

## 1. Demand Estimation

The observed stock depletion process is approximated as

[
\varepsilon_t = S_t - S_{t+1}
]

where

* (S_t) = stock level at time (t)
* (\varepsilon_t) = observed stock reduction (demand proxy)

The possible demand states are defined as

[
\mathcal{K} = {0,1,\dots,K}
]

with

[
K = \max(\varepsilon_{1:\text{IdxPred}}) + \text{tail buffer}
]

---

## 2. Initial Dirichlet Prior

A prior distribution is constructed from historical observations.

For each demand state (i),

[
\alpha_i^{(0)}
==============

c_i
+
\eta
\frac{e^{-\lambda i}}
{\sum_{j \in \mathcal{K}} e^{-\lambda j}}
]

where

* (c_i) = observed count of demand state (i)
* (\eta) = prior strength
* (\lambda) = exponential decay parameter

The vector

[
\boldsymbol{\alpha}^{(0)}
=========================

(\alpha_0,\alpha_1,\dots,\alpha_K)
]

defines the initial Dirichlet distribution.

---

## 3. Bayesian Updating

Recent observations update the evidence vector according to

[
\alpha_i^{(t+1)}
================

\alpha_i^{(t)}
+
\omega_t \gamma
\mathbf{1}(\varepsilon_t=i)
]

where

* (\omega_t) = recency weight
* (\gamma) = observation strength
* (\mathbf{1}(\cdot)) = indicator function

The recency weights are

[
\omega_t = d^{,t}
]

where (d) is the user-defined decay factor.

This gives greater importance to recent observations while retaining historical information.

---

## 4. Long-Term Trend Updating

A slower baseline distribution is maintained:

[
\alpha_i^{\text{base}}(t+1)
===========================

\delta
\alpha_i^{\text{base}}(t)
+
\gamma
\mathbf{1}
(\varepsilon_{t-\ell}=i)
]

where

* (\delta) = long-term forgetting factor
* (\ell) = lag length

This allows the model to adapt to structural demand changes.

---

## 5. Manual Stock Checks

Whenever a manual inventory correction occurs (e.g. stock count, inventory correction, daily remainder correction), uncertainty is reset:

[
\boldsymbol{\alpha}^{(t)}
=========================

\boldsymbol{\alpha}^{\text{base}}(t)
]

This reflects the fact that the true stock level has become known again.

---

## 6. Demand Distribution Sampling

At prediction time, a demand probability vector is sampled from the Dirichlet distribution:

[
\mathbf{p}^{(m)}
\sim
\text{Dirichlet}
\left(
\boldsymbol{\alpha}^{(t)}
\right)
]

where

[
\mathbf{p}^{(m)}
================

(p_0,p_1,\dots,p_K)
]

represents one possible future demand distribution.

---

## 7. Monte Carlo Demand Simulation

Future demand is sampled from the categorical distribution:

[
d_\tau^{(m)}
\sim
\text{Categorical}
\left(
\mathbf{p}^{(m)}
\right)
]

for

[
\tau = 1,\dots,H
]

where (H) is the prediction horizon.

---

## 8. Stock Forecasting

Future stock levels are simulated as

[
S_{t+\tau}^{(m)}
================

## S_t

\sum_{j=1}^{\tau}
d_j^{(m)}
]

for each Monte Carlo simulation (m).

This produces a collection of possible future stock trajectories.

---

## 9. Uncertainty Quantification

For each future timestep, prediction intervals are obtained from the simulated trajectories:

[
q_{10,\tau}
===========

P_{10}
\left(
S_{t+\tau}^{(m)}
\right)
]

[
q_{50,\tau}
===========

P_{50}
\left(
S_{t+\tau}^{(m)}
\right)
]

[
q_{90,\tau}
===========

P_{90}
\left(
S_{t+\tau}^{(m)}
\right)
]

where

* (q_{10}) = pessimistic forecast
* (q_{50}) = median forecast
* (q_{90}) = optimistic forecast

These form the uncertainty bands shown in the forecast plots.

---

## 10. Ordering Decision

The future ordering date is determined from the first time the selected quantile crosses the stock threshold:

[
\tau^*
======

\min
\left{
\tau :
q_{\text{order},\tau}
\le 0
\right}
]

The ordering time is then

[
t_{\text{order}}
================

t
+
\tau^*
------

## L

1
]

where

* (L) = average delivery lead time

Using

[
q_{10}
]

results in a conservative ordering strategy, while

[
q_{50}
]

and

[
q_{90}
]

lead to progressively more optimistic decisions.

---

## Key Features

* Bayesian learning of demand distributions
* Explicit uncertainty quantification
* Automatic adaptation to changing demand patterns
* Reset of uncertainty after manual stock verification
* Monte Carlo inventory forecasting
* Quantile-based inventory ordering strategy
* Suitable for retail and warehouse inventory management
