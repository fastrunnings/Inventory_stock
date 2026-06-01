# Bayesian Dirichlet Inventory Prediction

## Overview

This project implements a Bayesian inventory forecasting framework based on an evolving Dirichlet distribution.

The method:

1. Learns a prior demand distribution from historical stock data.
2. Updates this distribution using new observations.
3. Simulates future demand using Monte Carlo sampling.
4. Computes uncertainty intervals using the 10th, 50th and 90th percentiles.
5. Determines future ordering dates based on predicted stock depletion.

---

## 1. Demand Estimation

Observed stock depletion is approximated as:

```text
εₜ = Sₜ − Sₜ₊₁
```

where:

* Sₜ = stock level at time t
* εₜ = estimated demand (stock depletion)

Possible demand states are:

```text
K = {0, 1, 2, ..., Kmax}
```

where:

```text
Kmax = max(ε₁:IdxPred) + tail_buffer
```

---

## 2. Initial Dirichlet Prior

The initial evidence vector is:

```text
αᵢ⁽⁰⁾ = cᵢ + η · exp(−λi) / Σ exp(−λj)
```

where:

* αᵢ = evidence for state i
* cᵢ = observed count of state i
* η = prior strength
* λ = exponential decay parameter

The complete Dirichlet parameter vector is:

```text
α = (α₀, α₁, α₂, ..., αₖ)
```

---

## 3. Bayesian Updating

As new observations arrive:

```text
αᵢ⁽ᵗ⁺¹⁾ = αᵢ⁽ᵗ⁾ + ωₜ · γ · I(εₜ = i)
```

where:

* ωₜ = recency weight
* γ = observation strength
* I(·) = indicator function

The recency weights are:

```text
ωₜ = decayᵗ
```

This gives more weight to recent observations.

---

## 4. Long-Term Trend Updating

A slower baseline distribution is maintained:

```text
α_base,ᵢ⁽ᵗ⁺¹⁾ = δ · α_base,ᵢ⁽ᵗ⁾ + γ · I(εₜ₋ₗₐg = i)
```

where:

* δ = long-term forgetting factor
* lag = update lag

This captures long-term demand changes.

---

## 5. Manual Stock Checks

Whenever:

* zerotelling
* stockcorrecties
* dayremnants

occurs, uncertainty is reset:

```text
α⁽ᵗ⁾ = α_base⁽ᵗ⁾
```

The stock level is then assumed to be known exactly.

---

## 6. Demand Distribution Sampling

Future demand probabilities are sampled from:

```text
p⁽ᵐ⁾ ~ Dirichlet(α)
```

where:

```text
p⁽ᵐ⁾ = (p₀, p₁, ..., pₖ)
```

Each sample represents one possible future demand distribution.

---

## 7. Monte Carlo Demand Simulation

Future demand values are generated as:

```text
dτ⁽ᵐ⁾ ~ Categorical(p⁽ᵐ⁾)
```

for:

```text
τ = 1, 2, ..., H
```

where H is the prediction horizon.

---

## 8. Stock Forecasting

Future stock levels are computed as:

```text
Sₜ₊τ⁽ᵐ⁾ = Sₜ − Σ dⱼ⁽ᵐ⁾
```

where:

```text
j = 1 ... τ
```

This produces many possible future stock trajectories.

---

## 9. Uncertainty Quantification

For every future timestep:

```text
q₁₀ = 10th percentile
q₅₀ = median prediction
q₉₀ = 90th percentile
```

computed from all simulated trajectories.

These form the prediction interval:

```text
[q₁₀ , q₉₀]
```

and the central forecast:

```text
q₅₀
```

---

## 10. Ordering Decision

The ordering time is determined by the first future timestep where the chosen prediction path reaches zero stock:

```text
τ* = min{τ : q_order(τ) ≤ 0}
```

The order should then be placed at:

```text
t_order = t + τ* − L − 1
```

where:

* L = average delivery lead time

Possible choices are:

```text
q₁₀ → conservative strategy
q₅₀ → median strategy
q₉₀ → optimistic strategy
```

---

## Key Features

* Bayesian demand learning
* Dirichlet uncertainty modelling
* Monte Carlo stock simulation
* Dynamic adaptation to changing demand
* Uncertainty reset after manual stock checks
* Quantile-based inventory forecasting
* Automatic ordering recommendations

---

## Workflow

```text
Historical Stock Data
          │
          ▼
 Estimate εₜ = Sₜ − Sₜ₊₁
          │
          ▼
 Build Initial Dirichlet Prior
          │
          ▼
 Bayesian Updating
          │
          ▼
 Sample Dirichlet Distributions
          │
          ▼
 Monte Carlo Demand Simulation
          │
          ▼
 Future Stock Paths
          │
          ▼
 q10 / q50 / q90 Forecasts
          │
          ▼
 Ordering Recommendation
```
