# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:54:40 2026

@author: Elkri
"""

import numpy as np

#################################################Functions stock generation####################################################
def allocate_stock(actual_stock, sold, theft, rest):

    """
    Allocates sold, theft and rest 
    when the stock is running low. Allocation depends 
    on the size of the precomputed random sold theft and rest.
    Largest precomputed loss first -> second -> third
    """
    
    #Put in  dict
    flows = {
        "sold": sold,
        "theft": theft,
        "rest": rest
    }

    #Sort large → small
    sorted_flows = sorted(flows.items(), key=lambda x: x[1], reverse=True)

    remaining_stock = actual_stock
    corrected = {}

    for name, value in sorted_flows:
        if remaining_stock <= 0:
            corrected[name] = 0
        else:
            allocated = min(value, remaining_stock)
            corrected[name] = allocated
            remaining_stock -= allocated

    return corrected["sold"], corrected["theft"], corrected["rest"]



def theft_probabilities_from_price_size(price):

    """
    Determines the chance of stealing 0,1 or 2 items
    depending on the price and size of the product
    (Higher price and or smaller size -> more chance of stealing 1,2 products)
    """

    
    # Normalize to [0,1]
    price_norm = (price - 5) / (20 - 5)
    
    # Risk score (tune weights if needed)
    risk = 1 * price_norm 
    # Convert risk to probabilities
    p2 = 0.005 + 0.025 * risk      # theft = 2 increases with risk
    p1 = 0.015 + 0.035 * risk      # theft = 1 increases with risk
    p0 = 1 - (p1 + p2)           # remaining probability

    # Safety (numerical stability)
    p0 = max(p0, 0.05)

    # Normalize (just in case)
    total = p0 + p1 + p2
    return [p0 / total, p1 / total, p2 / total]





def double_scan_prob(sold, rest, max_volume=10):
    """
    Linear probability scaling:
    - volume = sold + rest
    - p1 and p2 scale linearly with volume
    - volume = 0 => p1 = p2 = 0
    """

    volume = min(sold + rest, max_volume)

    # per-item probabilities (tune these!)
    p1_per_item = 0.025   # probability of 1 double scan per item handled
    p2_per_item = 0.005  # probability of 2 double scans per item handled

    p1 = volume * p1_per_item
    p2 = volume * p2_per_item

    # ensure probabilities stay valid
    p1 = min(p1, 0.9)
    p2 = min(p2, 0.9 - p1)

    p0 = 1 - p1 - p2

    return [p0, p1, p2]




def Colli_vault_prod(Vault_coll, max_delivery):
    """
    Calculate the probability of missing products within collies (normal planned delivery only)
    """
    
    volume = abs(max_delivery - Vault_coll)

    # prevent division by zero
    if max_delivery == 0:
        return np.array([1.0])  # only 0 missing

    scale = volume / max_delivery

    values = np.arange(0, max_delivery + 1)

    # decreasing probability with k (missing products)
    probs = np.exp(-scale * values)

    # normalize
    probs = probs / probs.sum()

    return probs




def subtract_one(data):
    """
    Function for updating experation dates of individual items
    """
    if data is None:
        return None
    elif isinstance(data, list):
        return [subtract_one(item) for item in data]
    else:
        return data - 1



def update_pools(t, pools, Initial_pool, sold, theft, rest, rng):

    loss_sum = sold + theft + rest

    if t == 1:
        pools[0] = Initial_pool if loss_sum == 0 else Initial_pool[:-loss_sum]
        return pools, []

    prob_history = []

    k = 0.1
    alpha = 0.3

    for _ in range(loss_sum):

        weights = []

        for pool in pools:

            if pool[0] > 0:
                exp_weight = np.tanh(k * pool[0])
            else:
                exp_weight = np.tanh(k * 0.1)

            size_weight = len(pool) ** alpha
            weights.append(exp_weight * size_weight)

        weights = np.asarray(weights)

        probs = weights / weights.sum()
        prob_history.append(probs.copy())

        # IMPORTANT: use expiration-specific RNG
        chosen = rng.choice(len(pools), p=probs)

        pools[chosen] = pools[chosen][:-1]

        if len(pools[chosen]) == 0:
            del pools[chosen]

    return pools, prob_history



def has_negative(data):
    """
    Function to flag if a pool of items with the same 
    expiring date has a negative value in it.
    """
    if data is None:
        return False
    elif isinstance(data, list):
        return any(has_negative(item) for item in data)
    else:
        return data < 0



def check_negative_pools(pools, counted_negative_pools):
    """
    Keep track of items that have expired and remove expired/empty pools.
    """
    expired_list = 0
    pools_to_keep = []

    for ii, pool in enumerate(pools):

        if pool is None or len(pool) == 0:
            # Remove None and empty pools
            continue

        if has_negative(pool):
            expired_list += len(pool)
            counted_negative_pools.add(ii)

            # Don't add it -> removes the pool

        else:
            pools_to_keep.append(pool)

    pools[:] = pools_to_keep

    return expired_list, counted_negative_pools

