# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:00:55 2026

@author: Elkri
"""


import numpy as np


def round_and_clip_nonnegative(x):
    """
    Round floats to integers if needed
    """
    x = np.asarray(x)
    return np.clip(np.rint(x).astype(int), 0, None)


def make_states_from_data(stock_prior, tail_buffer=5):
    """
    create possible states for prior.
    Essentialy just the maximum value in the prior data + tail_buffer
    """
    stock_prior = np.asarray(stock_prior, dtype=float)
    stock_prior = stock_prior[~np.isnan(stock_prior)]
    max_val = int(np.max(stock_prior)) #First max not persee needed but could be usefull later
    K = max_val + tail_buffer
    
    return np.arange(K + 1)


def pmf_from_samples_with_tail_prior(stock_prior, states, prior_strength=2.0, lam=1.2):
    """
    This function create a Probability Mass Function from the
    prior stock and the epsilon states, note this function consist of 2 part,
    1-> counts of the states in the stock_prior (most important part)
    2-> extra weighing of the counts based on own logical choices. 
        Note: that atm this second extra weighting is not every strong
        but can be used maybe later
    )
    """
    #some data cleaning
    stock_prior = np.asarray(stock_prior, dtype=float)
    stock_prior = stock_prior[~np.isnan(stock_prior)]
    stock_prior = round_and_clip_nonnegative(stock_prior)

    #Calc counts and weighting (pseudocounts)
    counts = np.bincount(stock_prior, minlength=len(states))[:len(states)].astype(float)
    tail_prior = decaying_tail_prior(states, lam=lam)
    pseudocounts = prior_strength * tail_prior

    pmf = counts + pseudocounts
    return normalize_pmf(pmf)



def decaying_tail_prior(states, lam=1.2):
    """
    Create a weights for each state 
    not this is not tuned but the principle should be okey
    """
    prior = np.exp(-lam * states)
    return normalize_pmf(prior)


    
def normalize_pmf(p):
    """
    Normalize the weights from from decaying_tail_prior function 
    """
    
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        raise ValueError("PMF sums to zero.")
    return p / s

    

def predictive_pmf_from_base_and_recent(
    base_prior_pmf,
    recent_data,
    states,
    decay=0.9):
    #recent_strength=8.0, # 8.0 makes actual base prior more important then weighting!
    #prior_strength=2.0, #keep the same called on in pmf_from_samples_with_tail_prior
    #lam=1.2):
   
    """
    Predictive PMF = baseline prior + + recent window contribution.
    Note that recent window contribution can be quite 
    """
    ##################lines below might be double (remove them?) #################################
    #weighted_counts = prior_strength * decaying_tail_prior(states, lam=lam) #Identical to pmf_from_samples_with_tail_prior
    #weighted_counts += recent_strength * base_prior_pmf #(recent_strength * base_prior_pmf)

    weighted_counts = base_prior_pmf.copy()
    #############################################################################################

    #Make sure recent data is in right format
    recent_data = np.asarray(recent_data, dtype=float)
    recent_data = recent_data[~np.isnan(recent_data)]
    recent_data = round_and_clip_nonnegative(recent_data)
    
    #Updating the weigting note k^l and window size can be quite influential! tune?!
    if len(recent_data) > 0:
        recency_weights = decay ** np.arange(len(recent_data) - 1, -1, -1) #Give each previous recent datapoint a weight
        recency_weights = recency_weights[::-1] #switch array around
        
        for obs, w in zip(recent_data, recency_weights):
            if obs < len(states): #This statement should always be true (otherwise increase tail_buffer in make_states_from_data)
                weighted_counts[obs] += w #Add weighting to certain obsvervation Do this for all obsverations

    return normalize_pmf(weighted_counts)



def update_epsilon(epsilon, stock, sold , Idx_pred):
    """
    update epsilon for each timestep (because you get new stock value per timestep)
    """
    # # #Mask deliverys  
    mask_neg = np.where((epsilon)[1:-1] < 0)[0] #have to add a zero to get out of stupid tuple
    
    #Mask maximum are due to corrections (zerotelling, stock correctie, day rest correctie)
    #and not actual changes in stock
    #Thus take what the maximum amount was that you sold previously and mask the rest 
    #(later on you make it also possible to predict up to tail_buffer values)
    max_sold =np.max(sold[:Idx_pred]) #Since sold is initiazed for all times before the loop you have to index! 
    mask_pos = np.where((epsilon)[1:-1] > max_sold)[0] 
    mask = np.concatenate([mask_pos, mask_neg])

    # #Eventual data used to make first prediction
    prev_epsilon = np.delete((epsilon[1:-1]),mask)

    #Saves epsilons with nans for base_pmf updating 
    base_pmf_epsilon = epsilon[1:-1].astype(float).copy()
    base_pmf_epsilon[mask] = np.nan
    
    return prev_epsilon, base_pmf_epsilon



def observation_kernel_pmf(obs, states, sigma=0.8):
    """
    Smooth pmf centered at one observation.
    """
    obs = float(obs)
    p = np.exp(-0.5 * ((states - obs) / sigma) ** 2)
    return normalize_pmf(p)



def update_base_prior_lagged(base_prior_pmf, obs, states, rho=0.03, sigma=0.8):
    """
    Slowly update base_pmf with one old observation at the time
    """
    obs_pmf = observation_kernel_pmf(obs, states, sigma=sigma)
    new_base = (1 - rho) * base_prior_pmf + rho * obs_pmf
    return normalize_pmf(new_base)
