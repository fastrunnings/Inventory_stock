# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:54:40 2026

@author: Elkri
"""

import Functions_stock_generation as fsg # Import stock generation functions
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def Initialize_prod(T, Shelf_length):

    #Generate price 
    price = np.random.randint(5, 21)  

    #Generate day of the week and Season    
    start_date = datetime(2026, 1, 1)
    day_of_week = np.empty(T, dtype=object)
    season = np.empty(T, dtype=object)
    for t in range(T):
        current_date = start_date + timedelta(days=t)
    
        # Add days
        day_of_week[t] = current_date.strftime("%A")
        
        #Determine Month
        month = current_date.month
    
        #Select season
        if month in [3, 4, 5]:
            season[t] = "spring"
        elif month in [6, 7, 8]:
            season[t] = "summer"
        elif month in [9, 10, 11]:
            season[t] = "autumn"
        else:
            season[t] = "winter" 
    
    #Generate theft
    p_theft = fsg.theft_probabilities_from_price_size(price)     
    theft = np.random.choice(
        [0, 1, 2],
        size=T,
        p=p_theft
    )
    theft[0] = 0 #initialize first timestep
    
    
    ######################################Important if you get sold and rest data!#################################################    
    #Get this for sold and rest from 1 years data (find all possible values in the data for each product)
    #Get the count of each of the max value, then determine the probabilities of each value (count value/sum count all values)
    ###################################################################################################################################
    
    # Generate random sold values and decreasing probabilities when values get higher    
    max_value_ver = np.random.randint(2, 9) #Get this from 1 years data (find all possible values in the data for each product)
    values_ver = np.arange(0, max_value_ver + 1) #Get the count of each of the max value, then determine the probabilities of each value (count value/sum count all values)
    
    decay_ver = np.exp(-0.6 * values_ver)
    noise_ver = np.random.rand(len(values_ver))
    
    probs_ver = decay_ver * noise_ver
    probs_ver /= probs_ver.sum()
    
    # Generate sold
    sold = np.random.choice(values_ver, size=T, p=probs_ver)
    sold[0] = 0 #initialize first timestep
    
    # Generate random rest values and decreasing probabilities when values get higher  
    max_value_rest = np.random.randint(0, 3) 
    values_rest = np.arange(0, max_value_rest + 1) 
    
    decay_rest = np.exp(-3 * values_rest)
    noise_rest = np.random.rand(len(values_rest))
    
    probs_rest = decay_rest * noise_rest
    probs_rest /= probs_rest.sum()
    
    # Generate rest 
    rest = np.random.choice(values_rest, size=T, p=probs_rest)
    rest[0] = 0 #initialize first timestep
    
    ###################################################################################################################################
    
    
    
    
    #Set maximum number of rows possible for each product and minimum experation date
    avg_sold_week = int(np.round(np.sum(sold)/(T/7)))
    if avg_sold_week <= 3:
        max_prod_rows = 1 # 1 row 
        min_exp_date = 28 # 1 month
    elif avg_sold_week > 3 and avg_sold_week <= 8:
        max_prod_rows = 2 # 2 rows
        min_exp_date = 10 # 10 days
    elif avg_sold_week > 8 and avg_sold_week <= 12:
        max_prod_rows = 3 # 3 rows
        min_exp_date = 7 # 5 days
    else:
        max_prod_rows = 4 # 4 rows
        min_exp_date = 7 # 7 days min

    #Generate experation date
    max_exp_date = int(min_exp_date * 5) # times 5 to try and engulf as many items as possible
    Exp_date = np.random.randint(min_exp_date, max_exp_date)
    
    prod_lenght = np.random.uniform(0.05, 0.15)
    prod_width = np.random.uniform(0.05, 0.15)
    prod_height = np.random.uniform(0.05, 0.15)
    size = prod_lenght*prod_width*prod_height # size in m^3
    
    
    
    max_row = int(np.round(Shelf_length/prod_lenght))
    
    
    
    if avg_sold_week >= (max_row * max_prod_rows) and avg_sold_week > 3:
        max_prod_rows += 1
    elif (max_row * max_prod_rows) >= (2 * avg_sold_week) and avg_sold_week > 3:
        max_prod_rows -= 1
    
    prod_max = max_prod_rows * max_row
    prod_min = 2 * max_prod_rows
    
    
    
    #Generate Colli (Size in terms of individual items for 1 package delivery to the shop)
    Colli = np.random.choice([6, 8, 10])
    
    
    
    #Determine how much collies can be delivered for each product
    if Colli * 2 >= prod_max:
        max_delivery = 1 # 1 collies can be delivered maximum!
    elif Colli * 3 >= prod_max:
        max_delivery = 2 
    elif Colli * 3 >= prod_max:
        max_delivery = 3 
    else:
        max_delivery = 4 
    
    
    #Set variable for tracking a new order
    # new_order == False
    
    #Generate delivery time in days
    Delivery_time = np.random.choice(
        [2, 3, 4],
        size=T,
        p=[0.30, 0.40, 0.30]
    )

    #Determine average delivery time for stock prediction
    avg_delivery_time = int(np.round(np.sum(Delivery_time)/ len(Delivery_time)))
    Delivery_time[0] = 0 #initialize first timestep

    
      
    days_after_order = 0
    Delivery_sceduled = 1
    new_order = True
    
    #Initialize delivery amount (keep track of how many products will be delivered)
    delivery_amount = np.zeros(T, dtype=int)
    delivery_amount[0] = prod_max
    
    
    #Generate an array for double scanned (verkoop of afgescanned) items and extra unintened not registered Colli deliveries 
    double_scanned = np.zeros(T, dtype=int)
    extra_colli = np.zeros(T, dtype=int)
    Extra_colli_stock = np.zeros(T, dtype=int)
    
    
    
    #Initialize tracking variables  
    coll_stat = np.zeros(T, dtype=int)
    stock_missing = np.zeros(T, dtype=int)
    zerotell = False
    zero_t = np.zeros(T, dtype=int)
    
    #Generate an initial stock at t =1(stock is stock that is registered!)
    stock = [prod_max] 
    
    #Actual stock
    actual_stock = np.zeros(T, dtype=int)
    actual_stock[0] = prod_max
    
    #stock correctie
    stock_correctie = False #initialize
    stock_cor = np.zeros(T, dtype=int)
    
    #dayrestanten correctie 
    day_rest_cor = np.zeros(T, dtype=int)

    #Initialze array for stock prediction
    pred_t = [[] for _ in range(T)]
    
    pred_q10 = [[] for _ in range(T)]
    pred_q50 = [[] for _ in range(T)]
    pred_q90 = [[] for _ in range(T)]

    #Initialze array for saving stock prediction
    stock_q10 = np.zeros(T, dtype=int)
    stock_q50 = np.zeros(T, dtype=int)
    stock_q90 = np.zeros(T, dtype=int)


    #initialize Variables for ordering
    days_till_order = None
    Frist_order_day = True
    time_to_order = np.nan


    #Initialize parameters for stock predicton
    alpha = None
    alpha_base = None
    states = None

    
    return {
        "price": price,
        "theft": theft,
        "sold": sold,
        "day_of_week":day_of_week,
        "season":season,
        "rest": rest,
        "max_prod_rows": max_prod_rows,
        "size": size,
        "prod_max": prod_max,
        "prod_min": prod_min,
        "Colli": Colli,
        "max_delivery": max_delivery,
        "Delivery_time": Delivery_time,
        "days_after_order": days_after_order,
        "Delivery_scheduled": Delivery_sceduled,
        "new_order": new_order,
        "delivery_amount": delivery_amount,
        "double_scanned": double_scanned,
        "extra_colli": extra_colli,
        "Extra_colli_stock": Extra_colli_stock,
        "coll_stat": coll_stat,
        "stock_missing": stock_missing,
        "zerotell": zerotell,
        "zero_t": zero_t,
        "stock": stock,
        "actual_stock": actual_stock,
        "stock_correctie": stock_correctie,
        "stock_cor": stock_cor,
        "day_rest_cor": day_rest_cor,
        "pred_t":pred_t,
        "pred_q10":pred_q10, 
        "pred_q50":pred_q50, 
        "pred_q90":pred_q90,    
        "avg_delivery_time":avg_delivery_time,
        "days_till_order": days_till_order,
        "Frist_order_day": Frist_order_day,
        "time_to_order": time_to_order,
        "alpha":alpha,
        "alpha_base":alpha_base,
        "states":states,
        "stock_q10":stock_q10,
        "stock_q50":stock_q50,
        "stock_q90":stock_q90,
        "Exp_date":Exp_date,

        
    }
