import Functions_stock_generation as fsg # Import stock generation functions
import Initialize_product as ip # Call Variables from initialization


from pmf_stock_pred import pmf_stock_predictions_up as pmf_up_script# Call functions from updated pmf prediction script
from Dirichlet_stock_pred import Functions_dirichlet_pred as Dir_script# Call functions from dirichlet prediction script



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


#############################################Create data set#############################################

#Set seed for random generations
np.random.seed(19)
#Set seed for stock prediction
rng = np.random.default_rng(42)
            
#Initialize dataframe and time periods
all_products = []
columns = ["Product", "time", "theft", "Price", "Size",
           "stock", "sold", "Rest", "Double scanned", "Extra colli",
           "Colli", "zerotelling"]

empty_row = pd.DataFrame([{col: "" for col in columns}]) # create empty row for readability
T = 365*2  # total time periods 2 years!
Shelf_length = 1 # 1 meters
nr_products = 10 #At 100 (but take 1000 to be sure!) you should be fine with your boxplots! no need to do more. 
Idx_pred = int(np.round(T/2)) #Set Idx for stock prediction 


#initialize variables for short and long timescales pmf updating 
window = 28 #4 weeks, trend (note this can matter a lot)
lag = 50 #time scale for rolling window base_pmf
decay = 0.9 #decay used for updating the pmf
rho = 0.03 #parameter for updating the base_pmf
sigma = 0.8 #parameter for updating the base_pmf


base_prior_pmf = 0 #for updating the base_pmf
states = 0 #for updating the states in base_pmf
history = [] #keep track of pmf history
draws = [] #keep track of single draws
multi_draws = [] #keep track of all draws
base_history = [] #keep track of base_pmf history 
alpha_history = [] ##keep track of Dirichlet history
delivered_stock_sum = 0 #Variable to keep track of deliveries for predictions 


#Calculations for size (needed for theft later on??)
# max_lenght, max_width, max_height = 0.25, 0.25, 0.25
# min_lenght, min_width, min_height = 0.1, 0.1, 0.1
# min_size = min_lenght * min_width * min_height 
# max_size = max_lenght * max_width * max_height 


#Flags

Flag_col_del_mis = False # Missing collies
Flag_col_del_prod_mis = False #Missing products within collies (normal delivery only)

Flag_col_del_extra = False # Not registerd extra colli delivery 
Flag_col_extra_del_prod_mis = False # Missing products within extra collies
Flag_double_scanned = True # Products accidently scanned double or triple

Flagzerotelling = True # Perform a zerotelling
Flagstockcorrecties = True # Perform stock correctie
Flagdayremnants = True # Perform checks in case the colli from the delivery does not fit in the shelf 
Flagorderprediction_pmf = False # Predict when to order new products based on probability mass function 
Flagorderprediction_dir = True # Predict when to order new products based on dirichlet mass function




#################################################Start generation of data###############################################


for product_id in range(0, nr_products): #Loop over alll products
    prod_init = ip.Initialize_prod(T, Shelf_length) 
    price = prod_init["price"]
    day_of_week = prod_init["day_of_week"]
    season = prod_init["season"]   
    theft = prod_init["theft"]
    sold = prod_init["sold"]
    rest = prod_init["rest"]
    max_prod_rows = prod_init["max_prod_rows"]
    size = prod_init["size"]
    prod_max = prod_init["prod_max"]
    prod_min = prod_init["prod_min"]
    Colli = prod_init["Colli"]
    max_delivery = prod_init["max_delivery"]
    Delivery_time = prod_init["Delivery_time"]
    days_after_order = prod_init["days_after_order"]
    Delivery_scheduled = prod_init["Delivery_scheduled"]
    new_order = prod_init["new_order"]
    delivery_amount = prod_init["delivery_amount"]
    double_scanned = prod_init["double_scanned"]
    extra_colli = prod_init["extra_colli"]
    Extra_colli_stock = prod_init["Extra_colli_stock"]
    coll_stat = prod_init["coll_stat"]
    stock_missing = prod_init["stock_missing"]
    zerotell = prod_init["zerotell"]
    zero_t = prod_init["zero_t"]
    stock = prod_init["stock"]
    actual_stock = prod_init["actual_stock"]
    stock_correctie = prod_init["stock_correctie"]
    stock_cor = prod_init["stock_cor"]
    day_rest_cor = prod_init["day_rest_cor"]
    pred_t = prod_init["pred_t"]
    pred_q10 = prod_init["pred_q10"] 
    pred_q50 = prod_init["pred_q50"]  
    pred_q90 = prod_init["pred_q90"]    
    avg_delivery_time = prod_init["avg_delivery_time"] 
    days_till_order = prod_init["days_till_order"]
    Frist_order_day = prod_init["Frist_order_day"]
    time_to_order = prod_init["time_to_order"]
    alpha = prod_init["alpha"]
    alpha_base = prod_init["alpha_base"]
    states = prod_init["states"]
    stock_q10 = prod_init["stock_q10"]
    stock_q50 = prod_init["stock_q50"]
    stock_q90 = prod_init["stock_q90"]
    Exp_date = prod_init["Exp_date"]
        
    print("product_id = ",product_id)



    
    #Will likely change this! put in initialize_product later 
    #prod_min = 10 #CHANGE!!!!!    
    # min_exp_date = np.mean(sold)
    
    # min_exp_date = int(round(min_exp_date*Colli) * 1.5) #1.5 to make a more realistic minimum 
    # #max(1, int(round(min_exp_date * Colli) * 1.5))
    # max_exp_date = int(min_exp_date * 10) # times 10 to try and engulf as many items as possible
    # Exp_date = np.random.randint(min_exp_date, max_exp_date)
    
    
    Initial_pool = [Exp_date] * actual_stock[0]
    pools = [[]]

    counted_negative_pools = set()

    #initialize expired products
    expired_products = np.zeros(T, dtype=int)


    
    
    # ====================================================================================================================                               
    #Loop over every timestep 
    # ====================================================================================================================                               
    for t in range(1, T):

        #Precompute actual stock
        actual_next_stock = (
            actual_stock[t-1]
            - sold[t]
            - theft[t]
            - rest[t]
            #+ Vault_coll_prod # In case colli was missing 1 product!
            )

        if actual_next_stock < 0:
            sold[t], theft[t], rest[t] = fsg.allocate_stock(
                actual_stock[t-1],
                sold[t],
                theft[t],
                rest[t]
            )

            #Correct actual stock
            actual_next_stock = (
                actual_stock[t-1]
                - sold[t]
                - theft[t]
                - rest[t]
                #+ Vault_coll_prod # In case colli was missing 1 product!
                )
        
        #Compute next stock in system with precomputed losses
        next_stock = (
            stock[t-1]
            - sold[t]
            - theft[t]
            - rest[t]
            #+ Vault_coll_prod # In case colli was missing 1 product!
            )
        
        # ====================================================================================================================                               
        #Manual zero-telling (System says stock is present but reality says we have no products)####
        # Check every 7 days, 
        #done in the morning!
        # ====================================================================================================================                               
        if t % 7 == 0 and t <= T and Flagzerotelling == True: 
            #print("zero telling mogelijkheid t =" , t)
            
            if actual_stock[t-1] == 0:
                #print("product_id =", product_id)
                #print("Correction t =",t)
                next_stock = 0
                zero_t[t] = 1
                sold[t] =0 
                theft[t] = 0
                rest[t] = 0
                actual_next_stock = 0
                #print("zero telling t =", t)
        else:
            zero_t[t] = 0
        
################################################################################################################
        # ============================================================
        # 1. Update prediction-based order countdown
        # ============================================================
        
        if not np.isnan(time_to_order):
            # A new predicted order moment was found
            days_till_order = int(time_to_order) - 1
            first_order_day = True
        
        elif days_till_order is not None and days_till_order > 0:
            # Count down toward predicted order day
            days_till_order -= 1
        
        
        # ============================================================
        # 2. Initialize event flags for this timestep
        # ============================================================
        
        delivery_arrived = False
        order_placed = False
        day_rest_check = False
        
        
        # ============================================================
        # 3. Check whether an already placed order arrives today
        # ============================================================
        
        order_is_pending = not new_order
        
        if order_is_pending:
            delivery_due_today = (Delivery_sceduled - days_after_order) == 0
        
            if delivery_due_today:
                delivered_stock = max_delivery * Colli
        
                next_stock += delivered_stock
                actual_next_stock += delivered_stock
        
                days_after_order = 0
                new_order = True
                delivery_arrived = True
                day_rest_check = True
        
                delivery_amount[t] = delivered_stock
        
                # ----------------------------------------------------
                # Simulate missing delivered collis
                # ----------------------------------------------------
                if Flag_col_del_mis:
                    values = np.arange(0, max_delivery + 1)
        
                    # Exponentially decreasing probabilities
                    probs = np.exp(-values)
                    probs = probs / probs.sum()
        
                    Vault_coll = np.random.choice(values, p=probs)
                else:
                    Vault_coll = 0
        
                coll_stat[t] = Vault_coll
        
                # ----------------------------------------------------
                # Simulate missing products within collis
                # ----------------------------------------------------
                p_Colli_vault_prod = fsg.Colli_vault_prod(Vault_coll, max_delivery)
        
                if Flag_col_del_prod_mis:
                    Vault_coll_prod = np.random.choice(
                        np.arange(0, max_delivery + 1),
                        p=p_Colli_vault_prod
                    )
                else:
                    Vault_coll_prod = 0
        
                stock_missing[t] = abs(coll_stat[t] * Colli - Vault_coll_prod)

                #Add experation dates for deliveries
                Added_actual_stock = delivered_stock - stock_missing[t]
                #print("Added_actual_stock =", Added_actual_stock)
                pools.append([Exp_date] * Added_actual_stock)
                
        # ============================================================
        # 4. Decide whether to place a new order
        #    Only possible if no delivery arrived this timestep
        # ============================================================
        
        can_place_order = new_order and not delivery_arrived
        
        normal_order_needed = stock[t - 1] <= prod_min
        predicted_order_needed = days_till_order == 0
        
        if can_place_order and (normal_order_needed or predicted_order_needed):
            Delivery_sceduled = Delivery_time[t]
            days_after_order = 1
            new_order = False
            order_placed = True
        
            # Reset prediction countdown if the order was prediction-triggered
            if predicted_order_needed:
                days_till_order = None
                first_order_day = False
        
        elif order_is_pending and not delivery_arrived:
            # Order is still on its way
            days_after_order += 1
        
        
        # ============================================================
        # 5. Delivery_time is only non-zero when an order is placed
        # ============================================================
        
        if not order_placed:
            Delivery_time[t] = 0

            
################################################################################################################################

        
        # =======================================================================                               
        #Add randomized double_scanned in cost if scanned products (sold or rest)
        # =======================================================================
        
        if Flag_double_scanned == True:
            if sold[t] > 0 or rest[t] > 0:
                double_scanned[t] = np.random.choice(
                [0, 1, 2],
                p=fsg.double_scan_prob(sold[t], rest[t])
                )
                next_stock = next_stock - double_scanned[t] #Register extra scanned product from stock
                
            else:
                double_scanned[t] = 0 #Double but a well :)
                
        else:
            double_scanned[t] = 0

################################################################################################################################
        
        # =======================================================================                               
        #Add randomized Extra delivery
        # =======================================================================
        
        if Flag_col_del_extra == True:
            #There is a small chance of 1 or 2 extra collies
            extra_probs = np.array([0.995, 0.004, 0.001])  # corresponds to 0,1,2 extra collies
            extra_colli[t] = np.random.choice([0, 1, 2], p=extra_probs)
            max_extra = extra_colli[t]
            
            # Calculate missing products per colli if there are extra collies comming
            if max_extra > 0 and Flag_col_extra_del_prod_mis == True:
                volume = max_extra  # treat max_extra as "delivery size"
                scale = volume / max(volume, 1)  # avoid divide by zero
                missing_values = np.arange(0, 3)  # 0,1,2 missing products per extra colli
                missing_probs = np.exp(-scale * missing_values)
                missing_probs /= missing_probs.sum()  # normalize
            
                Vault_extra_coll_prod = np.random.choice(missing_values, p=missing_probs)
            else:
                Vault_extra_coll_prod = 0
        else:
            extra_colli[t] = 0
            Vault_extra_coll_prod = 0
        
            
        Extra_colli_stock[t] = int(extra_colli[t]) * Colli - int(Vault_extra_coll_prod)

        #Add experation dates for deliveries
        Added_actual_stock_EC = Extra_colli_stock[t]
        if Added_actual_stock_EC > 0:
            Added_actual_stock_EC = Extra_colli_stock[t]
            #print("Added_actual_stock_EC = ", Added_actual_stock_EC)
            pools.append([Exp_date] * Added_actual_stock_EC)
        

################################################################################################################################

        # =======================================================================                                   
        #Daily overload products from deliveries
        # =======================================================================                               
        if Flagdayremnants == True and day_rest_check == True: # Only perform a check in case you get a delivery!
            if (actual_next_stock + Extra_colli_stock[t] - stock_missing[t]) > prod_max:
                next_stock = actual_next_stock + Extra_colli_stock[t] - stock_missing[t]
                day_rest_cor[t] = 1

################################################################################################################################

        # ====================================================================================================================                               
        # stock corrections
        ###Manual stock correctie In case stock becomes negative you correct this product ones every 7 days in the evening preferably :)
        # Check every 7 days on wensdays, (two days apart from zerotelling), 
        #done in the evening after delivery is processed on wensdays (thus missing collies and products from deliveries from that day are also taken into account!)
        # ====================================================================================================================                               
        if Flagstockcorrecties == True:
            #print("prod = ",product_id)
            if stock[t-1] < 0:
                #print("First time neg stock t =", t)
                #print("t =", t)
                stock_correctie = True
                
            if (t+2) % 7 == 0 and t <= T and stock_correctie == True: 
                #print("stock correctie at t =",t)
                next_stock = actual_next_stock + Extra_colli_stock[t] - stock_missing[t]
                stock_cor[t] = 1 
                stock_correctie = False
                             

################################################################################################################################        
        #Update stock shown in system 
        stock.append(next_stock)
        #Update the actual stock 
        actual_stock[t] = actual_next_stock + Extra_colli_stock[t] - stock_missing[t]


################################################################################################################################        

        # =======================================================================                                   
        #Keep track of experation dates of each item
        # =======================================================================                               
        pools = fsg.update_pools(
            t=t,
            pools=pools,
            Initial_pool=Initial_pool,
            sold=sold[t],
            theft=theft[t],
            rest=rest[t]
        )

        #Update experation date of each individual items 
        pools = fsg.subtract_one(pools)

                    

        #print("t =", t)
        #print("pools = ", pools)
        #print("actual_stock[t] = ", actual_stock[t])
        #print("")
        
        #Keep track of expired items 
        expired_list, counted_negative_pools = fsg.check_negative_pools(pools, counted_negative_pools)
        expired_products[t] = expired_list 
        #print("t =", t)
        #print("expired_list =", expired_list)

        #print("pools =", pools)
        #print("expired_list =", expired_items)
        #print("")
        #print("")
             
        

################################################################################################################################                
        # ====================================================================================================================                               
        #Stock prediction pmf    
        # ====================================================================================================================                               
        if Flagorderprediction_pmf == True:

            manual_check_t = (
                zero_t[t] == 1
                or stock_cor[t] == 1
                or day_rest_cor[t] == 1
            )
            
            (base_prior_pmf, states, history,
             draws, multi_draws, base_history,
             pred_q10, pred_q50, pred_q90, time_to_order
            ) = pmf_up_script.stock_prediction(
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
                manual_check=manual_check_t,
                reset_on_manual_check=True,
                n_sims=1000, #number of simulations to run
                horizon=500, #Set amount of draws
                order_quantile="q50" #order based on which prediction
            )
        


        
        # ====================================================================================================================                               
        #Stock prediction Dirichlet  
        # ====================================================================================================================                                
        if Flagorderprediction_dir == True: 
            manual_check_t = (
                zero_t[t] == 1
                or stock_cor[t] == 1
                or day_rest_cor[t] == 1)

            # manual_check_del = (
            #     stock_cor[t] == 1
            #     or day_rest_cor[t] == 1)
    
    
            
            (alpha, states, alpha_base, 
            history, draws, multi_draws, 
            alpha_history, pred_q10, pred_q50, 
            pred_q90, time_to_order
            ) = Dir_script.stock_prediction_dirichlet(
                t=t,
                Idx_pred=Idx_pred,
                stock=stock,
                sold=sold,
                delivery_amount=delivery_amount,
                rng=rng,
                pred_t=pred_t,
                avg_delivery_time=avg_delivery_time,
                window=window,
                lag=lag,
                decay=decay,
                alpha_decay=0.995,
                obs_strength=1.0,
                alpha=alpha,
                states=states,
                alpha_base=alpha_base,
                history=history,
                draws=draws,
                multi_draws=multi_draws,
                alpha_history=alpha_history,
                pred_q10=pred_q10,
                pred_q50=pred_q50,
                pred_q90=pred_q90,
                manual_check=manual_check_t,
                reset_on_manual_check=True,
                horizon=400, #set amount of draws
                n_sims=1000, #number of simulations to run
                order_quantile="q50" #order based on which prediction
            )
        

        # ====================================================================================================================                               
        #Compute uncertaintly for Stock prediction (pmf or Dirichlet)  
        # ====================================================================================================================                                
        if Flagorderprediction_pmf == True or Flagorderprediction_dir == True:
            #Determine the predictions
            #print(manual_check_t)
            if t > Idx_pred:
                if manual_check_t:
                    stock_q10[t] = actual_stock[t]
                    stock_q50[t] = actual_stock[t]
                    stock_q90[t] = actual_stock[t]
                    delivered_stock_sum = 0
                    #print(stock_q10[t])
                    #print(actual_stock[t])
                    #print("check")
                    #print(t)
                    #print(pred_q10[t])
                
                elif manual_check_t == False:
                    for k in range(1, t + 1):
                        if len(pred_q10[t - k]) > 0:
                            stock_q10[t] = int(pred_q10[t - k][0][k - 1])
                            stock_q50[t] = int(pred_q50[t - k][0][k - 1])
                            stock_q90[t] = int(pred_q90[t - k][0][k - 1])
                            
                            #print(t)
                            #print(stock_q10[t])
                            break
                    if delivery_amount[t] > 0:
                        delivered_stock_sum += delivered_stock
                        # stock_q10[t] += delivered_stock_sum
                        # stock_q50[t] += delivered_stock_sum
                        # stock_q90[t] += delivered_stock_sum
                    
                    stock_q10[t] += delivered_stock_sum
                    stock_q50[t] += delivered_stock_sum
                    stock_q90[t] += delivered_stock_sum

                
                                        
                
            
            


    
    # ====================================================================================================================                                  
    #Add all variables to a dataframe
    # ====================================================================================================================  
    
    df_product = pd.DataFrame({
        "Product": [product_id]*T,
        "time": list(range(1, T+1)),
        "theft": theft,
        "Price": [price]*T,
        "day_of_week": day_of_week,
        "season": season,
        "stock": stock,
        "sold": sold,
        "Rest": rest,
        "prod_max": prod_max,
        "prod_min":prod_min,
        "max_delivery": max_delivery,
        "stock missing (C_M+P_M_C)": stock_missing,
        "Double scanned": double_scanned,
        "Colli extra (EX_C+P_M_C)": Extra_colli_stock,
        "actual_stock": actual_stock,
        "Colli": [Colli]*T,
        "day rest correctie": day_rest_cor, 
        "stock correctie": stock_cor,
        "zerotelling": zero_t,
        "Delivery time": Delivery_time,
        "delivery_amount": delivery_amount,
        "stock_q10": stock_q10,
        "stock_q50": stock_q50,
        "stock_q90": stock_q90, 
        "expired_products":expired_products,
    })

    all_products.append(df_product)
    
    empty_row = pd.DataFrame({
    "Product": [None],
    "time": [None],
    "theft": [None],
    "Price": [None],
    "day_of_week": [None],
    "season": [None],
    "stock": [None],
    "sold": [None],
    "Rest": [None],
    "prod_max": [None],
    "prod_min":[None],
    "max_delivery": [None],
    "stock missing (C_M+P_M_C)": [None],
    "Double scanned": [None],
    "Colli extra (EX_C+P_M_C)": [None],
    "actual_stock": [None],
    "Colli": [None],
    "day rest correctie": [None],
    "stock correctie": [None],
    "zerotelling": [None],
    "Delivery time": [None],
    "delivery_amount": [None],
    "stock_q10": [None],
    "stock_q50": [None],
    "stock_q90": [None],
    "expired_products": [None],
    })
    all_products.append(empty_row)

# Combine dataset
df_no_pred = pd.concat(all_products, ignore_index=True)

#Save data from first product 
#one_product = df_dir[df_dir["Product"] == 0]
