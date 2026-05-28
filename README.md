# Inventory stock model with stock predictions


## files:
  - Model_git.ipynb generates the data (Also saves the data as a csv file)
  - Analysis.ipynb is a example analysis file for the output of the model
  - Dirichle folder includes the inventory stock predictions based on updating a pmf and using a dirichlet distribution
  - pmf folder includes the inventory stock predictions based on pmf only (works not as good as dirichlet distribution)
  - Functions_stock_generation.py & Initialize_product.py are files called in the Model_git.ipynb for stock generation



## Model 

<img width="3245" height="1688" alt="Overvieuw" src="https://github.com/user-attachments/assets/7eb1e53e-6c97-4cc8-b608-8048d142dd99" />

### Initialzation 

  - Each product is initialized with draws from a randomly chosen probability mass function (pmf) for the whole timeseries for selling (Sold) and Scanning (Rest; inclused broken items, expired items etc...). Theft is initilazed with draws from a pmf which is influenced by the price of the product, which is randomly generated.
  
  - Depending on the average amount of items sold each week and the physical size of the items (generated randomly), a maximum (how many items can physically fit a shelf), minimum amount (minimum amount of prefered items to have in stock), and colli size (amount of items delivered per delivery) of items is calculated. 
  
  - Delivery times for each item are generated randomly from a predetermined pmf.



### Stock and actual stock

Note that there is an important distinction between the actual stock (actual_stock) and the stock (stock or next_stock) shown in the system. The stock in the system does not include the effects of Theft or missing deliveries and also is subjected to accidentaly double scanning during selling of products at the payment desks. Therefore three types of stock corrections are made performed troughout the simulation as to correct the stock in the system to the actual stock levels:
  - Flagzerotelling: Once every 7 days a manual stock check is performed on the products that have no actual stock present.
  - Flagstockcorrecties: The system flags a certain product when this product has a negative stock value. Once every 7 days a manual stock check is performed on the products that have been flagged
  - Flagdayremnants: When no more items can physically fit into the shelf anymore after a delivery, a manual stock check is performed.



### Main programme

- First a new_stock[t] and actual_stock[t] is computed by substracting the Sold[t], Rest[t] and Theft[t] from the previous stock[t-1] and previous actual stock[t-1]. 

- Flagzerotelling: Actual stock[t] is manually check in case there is no stock present.

- In case a delivery has arrived; twice the amount present within a colli is added to the next_stock[t]. And a corrected amount (subjected to losses of products during the delivery process) of items is added to the actual_stock[t]

- Baseline: In case the previous stock[t-1] is equal or has fallen below the predetermined minimum amount threshold, a new order is placed and a delivery time is determined. Dirichlet: A new order is made just before the stock is predicted to be nearly empty (1 product left) taking into account the delivery time. (Note this part is actively being worked on!)

- Based on how many items are sold[t] or scanned (rest[t]), a pmf is construced an a random draw is made which consitutes to how often a product is double scanned in the current timestep. The result is then substracted from the new_stock[t].

- In case there is an unplanned extra delivery (subjected to losses during delivery) add these items to the actual_stock[t].

- Flagdayremnants: When no more items can physically fit into the shelf anymore (Actual_stock[t] > maximum amount of items) after a delivery, a manual stock check is performed.

- Flagstockcorrecties: When the privious stock (stock[t-1]) is lower then 0 the item is flagged. Then once every 7 days a manual stock check is performed on the products that have been flagged.

- pmf prediction (currently underperforming)

- Dirichlet prediction: Compute a prior based on the first half of the data -> Compute multiple distributions with a dirichlet after a delivery arrived -> make multiple draws from these distributions -> compute the mean (q50) and the 10th/90th percentile -> Determine optimal ordering date for future timesteps. Update the prior based on long term trends and short term trend.

- Save variables


## Example Plots 
Note in both simulations extra delivery's and missing delivery's are not active.

### Baseline no predictions
<img width="3567" height="1168" alt="no_prediction_1" src="https://github.com/user-attachments/assets/ad4011a5-574f-41d6-8e86-0ec47cef1679" />

### Dirichlet prediction
<img width="3567" height="1168" alt="dirichlet_q50_prediction_2" src="https://github.com/user-attachments/assets/38abf32a-6a2f-4fff-acb9-dc981bb91ae9" />



