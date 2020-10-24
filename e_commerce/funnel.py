import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
ANALYZING DATA TO BUILD A FUNNEL => description of how many people continue to the next step of a multi-step process.
1) A user visits CoolTShirts.com
2) A user adds a t-shirt to their cart
3) A user clicks “checkout”
4) A user actually purchases a t-shirt
"""
## INSPECT DATA
visits = pd.read_csv('visits.csv', parse_dates=[1])
cart = pd.read_csv('cart.csv', parse_dates=[1])
checkout = pd.read_csv('checkout.csv', parse_dates=[1])
purchases = pd.read_csv('purchases.csv', parse_dates=[1])

## MERGE DATAFRAMES
visits_cart = pd.merge(visits,cart,how='left')                  # => LEFT-MERGE DATAFRAMES TO  DETERMINE WHO PASSES TO THE CART PHASE
visits_cart['cart_ok'] = visits_cart['cart_time'].isnull()      # => CREATE COLUMN FOR TRUE IF CUSTOMER PASSES, FALSE IF NOT PASSES
not_passes = visits_cart[visits_cart['cart_ok'] == True].count()[0] 
passes = visits_cart[visits_cart['cart_ok'] == False].count()[0] 
percent_cart_ok = passes / (not_passes+passes)
percent_cart_not = not_passes / (not_passes+passes)

cart_checkout = pd.merge(cart,checkout,how='left')                       # => LEFT-MERGE DATAFRAMES TO DETERMINE WHO PASSES TO THE CHECKOUT PHASE
cart_checkout['check_ok'] = ~cart_checkout['checkout_time'].isnull()      # => CREATE COLUMN FOR TRUE IF CUSTOMER PASSES, FALSE IF NOT PASSES
passes2 = cart_checkout[cart_checkout['check_ok'] == True].count()[0]    
not_passes2 = cart_checkout[cart_checkout['check_ok'] == False].count()[0]
percent_check_ok = passes2 / (passes2+not_passes2)
percent_check_not = not_passes2 / (passes2+not_passes2)

check_purchase = pd.merge(checkout,purchases,how='left')
check_purchase['ok'] = ~check_purchase['purchase_time'].isnull()
purchase_ok = check_purchase[check_purchase['ok'] == True]
purchase_ok_q = purchase_ok.count()[0]
purchase_not_q = len(check_purchase) - purchase_ok_q
percent_pur_ok = purchase_ok_q / (purchase_ok_q+purchase_not_q)
percent_pur_not = purchase_not_q / (purchase_ok_q+purchase_not_q)

negative_percentages = [percent_cart_not, percent_check_not, percent_pur_not]

fig = plt.figure(figsize=(10,9))
ax1 = fig.subplots()
plt.bar(list(range(1,4)), negative_percentages)
ax1.set_xticks(list(range(1,4)))
ax1.set_xticklabels(['CART', 'CHECKOUT', 'PURCHASE'])
plt.title('MKT FUNNEL - LOSSES PER STAGE')
plt.show()

## FOR EACH CUSTOMER THAT ACTUALLY PURCHASES, WE CAN ANALYZE HOW MUCH TIME EVERY STAGE TAKES AND RELATED STATISTICS
all_data = visits.merge(cart, how='left').merge(checkout, how='left').merge(purchases, how='left')
all_data['time_to_purchase'] = all_data['purchase_time'] - all_data['visit_time']
all_data_clean = all_data[~all_data.time_to_purchase.isnull()]
stats = all_data_clean.time_to_purchase.describe().reset_index()