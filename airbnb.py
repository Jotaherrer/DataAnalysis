"""
https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
"""
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('airbnb.csv')
data = data.loc[:,[data.columns[4],data.columns[5],data.columns[6],data.columns[7],data.columns[8],data.columns[9],
                   data.columns[10],data.columns[11],data.columns[13],data.columns[14],data.columns[15]]]
# Explore data
data.info()
data.head()
# Review nulls
data[~data['reviews_per_month'].isnull()].count().values[0]
data[~data['reviews_per_month'].isnull()]
data.fillna(0,inplace=True)
data.info()
# Check duplicate values
sum(data.duplicated())

# Plot visualizations to understand the dataset
# 1 - Pie chart
room_type = data.groupby('room_type')['latitude'].count().reset_index()
room_type.rename(columns={'latitude':'n_rooms'},inplace=True)

plt.figure(figsize=(10,8))
plt.pie(room_type['n_rooms'],autopct='%1.2f%%', colors=['darkcyan','steelblue','powderblue'])
plt.axis('equal')
plt.legend(labels=room_type['room_type'],loc='best')
plt.title('Room-type Rental Distributions', fontsize='15',color='b')
plt.show()
plt.close()

# 2 - Bar plot with neighbourhood distribution
neighbourhood = data.groupby('neighbourhood_group')['neighbourhood'].count().reset_index()

fig,ax = plt.subplots(figsize=(12,8))
sns.barplot(x=neighbourhood[neighbourhood.columns[0]],
            y=neighbourhood[neighbourhood.columns[1]],
            color='#004488',
            ax=ax)
sns.lineplot(x=neighbourhood[neighbourhood.columns[0]],
             y=neighbourhood[neighbourhood.columns[1]],
             color='r',
             marker='o',
             ax=ax)
plt.ylabel('Count', fontsize='15')
plt.xlabel('Neighbourhood',fontsize='15')
plt.title('Rental Distribution by Neighbourhood Group',fontsize='15')
plt.grid('x')
plt.show()
sns.set()

# 3 - Bar plot with price distribution
price = data.loc[:,['neighbourhood','price']].set_index('neighbourhood')
price_stats = data['price'].describe().reset_index()
price_counts = price.price.value_counts().reset_index()
price_counts.rename(columns={'index':'price','price':'count'},inplace=True)

fig2,ax = plt.subplots(figsize=(12,8))
fig2.patch.set_facecolor('lightgray')
ax.set_facecolor('lightgray')
plt.hist(price_counts['price'],bins=30,color='#004488',edgecolor='salmon')
ax.set_xticks(range(0,10000,500))
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
plt.xlabel('Price',fontsize='15')
plt.ylabel('Count', fontsize='15')
plt.xlim((-0.5,10000))
plt.title('New York Price-Rental Distribution',fontsize='15')
plt.show()

# 4 - Bar plot with price to location distribution
loc_price = data.groupby(['neighbourhood_group','room_type'])['price'].mean().reset_index()
locations = loc_price.neighbourhood_group.unique()

x_rooms1 = [3 * element + 0.8*1 for element in range(5)]
x_rooms2 = [3 * element + 0.8*2 for element in range(5)]
x_rooms3 = [3 * element + round(0.8*3,2) for element in range(5)]
y_values1 = loc_price[loc_price['room_type'] == 'Entire home/apt']['price'].values
y_values2 = loc_price[loc_price['room_type'] == 'Private room']['price'].values
y_values3 = loc_price[loc_price['room_type'] == 'Shared room']['price'].values

fig3,ax2 = plt.subplots(figsize=(14,8))
fig3.patch.set_facecolor('lightgray')
ax2.set_facecolor('lightgray')

plt.bar(x_rooms1, y_values1, color='purple', edgecolor='b')
plt.bar(x_rooms2, y_values2, color='b', edgecolor='b')
plt.bar(x_rooms3, y_values3, color='yellowgreen', edgecolor='b')

ax2.set_xticks(range(1,16,3))
ax2.set_xticklabels(locations, fontsize='12')
for tick in ax2.get_xticklabels():
    tick.set_rotation(45)

plt.xlabel('Location/Room-type',fontsize='15')
plt.ylabel('Prices', fontsize='15')
plt.legend(labels=loc_price.room_type.unique(), loc='best')
plt.title('New York Price-Rental Distribution',fontsize='15')
plt.show()
