"""
https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
http://insideairbnb.com/get-the-data.html
"""
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('new_york.csv')
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

plt.figure(figsize=(14,11))
plt.pie(room_type['n_rooms'],autopct='%1.2f%%', colors=['darkcyan','steelblue','powderblue'])
plt.axis('equal')
plt.legend(labels=room_type['room_type'],loc='best',fontsize='12')
plt.title('Room-type Rental Distribution', fontsize='15',color='b')
plt.savefig('image1.png')
plt.show()
plt.close()

# 2 - Bar plot with neighbourhood distribution
neighbourhood = data.groupby('neighbourhood_group')['neighbourhood'].count().reset_index()

fig,ax = plt.subplots(figsize=(14,11))
sns.barplot(x=neighbourhood[neighbourhood.columns[0]],
            y=neighbourhood[neighbourhood.columns[1]],
            color='#004488',
            ax=ax)
sns.lineplot(x=neighbourhood[neighbourhood.columns[0]],
             y=neighbourhood[neighbourhood.columns[1]],
             color='r',
             marker='o',
             ax=ax)
plt.ylabel('Rentals', fontsize='15')
plt.xlabel('Borough',fontsize='15')
plt.title('Rental Distribution by Neighbourhood Group',fontsize='15')
plt.grid('x')
plt.savefig('image2.png')
plt.show()
sns.set()

# 3 - Histogram plot with price distribution
price = data.loc[:,['neighbourhood','price']].set_index('neighbourhood')
price_stats = data['price'].describe().reset_index()
price_counts = price.price.value_counts().reset_index()
price_counts.rename(columns={'index':'price','price':'count'},inplace=True)

fig2,ax = plt.subplots(figsize=(14,11))
fig2.patch.set_facecolor('lightgray')
ax.set_facecolor('lightgray')
plt.hist(price_counts['price'],bins=30,color='#004488',edgecolor='salmon')
ax.set_xticks(range(0,10000,500))
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
plt.xlabel('Price',fontsize='15')
plt.ylabel('Rentals', fontsize='15')
plt.xlim((-0.5,10000))
plt.title('New York Price-Rental Distribution',fontsize='15')
plt.savefig('image3.png')
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

fig3,ax2 = plt.subplots(figsize=(16,11))
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
plt.title('New York Price-Rental Distribution by Location and Room-type',fontsize='15')
plt.savefig('image4.png')
plt.show()

# 5 - Most reviewed spots
review = data.sort_values('number_of_reviews',ascending=False)
top_reviewed = review.loc[:,['neighbourhood','number_of_reviews']][:20]
top_reviewed = top_reviewed.groupby('neighbourhood').mean().sort_values('number_of_reviews',ascending=False).reset_index()

fig4,ax3 = plt.subplots(figsize=(12,8))
sns.barplot(x=top_reviewed['neighbourhood'],
            y=top_reviewed['number_of_reviews'].values,
            color='yellowgreen',
            ax=ax3)
plt.plot(top_reviewed['number_of_reviews'], marker='o', color='red',linestyle='--')
plt.ylabel('Reviews', fontsize='15')
plt.xlabel('Location',fontsize='15')
plt.ylim((400,580))
for ax in ax3.get_xticklabels():
    ax.set_rotation(50)
plt.title('Most-Reviewed Rentals by location',fontsize='15')
plt.savefig('image5.png')
plt.show()
sns.set()

# 6 - Cheapest in the UES with most reviews
import numpy as np
upper_east = data[data['neighbourhood'] == 'Upper East Side']
ninetieth_percentile = np.quantile(upper_east['number_of_reviews'], 0.85)
upper_east = upper_east[upper_east['number_of_reviews'] >= ninetieth_percentile]
upper_east = upper_east.sort_values('price',ascending=True)

private_room = upper_east[upper_east['room_type'] == 'Private room'].reset_index()
entire_home = upper_east[upper_east['room_type'] == 'Entire home/apt'].reset_index()
shared_room = upper_east[upper_east['room_type'] == 'Shared room'].reset_index()
private_cheapest = private_room.loc[0,:].reset_index()
private_cheapest.rename(columns={'index':'data','0':'values'},inplace=True)
entire_cheapest = entire_home.loc[0,:].reset_index()
entire_cheapest.rename(columns={'index':'data','0':'values'},inplace=True)


plt.figure(figsize=(10,8))
plt.bar()