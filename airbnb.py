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
plt.pie(room_type['n_rooms'],autopct='%1.2f%%', colors=['darkcyan','skyblue','powderblue'])
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

plt.figure(figsize=(12,8))
plt.bar(price_counts['index'],price_counts['price'],color='steelblue')
plt.ylim((0,1000))
plt.xlim((0,1500))
plt.show()