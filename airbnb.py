"""
https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
"""
# Imports
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('airbnb.csv')
data = data.loc[:,[data.columns[4],data.columns[5],data.columns[6],data.columns[7],data.columns[8],
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
plt.title('Room-type Distributions', fontsize='15',color='b')
plt.show()

# 2 - Bar plot
