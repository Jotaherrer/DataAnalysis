import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

"""
https://www.kaggle.com/ssaketh97/black-friday-sale-analysis
https://towardsdatascience.com/a-step-by-step-guide-for-creating-advanced-python-data-visualizations-with-seaborn-matplotlib-1579d6a1a7d0

data:
https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
https://towardsdatascience.com/interpreting-data-through-visualization-with-python-matplotlib-ef795b411900
https://towardsdatascience.com/a-beginners-guide-to-data-visualization-with-python-49f1d257c781
https://towardsdatascience.com/practical-statistics-visualization-with-python-plotly-770e96e35067
https://towardsdatascience.com/complete-guide-to-data-visualization-with-python-2dd74df12b5e
https://towardsdatascience.com/heatmap-basics-with-pythons-seaborn-fb92ea280a6c
https://towardsdatascience.com/5-visualisations-to-level-up-your-data-story-e131759c2f41
https://medium.com/@jaimejcheng/data-exploration-and-visualization-with-seaborn-pair-plots-40e6d3450f6d

finance:
https://medium.com/fintechexplained/credit-spread-in-finance-and-their-probability-distributions-in-data-science-b34650214cc0
https://towardsdatascience.com/how-nlp-has-evolved-for-financial-sentiment-analysis-fb2990d9b3ed
https://medium.com/@codingfun89/analysing-companies-leverage-with-python-eea490689c5b
https://towardsdatascience.com/beat-the-stock-market-with-machine-learning-d9432ea5241e
https://towardsdatascience.com/pull-and-analyze-financial-data-using-a-simple-python-package-83e47759c4a7
https://towardsdatascience.com/stock-analysis-in-python-a0054e2c1a4c
https://medium.com/financeexplained/finding-historical-data-sets-for-financial-analysis-7d77d8031e97
https://towardsdatascience.com/how-to-simulate-trades-in-python-7e613c83fd5a
https://towardsdatascience.com/quants-guide-finding-key-metrics-ratios-using-python-390bb7873e62
https://towardsdatascience.com/how-to-code-different-types-of-moving-averages-in-python-4f8ed6d2416f
https://medium.com/swlh/creating-and-back-testing-a-pairs-trading-strategy-in-python-caa807b70373
https://medium.com/@kaabar.sofien/the-normalized-bollinger-indicator-another-way-to-trade-the-range-back-testing-in-python-db22c111cdde
https://medium.com/@kaabar.sofien/trading-performance-measurement-the-necessary-tools-and-metrics-8bcca001f5c6
https://medium.com/stockswipe-trade-ideas/option-greeks-not-greek-anymore-9e883273e6e6
https://towardsdatascience.com/option-greeks-in-python-97980df3ab0b
https://medium.com/financeexplained/what-are-the-greeks-2f79caa2f61f
https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

"""

data = pd.read_csv('./black_friday/blackFriday.csv')
data.head()
data.describe()
data.columns

# Exploring data
data.info()

# Clean nulls
data.isnull().sum()
data.fillna(0, inplace=True) # Re-check N/A was replaced with 0.

# Group by buyer
purchases = data.groupby(['User_ID']).sum().reset_index()
data[data['User_ID'] == 1000001]

# Check duplicate values
sum(data.duplicated())

# which age group of customers is more likely to spend more money?
purchase_by_age = data.groupby('Age')['Purchase'].mean().reset_index()
purchase_by_age.set_index('Age', inplace=True)

plt.figure(figsize=(16,4))
plt.plot(purchase_by_age.index, purchase_by_age.values, color='purple', marker='*')
plt.grid()
plt.xlabel('Age Group', fontsize=10)
plt.ylabel('Total Purchases in $', fontsize=10)
plt.title('Average Sales distributed by age group', fontsize=15)
plt.show()

info = data.groupby('Age')['Purchase'].sum()
info  = pd.DataFrame({'Age':info.index, 'Average_purchase':info.values})
plt.figure(figsize = (16,4))
plt.plot('Age','Average_purchase','ys-',data = info)
plt.grid()
plt.xlabel('Age group')
plt.ylabel('Total amount in $')
plt.title('Age group vs total amount spent')


# which age group and gender have higher visiting rate to the retail store?
age_and_gender = data.groupby('Age')['Gender'].count().reset_index()
gender = data.groupby('Gender')['Age'].count().reset_index()

plt.figure(figsize=(10,6))
plt.pie(age_and_gender['Gender'], labels=age_and_gender['Age'],autopct='%d%%', colors=['cyan', 'steelblue','peru','blue','yellowgreen','salmon','#0040FF'])
plt.axis('equal')
plt.title("Age Distribution", fontsize='20')
plt.show()

plt.figure(figsize=(10,6))
plt.pie(gender['Age'], labels=gender['Gender'],autopct='%d%%', colors=['salmon','steelblue'])
plt.axis('equal')
plt.title("Gender Distribution", fontsize='20')
plt.show()


# which occupation type have highest purchase rate?
occupation = data.groupby('Occupation')['Purchase'].mean().reset_index()

sns.set(style="white", rc={"lines.linewidth": 3})
fig, ax1 = plt.subplots(figsize=(10,8))

sns.barplot(x=occupation['Occupation'],
            y=occupation['Purchase'],
            color='#004488',
            ax=ax1)

sns.lineplot(x=occupation['Occupation'],
             y=occupation['Purchase'],
             color='salmon',
             marker="o",
             ax=ax1)
plt.axis([-1,21,8000,10000])
plt.title('Occupation Bar Chart', fontsize='15')
plt.show()
sns.set()


# Top 10 products which made highest sales in the store?
product = data.groupby('Product_ID')['Purchase'].count().reset_index()
product.rename(columns={'Purchase':'Count'},inplace=True)
product_sorted = product.sort_values('Count',ascending=False)

plt.figure(figsize=(14,6))
plt.plot(product_sorted['Product_ID'][:10], product_sorted['Count'][:10], linestyle='-', color='purple', marker='o')
plt.title("Best-selling Products", fontsize='15')
plt.xlabel('Product ID', fontsize='15')
plt.ylabel('Products Sold', fontsize='15')
#for a,b,c in zip(product_sorted['Product_ID'], product_sorted['Count'], product_sorted['Count']):
#    plt.text(a,b,str(c))
plt.show()


# Which product is more popular for each age group?
popular = data.groupby('Age')['Product_ID'].apply(lambda x: x.value_counts().index[0]).reset_index()


# How age and gender would affect the average purchased item price?
df_gp_1 = data[['User_ID', 'Purchase']].groupby('User_ID').agg(np.mean).reset_index()
df_gp_2 = data[['User_ID', 'Gender', 'Age']].groupby('User_ID').agg(max).reset_index()
df_gp = pd.merge(df_gp_1, df_gp_2, on = ['User_ID'])

freq = ((df_gp.Age.value_counts(normalize = True).reset_index().sort_values(by = 'index').Age)*100).tolist()
number_gp =7

def ax_settings(ax, var_name, x_min, x_max):
    ax.set_xlim(x_min,x_max)
    ax.set_yticks([])

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.spines['bottom'].set_edgecolor('#444444')
    ax.spines['bottom'].set_linewidth(2)

    ax.text(0.02, 0.05, var_name, fontsize=17, fontweight="bold", transform = ax.transAxes)
    return None

fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(nrows=number_gp,
                       ncols=2,
                       figure=fig,
                       width_ratios= [3, 1],
                       height_ratios= [1]*number_gp,
                       wspace=0.2, hspace=0.05
                      )

ax = [None]*(number_gp + 1)
features = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']

for i in range(number_gp):
    ax[i] = fig.add_subplot(gs[i, 0])

    ax_settings(ax[i], 'Age: ' + str(features[i]), -1000, 20000)

    sns.kdeplot(data=df_gp[(df_gp.Gender == 'M') & (df_gp.Age == features[i])].Purchase,
            ax=ax[i], shade=True, color="blue",  bw=300, legend=False)
    sns.kdeplot(data=df_gp[(df_gp.Gender == 'F') & (df_gp.Age == features[i])].Purchase,
            ax=ax[i], shade=True, color="red",  bw=300, legend=False)

    if i < (number_gp - 1):
        ax[i].set_xticks([])

ax[0].legend(['Male', 'Female'], facecolor='w')

ax[number_gp] = fig.add_subplot(gs[:, 1])
ax[number_gp].spines['right'].set_visible(False)
ax[number_gp].spines['top'].set_visible(False)
ax[number_gp].barh(features, freq, color='#004c99', height=0.4)
ax[number_gp].set_xlim(0,100)
ax[number_gp].invert_yaxis()
ax[number_gp].text(1.09, -0.04, '(%)', fontsize=10, transform = ax[number_gp].transAxes)
ax[number_gp].tick_params(axis='y', labelsize = 14)

plt.show()


# Top and worst buyers
top_buyers = purchases.sort_values('Purchase', ascending=False).head(10)
top_buyers = top_buyers.loc[:,['User_ID','Purchase']]
top_buyers['ID'] = top_buyers['User_ID'].astype(str)
top_buyers = top_buyers.loc[:,['ID', 'Purchase']]

worst_buyers = purchases.sort_values('Purchase', ascending=True).head(10)
worst_buyers = worst_buyers.loc[:,['User_ID','Purchase']]
worst_buyers['ID'] = worst_buyers['User_ID'].astype(str)
worst_buyers = worst_buyers.loc[:,['ID', 'Purchase']]


# Visualizations
# Top Buyers
fig,ax = plt.subplots(figsize=(14,8))
fig.patch.set_facecolor('xkcd:salmon')
ax.set_facecolor('xkcd:gray')

ax.bar(x=list(range(len(top_buyers['ID']))), height=top_buyers['Purchase'], color='peru',edgecolor='blue')

ax.set_yticks(top_buyers['Purchase'])
#ax.set_yticklabels(range(10))

ax.set_xticks(list(range(len(top_buyers['ID']))))
ax.set_xticklabels(top_buyers['ID'], size=12)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

ax.set_xlabel(xlabel='User ID', size=15)
ax.set_ylabel(ylabel='Amount Purchased', size=15)

ax.set_title('Top Buyers', fontsize=15)
plt.show()
