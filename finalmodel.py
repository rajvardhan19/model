import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from fcmeans import FCM

# Load the data
spending_data = pd.read_csv('spending.csv')

# Clean the data
spending_data = spending_data.dropna()
spending_data['date'] = pd.to_datetime(spending_data['date'], format='%d-%m-%Y')



# Calculate total spending by category
category_spending = spending_data.groupby('category')['amount'].sum().reset_index()

# Bar chart of total spending by category
plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='amount', data=category_spending)
plt.title('Total Spending by Category')
plt.xlabel('Category')
plt.ylabel('Total Spending')
plt.show()

# Analyze spending by day of the week
spending_data['Day_of_week'] = spending_data['date'].dt.dayofweek
day_of_week_spending = spending_data.groupby('Day_of_week')['amount'].sum()
plt.figure(figsize=(8,6))
sns.barplot(x=day_of_week_spending.index, y=day_of_week_spending.values, palette='Blues_d')
plt.title('Total Spending by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Spending')
plt.show()

# Analyze spending by description
top_descriptions = spending_data['description'].value_counts().head(10)
plt.figure(figsize=(8,6))
sns.barplot(x=top_descriptions.index, y=top_descriptions.values, palette='Blues_d')
plt.title('Top 10 Most Frequent Descriptions')
plt.xlabel('Description')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


# Conduct a t-test to compare spending on weekdays vs weekends
weekday_spending = spending_data.loc[spending_data['Day_of_week'] < 5]['amount']
weekend_spending = spending_data.loc[spending_data['Day_of_week'] >= 5]['amount']
t, p = ttest_ind(weekday_spending, weekend_spending, equal_var=False)
print('T-test results: t = {:.2f}, p = {:.4f}'.format(t, p))
if p < 0.05:
    print('There is a significant difference in spending between weekdays and weekends.')
else:
    print('There is no significant difference in spending between weekdays and weekends')


# Convert date column to datetime format and set it as the index
spending_data['date'] = pd.to_datetime(spending_data['date'])
spending_data.set_index('date', inplace=True)

# Create a pivot table to aggregate the spending by category and month
pivot_table = spending_data.pivot_table(index=pd.Grouper(freq='M'), columns='category', values='amount', aggfunc='sum')

# Compute the correlation matrix between categories
corr_matrix = pivot_table.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between Spending Categories')
plt.show()

# Pie chart of spending by category
plt.figure(figsize=(10, 6))
plt.pie(category_spending['amount'], labels=category_spending['category'], autopct='%1.1f%%')
plt.title('Spending by Category')
plt.show()

# Line chart of daily spending over time
daily_spending = spending_data.groupby('date')['amount'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='date', y='amount', data=daily_spending)
plt.title('Daily Spending Over Time')
plt.xlabel('Date')
plt.ylabel('Spending')
plt.show()

# Scatter plot of spending by category and date
plt.figure(figsize=(10, 6))
sns.scatterplot(x='date', y='category', hue='amount', size='amount', data=spending_data)
plt.title('Spending by Category and Date')
plt.xlabel('Date')
plt.ylabel('Category')
plt.show()

# Preprocess the data
scaler = MinMaxScaler()
spending_data_scaled = scaler.fit_transform(spending_data[['amount']])

# Choose the number of clusters
n_clusters = 3
cluster_scores = []
for n in range(2, 10):
    fcm = FCM(n_clusters=n, m=2)
    fcm.fit(spending_data_scaled)
    cluster_scores.append(silhouette_score(spending_data_scaled, fcm.u.argmax(axis=1)))
n_clusters = np.argmax(cluster_scores) + 2

# Apply FCM clustering
fcm = FCM(n_clusters=n_clusters, m=2)
fcm.fit(spending_data_scaled)
spending_data['Cluster'] = fcm.u.argmax(axis=1)

# Compute the average spending for each cluster
cluster_means = spending_data.groupby('Cluster').mean()

# Identify the cluster with the highest average spending
high_spending_cluster = cluster_means['amount'].idxmax()

# Provide recommendations for sustainable behavior based on the high-spending cluster
if high_spending_cluster == 0:
    print("You might want to consider reducing your spending on entertainment.")
elif high_spending_cluster == 1:
    print("You might want to consider reducing your spending on food.")
elif high_spending_cluster == 2:
    print("You might want to keep yourself healthy.")
else:
    print("You might want to consider reducing your spending on traveling.")

# Preprocess data
spending_data['date'] = pd.to_datetime(spending_data['date'])
spending_data['month'] = spending_data['date'].dt.month
spending_data['weekday'] = spending_data['date'].dt.weekday
spending_data['is_weekend'] = spending_data['weekday'] >= 5
spending_data['days_in_month'] = spending_data['date'].dt.days_in_month
spending_data['category'] = pd.Categorical(spending_data['category'], categories=['Food', 'Transportation', 'Entertainment'], ordered=True)
spending_data['category_code'] = spending_data['category'].cat.codes

# Feature selection
features = ['days_in_month', 'is_weekend', 'month', 'weekday', 'category_code']
X = spending_data[features]
y = spending_data['amount']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next month's spending
next_month_start_date = spending_data['date'].max() + pd.DateOffset(days=1)
next_month_end_date = next_month_start_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)
next_month_dates = pd.date_range(next_month_start_date, next_month_end_date, freq='D')
next_month_df = pd.DataFrame({'date': next_month_dates})
next_month_df['month'] = next_month_df['date'].dt.month
next_month_df['weekday'] = next_month_df['date'].dt.weekday
next_month_df['is_weekend'] = next_month_df['weekday'] >= 5
next_month_df['days_in_month'] = next_month_df['date'].dt.days_in_month
next_month_df['category_code'] = 0  # Assume food category
next_month_spending = model.predict(next_month_df[features])
predicted_spending = np.sum(next_month_spending)
print(f"Your predicted spending for next month is: ${predicted_spending:.2f}")
if predicted_spending > 500:
    print("You might want to consider reducing your spending.")