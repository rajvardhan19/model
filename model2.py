import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from fcmeans import FCM

# Load the user's spending data
spending_data = pd.read_csv('spending.csv')

# Clean the data
spending_data = spending_data.dropna()
spending_data['date'] = pd.to_datetime(spending_data['date'], format='%d-%m-%Y')

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

# Use linear regression to predict future spending behavior
X = spending_data[['date']]
y = spending_data['amount']

model = LinearRegression()
model.fit(X, y)

# Predict future spending for the next month
next_month = pd.date_range(start='2023-01-01', end='2023-05-31')
next_month_df = pd.DataFrame({'date': next_month})
next_month_df['Month'] = next_month_df['date'].dt.month
next_month_df['Year'] = next_month_df['date'].dt.year
next_month_df['Days_in_month'] = next_month_df['date'].dt.daysinmonth
next_month_df['Weekday'] = next_month_df['date'].dt.weekday
next_month_df['Is_weekend'] = np.where(next_month_df['Weekday'] < 5, 0, 1)
next_month_df = next_month_df.drop('date', axis=1)

predicted_spending = model.predict(next_month_df)

# Provide personalized advice based on the predicted spending
if predicted_spending.mean() > spending_data['amount'].mean():
    print("You might want to consider creating a budget to manage your spending.")
else:
    print("Your spending is within a healthy range.")
