import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#----------> Data Exploration & Cleaning

#1: Load Data
df = pd.read_excel("Online Retail.xlsx")

#2: Inspect first few rows of the dataset

# print(df.head())
# print(df.info())
# print(df.describe())

#3: Handle missing values

# Check for missing values
# print(df.isnull().sum())
# Dropping missing values from Description and CustomerID
df.dropna(subset=["Description","CustomerID"], inplace=True)   

#4: Remove duplicates
# Drop duplicate rows based on all columns
df.drop_duplicates(inplace=True)

#5: checking and converting datatypes
# print(df.info())

#4: Correct Negative and Zero Quantities

# Check the distribution of quantities before cleaning
# print(f"Number of records with zero or negative quantities: {df[df['Quantity'] <= 0].shape[0]}")

#Remove rows where the quantity or unitprize is zero or negative
# Remove negative quantities and unit prices
df_cleaned = df[(df.Quantity > 0) & (df.UnitPrice > 0)].copy()

# Remove canceled orders (those with 'C' in StockCode)
df_cleaned['StockCode'] = df_cleaned['StockCode'].astype(str)
df_cleaned = df_cleaned[~df_cleaned['StockCode'].str.startswith('C')]

# Create a 'TotalPrice' column
df_cleaned['TotalPrice'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']

# Check the cleaned data
# print(f"Number of records after cleaning: {df_cleaned.shape[0]}")
print(df_cleaned.head())

#-------------> RFM Analysis

# Convert InvoiceDate to datetime
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])

# Snapshot date (one day after the last purchase)
snapshot_date = df_cleaned['InvoiceDate'].max() + dt.timedelta(days=1)

# Aggregate data by CustomerID
rfm = df_cleaned.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

# Rename columns
rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)

# Calculate RFM scores
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Combine RFM scores into a single RFM_Score
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Display the RFM dataframe
# print(rfm.head())

#-------------> EDA on Cleaned Dataset
# print(df_cleaned.describe())

# Univariate Analysis 

# print(df_cleaned.columns)

# plt.figure()
# sns.histplot(df_cleaned['CustomerID'].dropna(), kde=True)
# plt.title(f'Histogram of {'CustomerID'}')
# plt.xlabel('CustomerID')
# plt.ylabel('Frequency')

# plt.show()

# plt.figure()
# sns.histplot(df_cleaned['InvoiceNo'].dropna(), kde=True)
# plt.title(f'Histogram of {'InvoiceNo'}')
# plt.xlabel('InvoinceNo')
# plt.ylabel('Frequency')

# plt.show()

# Distribution plots for key features (Recency, Frequency, and Monetary)
# Recency (Days since last purchase)
# plt.figure(figsize=(10, 6))
# sns.histplot(df['InvoiceDate'], kde=True, bins=30, color='skyblue')
# plt.title('Distribution of Recency (Days since Last Purchase)')
# plt.xlabel('Recency (Days)')
# plt.ylabel('Frequency')
# plt.show()

# Histogram of invoice dates by day (if datetime formatted)
# df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
# plt.figure()
# df_cleaned['InvoiceDate'].dt.date.value_counts().sort_index().plot(kind='bar')
# plt.title('Invoices per Date')
# plt.xlabel('Date')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# BiVariate Analysis

# df_cleaned['StockCode'] = df_cleaned['StockCode'].astype(str)
# sns.scatterplot(data=df_cleaned, x='StockCode', y='Quantity')
# plt.show()

sns.pairplot(df_cleaned)
plt.show()


#-----------> Clustering

# Prepare data for clustering
rfm_cluster = rfm[['Recency', 'Frequency', 'Monetary']].copy()

# Standardize the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_cluster)


# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# From the Elbow Method, choose the optimal number of clusters (e.g., 4)
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Display the clusters
print(rfm.groupby('Cluster').agg({
    'Recency': ['mean', 'std'],
    'Frequency': ['mean', 'std'],
    'Monetary': ['mean', 'std']
}))

# Display the clusters
print(rfm.groupby('Cluster').agg({
    'Recency': ['mean', 'std'],
    'Frequency': ['mean', 'std'],
    'Monetary': ['mean', 'std']
}))


# ### 6. Interpretation and Action
# Cluster Analysis: Examine the mean and standard deviation of Recency, Frequency, and Monetary values for each cluster to understand customer behavior.

# Marketing Strategies: Develop targeted marketing strategies for each cluster. For example:

# Cluster 0: High frequency, high monetary value – loyal customers.

# Cluster 1: High recency, low frequency – recent but infrequent buyers.

# Cluster 2: Low recency, low frequency – at-risk customers.

# Cluster 3: Low recency, high frequency – frequent but recent customers.