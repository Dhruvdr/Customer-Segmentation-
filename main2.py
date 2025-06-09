import math
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle

# setting max display columns limmit to 30 
pd.set_option("display.max_columns",20)

import warnings
warnings.filterwarnings('ignore')
from numpy import emath

try :
    #try to load the pickle file
    with open("data.pkl", "rb") as f:
        df= pickle.load(f)
except FileNotFoundError:
    #if it doesnt exist, load CSV and create the pickle 
    df = pd.read_excel("Online Retail.xlsx")
    with open("data.pkl","wb") as f:
        pickle.dump(df,f)
 
# print(df.head) 

#shape of dataset  [541909 rows x 8 columns]>
# print(df.shape)


## -----Data Wrangling

# print(df.info())

### Observations :  1 . Datatype of invoice convert from object to datatime
###                 2 . If Invoice starts with C means its a cancelled -> drop these entries.
###                 3 . Null Values in CustomerID and Descriptions

# Null values Count
# print(df.isnull().sum().sort_values(ascending=False))

# Visulizing null values using heatmap 

# plt.figure(figsize=(15,5))
# sns.heatmap(df.isnull(),cmap='plasma',annot=False,yticklabels=False)
# plt.title("Visualising Missing Values")
# plt.show()

### Observations : 1. Missing Values in CustomerID and Description Columns
###                2. CustomerID is our identification feature so if its missing means other wont help us in analysis 
###                 --- dropping all missing datapoints

df.dropna(inplace=True)
# print(df.shape) #(406829, 8)

# print(df.describe())

### Observatoin : 1. Minimum value for Quanitity Column is Negative
###               2. Unitprice has 0 as min Value

# print(df[df["Quantity"]<0])
### Quanity is Negative for Invoice number starting with C

#Changing the Datatype to str
df['InvoiceNo'] = df["InvoiceNo"].astype('str')
#drop Invoice with C
df = df[~df["InvoiceNo"].str.contains('C')]

#Checking how many values are present for unitprice == 0 ---> 40 values 
# print(len(df[df["UnitPrice"]==0]))
#Taking unitprice values greater than 0 
df = df[df["UnitPrice"]>0]
# print(df.head())
# print(df.describe())
# print(df.shape)     #(397884, 8)

## -------- Feature Engineering 
# Converting InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"],format= "%d-%m-%Y %H:%M")

df["year"] = df["InvoiceDate"].apply(lambda x: x.year)
df["month_num"] = df["InvoiceDate"].apply(lambda x: x.month)
df["day_num"] = df["InvoiceDate"].apply(lambda x: x.day)
df["hour"] = df["InvoiceDate"].apply(lambda x: x.hour)
df["minute"] = df["InvoiceDate"].apply(lambda x: x.minute)

# extracting month from Invoice Date
df['Month'] = df['InvoiceDate'].dt.month_name()
# extracting dat from Invoice date
df['Day'] = df["InvoiceDate"].dt.day_name()

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# print(df.head())

## ------- EDA (Exploratory Data Analysis)

# print(df.columns)
# Index(['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate','UnitPrice', 'CustomerID', 'Country', 'year', 'month_num', 'day_num', 'hour', 'minute', 'Month', 'Day', 'TotalAmount'],

# top_10_products = df['Description'].value_counts().head(10).reset_index().rename(columns={'Description' : 'Product_name', 'count' : 'Count'})
# print(top_10_products)

# Top 10 Product in terms of description
# plt.figure(figsize=(12,6))
# sns.barplot(data= top_10_products, y= "Product_name" , x= "Count")
# plt.title("Top 10 Products")
# plt.xlabel('Number of Sales')
# plt.ylabel('Product Name')
# plt.tight_layout()
# plt.show()

### Observation : 1.White Hanging heart most selling product with 2028 units sold
###               2.Regency Cakestand 3 Tier is second highest selling product with 1723 unit sold

# bottom_10_products = df["Description"].value_counts().reset_index()
# bottom_10_products.columns = ['Product', 'Count']
# bottom_10_products = bottom_10_products.sort_values(by='Count', ascending=True).head(10)
# print(bottom_10_products)

# plt.figure(figsize=(12,6))
# sns.barplot(data= bottom_10_products , x ='Count' ,y= 'Product' )
# plt.title("Bottom 10 Products")
# plt.xlabel("Number of Sales")
# plt.ylabel("Product Name")
# plt.tight_layout()
# plt.show()

# top_10_stockcode = df['StockCode'].value_counts().reset_index().head(10)
# # print(top_10_stockcode)

# plt.figure(figsize=(12,6))
# sns.barplot(data = top_10_stockcode , y= 'StockCode', x= 'count')
# plt.title("Top 10 Products StockCode")
# plt.xlabel("Number of Sales")
# plt.ylabel("StockCode")
# plt.tight_layout()
# plt.show()

### Observation : 1.StockCode 85123A is highest selling product and 22423 is 2nd highest 

# top_10_customers = df['CustomerID'].value_counts().reset_index().head(10).rename(columns={'count':'Product_purchased'})
# print(top_10_customers)

# plt.figure(figsize=(18,6))
# sns.barplot(data = top_10_customers , x= 'CustomerID', y= 'Product_purchased')
# plt.title("Top 10 Customers")
# plt.xlabel("CustomerID")
# plt.ylabel("Products Purchased")
# plt.tight_layout()
# plt.show()

### Observation : 1. 17841 had purchased highest number of products
###               2. 14911 is 2nd highest customer 

# top_5_countries = df['Country'].value_counts().reset_index().head(5).rename(columns={'count':'Customer_count'})
# print(top_5_countries)

# plt.figure(figsize=(15,6))
# sns.barplot(data=top_5_countries ,x= 'Country', y='Customer_count')
# plt.title('Top 5 Countries having highest No. of Customers')
# plt.show()

### Observation  :  1. UK has highest number of Customers
###                 2. Gernamy , France and Ireland have almost equal amount of numbers

# top_5_countries_wleast = df['Country'].value_counts().reset_index().tail(5).rename(columns={'count':'Customer_count'})
# print(top_5_countries_wleast)

# plt.figure(figsize=(15,6))
# sns.barplot(data=top_5_countries_wleast ,x= 'Country', y='Customer_count')
# plt.title('Top 5 Countries having least highest No. of Customers')
# plt.show()

### Observation : 1. There are very less customers from Saudi Arabia
###               2. bahrain is the second country 

# sales_in_month = df["Month"].value_counts().reset_index()
# print(sales_in_month)

# plt.figure(figsize=(20,6))
# sns.barplot(data=sales_in_month , x= 'Month', y='count')
# plt.title('Sales Count in Months')
# plt.xlabel('Months')
# plt.ylabel('Sales Count')
# plt.tight_layout()
# plt.show()

### Observation : 1.Most of the sale happened in November Month 
###               2. February have the least sales 


# sales_on_days = df["Day"].value_counts().reset_index()

# plt.figure(figsize=(20,6))
# sns.barplot(data=sales_on_days , x= 'Day', y='count')
# plt.title('Sales Count on different Days')
# plt.show()

## Observation : 1. Sales on thursday are very high 
##               2. Sales on Friday are very less

# print(df['hour'].unique())  #[ 8  9 10 11 12 13 14 15 16 17  7 18 19 20  6]

def time(time):
    if ( time >= 6 and time <= 11):
        return 'Morning'
    elif ( time >= 12 and time<=17):
        return 'Afternoon'
    else:
        return 'Evening'

df['Day_time_type'] = df['hour'].apply(time)

# sales_timing = df["Day_time_type"].value_counts().reset_index()
# print(sales_timing)

# plt.figure(figsize=(12,6))
# sns.barplot(data= sales_timing , x='Day_time_type',y= 'count')
# plt.title('Sales Count on days')
# plt.show()

### Observation : 1. Most of sales happened in Afternoon
###               2. least happens in evening 

# avg_amount= df.groupby('CustomerID')['TotalAmount'].mean().reset_index().sort_values(by= 'TotalAmount', ascending=False)
# print(avg_amount)

# plt.figure(figsize=(12,6))
# sns.barplot( x = avg_amount['CustomerID'].head(5) ,y= avg_amount['TotalAmount'].head(15)) 
# plt.title("Average amount spend by each Customer")
# plt.show()

### Observation : 1. 77183 is the highest avg spent by Customer 12346
###               2. 56157 is 2nd highest avg spent by Customer 16446


#-----------> Model Building 
#-------> RFM model analysis
#-----> help business to segment their customer base into different homogenous groups
#  so that they can engange with each group with diff tarageted marketing strategies

rfm_df = df.copy()
# print(rfm_df.head())

# Recency = Latest Date - Last Invoice Date ----> how recently a customer made purchase
# Frequency = count of invoice no. of transaction(s) -----> how often they purchase
# Monetory = Amount for each Customer ---------> how much money they spend
import datetime as dt
#set lastest date 2011-12-10 as last invoice date is 2011-12-09
Latest_date =  dt.datetime(2011,12,10)

# Create RFM Modelling scores for each Customer

rfm_df = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (Latest_date - x.max()).days,       #give no. of days since last purchase
                                       'InvoiceNo': 'nunique',      # give count of unique invoices
                                       'TotalAmount': 'sum'})       #total amount a custommer spent

#convert InvoiceDate into int type
rfm_df['InvoiceDate'] = rfm_df['InvoiceDate'].astype(int)

#Rename column names to Recency, Frequency, Monetary 
rfm_df.rename(columns={'InvoiceDate': 'Recency', "InvoiceNo":"Frequency", "TotalAmount":"Monetary"},inplace=True)

# print(rfm_df.head())

#Descriptive stats= Recency
# print(rfm_df.Recency.describe())

# plt.figure(figsize=(12,6))
# sns.distplot(x=rfm_df["Recency"])
# plt.title('Distribution of Recency ')
# plt.show()
# --- Distribution of Recency is right skewed

# Description Stats = Frequency
# print(rfm_df['Frequency'].describe())

# plt.figure(figsize=(12,6))
# sns.distplot(x=rfm_df["Frequency"])
# plt.title('Distribution of Frequency')
# plt.show()
# --- Distribution of Frequency is highly right skewed

# Description Stats = Monetary
# print(rfm_df['Monetary'].describe())

# plt.figure(figsize=(12,6))
# sns.distplot(x=rfm_df["Monetary"])
# plt.title('Distribution of Monetary')
# plt.show()
# --- Distribution of Monetary is highly right skewed

# Split the data into four segment using quantile 
quantile = rfm_df.quantile(q = [0.25,0.50,0.75])
quantile = quantile.to_dict()       # converting to dict makes easier to use
# print(quantile)

#               Function to create R, F and M segments

# lower the recency, good for the company
# higher the frequency or Monetary, good for the company 

def r_score(x):
    if x <= quantile['Recency'][0.25]:
        return 1
    elif x <= quantile['Recency'][0.50]:
        return 2
    elif x <= quantile['Recency'][0.75]:
        return 3
    else:
        return 4
def fm_score(x, col):
    if x <= quantile[col][0.25]:
        return 4
    elif x <= quantile[col][0.50]:
        return 3
    elif x <= quantile[col][0.75]:
        return 2
    else:
        return 1

rfm_df['R_score']= rfm_df['Recency'].apply(r_score)
rfm_df['F_score']= rfm_df['Frequency'].apply(lambda x: fm_score(x,'Frequency'))
rfm_df['M_score']= rfm_df['Monetary'].apply(lambda x: fm_score(x,'Monetary'))
# print(rfm_df.head())

# Column to combine RFM score
rfm_df['RFM_Segment'] =  rfm_df['R_score'].astype(str) +  rfm_df['F_score'].astype(str) + rfm_df['M_score'].astype(str)
rfm_df['RFM_Score'] = rfm_df[['R_score', 'F_score', 'M_score']].sum(axis=1)
# print(rfm_df.head())
# print(rfm_df.info())

# print(rfm_df["RFM_Score"].unique())     #[ 9  3  6  7 11  4 12 10  5  8]

Loyal_level= ['A' , 'B' , 'C' ,'D']

score_cut = pd.qcut(rfm_df['RFM_Score'], q = 4, labels= Loyal_level)
rfm_df['Loyalty'] = score_cut.values
# print(rfm_df.reset_index().head())

# Validate the data for RFM Segment = 111
# print(rfm_df[rfm_df['RFM_Segment'] == '111'].sort_values("Monetary",ascending=False).reset_index().head(10))

# Plot the loyalty Level
# plt.figure(figsize=(12,6))
# sns.countplot(x=rfm_df["Loyalty"])
# plt.title('Loyalty level of Companies')
# plt.show()

# Target People
# print(rfm_df[rfm_df['Loyalty'] == 'A'].sort_values("Monetary",ascending=False).reset_index().head(10))

segmentation_based_RFM = rfm_df[['Recency','Frequency','Monetary','Loyalty']]
segmentation_RFM = segmentation_based_RFM.groupby('Loyalty').agg({
    'Recency': ['mean','min','max'],
    'Frequency':['mean','min','max'],
    'Monetary':['mean','min','max','count']
})
# print(segmentation_RFM)

# Handle negative and zero values so as to handle infinite numbers during log transformation
def handle_neg_n_zero(num):
    if num<=0:
        return 1
    else:
        return num
rfm_df['Recency'] = [handle_neg_n_zero(x) for x in rfm_df.Recency]
rfm_df['Monetary'] = [handle_neg_n_zero(x) for x in rfm_df.Monetary]
rfm_df['Frequency'] = [handle_neg_n_zero(x) for x in rfm_df.Frequency]

# Perform Log Transformation to bring data into normal or near normal distribution 
log_rfm_df = rfm_df[['Recency','Frequency','Monetary']].apply(np.log,axis=1).round(3)
# print(log_rfm_df.head())

# Visualise the Distribution of R,F & M 

# plt.figure(figsize=(12,6))
# sns.distplot(x=log_rfm_df['Recency'])
# plt.title('Distribution of Recency')
# plt.show()
# plt.figure(figsize=(12,6))
# sns.distplot(x=log_rfm_df['Frequency'])
# plt.title('Distribution of Frequency')
# # plt.show()
# plt.figure(figsize=(12,6))
# sns.distplot(x=log_rfm_df['Monetary'])
# plt.title('Distribution of Monetary')
# plt.show()

rfm_df['Recency_log'] = rfm_df['Recency'].apply(math.log).round(3)
rfm_df['Frequency_log'] = rfm_df['Frequency'].apply(math.log).round(3)
rfm_df['Monetary_log'] = rfm_df['Monetary'].apply(math.log).round(3)
# print(rfm_df)

#----               KMeans Clustering 
from sklearn.metrics import silhouette_samples,silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#-------        Recency and Monetary
#Applying Elbow Method on Recency and Monetary
# taking Rec and Mon log in list 
Rec_Mon_feat = ['Recency_log','Monetary_log']
X = rfm_df[Rec_Mon_feat].values

#Standardising the data 
scaler = StandardScaler()
X = scaler.fit_transform(X)

#applying Elbow Method
# wcss = {}
# for k in range(1,15):       # try 1 to 14 clusters
#     km= KMeans(n_clusters= k , init= 'k-means++', max_iter= 1000)
#     km = km.fit(X)
#     wcss[k] = km.inertia_       # inertia_ = total within-cluster variance

# sns.pointplot(x = list(wcss.keys()),y= list(wcss.values()))
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Sum of Square Distances')
# plt.show()

### Observation : The Optimal Value for Cluster came out to be 2.

# Silhouette Score *(Validating Above optimal cluster Value i.e. 2)
# to evaluate the quality of K-Means clustering for different numbers of clusters using the Silhouette Score

# range_n_cluster = list(range(2,16))
# print(range_n_cluster)

# for n_clusters in range_n_cluster:
#     clusterer = KMeans(n_clusters=n_clusters , random_state=1)
#     preds = clusterer.fit_predict(X)
#     centers = clusterer.cluster_centers_

#     score =  silhouette_score(X , preds)
#     print("For n_clusters = {}, silhouette score is {}".format(n_clusters,score))
# for n_clusters = 2 sihoutte score is good 
#(if value is close to 1 , data point are clustered very well to respective clusters and distance of that datapoint is very far from the other cluster)

from matplotlib import cm

# range2_n_cluster = list(range(2,11))

# for n_clusters in range2_n_cluster :
#     # subplot with 1 row and 2 columns
#     fig, (ax1 ,ax2) = plt.subplots(1,2)
#     fig.set_size_inches(16,7)
    
#     # the 1st subplot is the silhouette plot 
#     # the silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1,1]
#     ax1.set_xlim([-0.1,1])
#     # the (n_clusters+1)*10 is for inserting blank space between silhoutte plots of individual clusters, to demarcate them clearly
#     ax1.set_ylim([0, len(X) + n_clusters * 10 ])

#     # Initialise the cluster with n_clusters value and a randon generator seed of 10 for reproducibility 
#     clusterer  = KMeans(n_clusters=n_clusters, random_state=1)
#     cluster_labels = clusterer.fit_predict(X)

#     # The silhoutte_score gives the average value for all the samples
#     # this give a perspective into the density and separation of the formed clusters
#     silhouette_avg  = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average Silhouette_score is: ",silhouette_avg)
    
#     #Compute silhoutte scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)

#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belongings to cluster i, and sort them
#         ith_cluster_sil_values = sample_silhouette_values[cluster_labels == i]
#         ith_cluster_sil_values.sort()

#         size_cluster_i = ith_cluster_sil_values.shape[0]
#         y_upper = y_lower + size_cluster_i

# #Plotting Silhouette values : Shows the silhouette scores for each sample and each cluster.
#         color = cm.nipy_spectral(float(i) / n_clusters)

#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                          0 , ith_cluster_sil_values,
#                          facecolor =color , edgecolor = color ,alpha=0.7)
#         # label the silhoutte plots with their cluster numbers at the middle 
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10 # 10 for the 0 samples

#     ax1.set_title("the silhoutte plot for various clusters.")
#     ax1.set_xlabel("The silhoutte coefficient values")
#     ax1.set_ylabel("Cluster label")

#     ax1.axvline(x=silhouette_avg , color= 'red', linestyle="--")
#     ax1.set_yticks([])      # clear the yaxis labels/ticks
#     ax1.set_xticks([-0.1, 0 , 0.2, 0.4, 0.6, 0.8, 1])

#     # 2nd plot showing the actual clusters formed 
#     colors = cm.nipy_spectral(cluster_labels.astype(float)/n_clusters)
#     ax2.scatter(X[:,0], X[:,1], marker='.', s=30, lw=0 , alpha= 0.7, c = colors , edgecolor ='k')


# #Scatter plot of cluster centers (ax2) : Shows a scatter plot of the cluster centers on the 2D feature space.
#     # labeling the clusters 
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter (centers[:, 0], centers[:, 1], marker='o',
#                  c='white',alpha=1, s=200, edgecolor ='k')
#     for i,c in enumerate(centers):
#         ax2.scatter(c[0],c[1], marker = '$%d$' % i, alpha=1, s=50, edgecolor = 'k')
    
#     ax2.set_title("the visualisation of the clustered data")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
#     plt.suptitle(("Silhoutte analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight = 'bold')
    
#     plt.show()
### Observation : we got silhouette plot for cluster-2 but still few datapoints are on the negative side of the Silhouette Coefficient value

#Giving n_cluster 2 on kmeans model

#applying Kmeans_clustering algorithm 
kmeans_rec_mon = KMeans(n_clusters=2)
kmeans_rec_mon.fit(X)
y_kmeans = kmeans_rec_mon.predict(X)

# Find the clusters for the observation 
rfm_df['Cluster_based_rec_mon'] = kmeans_rec_mon.labels_
# print(rfm_df.head(10))

# centers of the clusters(coordinate)
# centers = kmeans_rec_mon.cluster_centers_
# print(centers)

# # Plotting visualization clusters
# plt.figure(figsize=(15,6)) 
# plt.title('Customer Segmentation based on Recency and Monetary')
# plt.scatter(X[:,0],X[:,1], c = y_kmeans , s=50 , cmap='winter')
# plt.scatter(centers[:,0], centers[:,1], c='red', s=300, alpha=0.8)
# plt.show()

#               Frequency and Monetary 

# applying elbow method on Freq and Mon
# taking Freq and Mon log in list 
Freq_Mon_feat = ['Frequency_log','Monetary_log']
X = rfm_df[Freq_Mon_feat].values

#Standardising the data 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# applying Elbow Method
# wcss = {}
# for k in range(1,15):       # try 1 to 14 clusters
#     km= KMeans(n_clusters= k , init= 'k-means++', max_iter= 1000)
#     km = km.fit(X)
#     wcss[k] = km.inertia_       # inertia_ = total within-cluster variance
# Plot the graph 
# sns.pointplot(x = list(wcss.keys()),y= list(wcss.values()))
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Sum of Square Distances')
# plt.show()

# Observation : Optimal Value for cluster came out to be 2 

# Silhouette Score
# range_n_clusters = list(range(2,16))
# for n_clusters in range_n_clusters:
#     clusterer = KMeans(n_clusters=n_clusters , random_state=1)
#     preds = clusterer.fit_predict(X)
#     centers = clusterer.cluster_centers_

#     score = silhouette_score(X, preds)
#     print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))
# n_cluster = 2 silhouette score is better.

#plotting

# range2_n_cluster = list(range(2,11))
# for n_clusters in range2_n_cluster :
#     # subplot with 1 row and 2 columns
#     fig, (ax1 ,ax2) = plt.subplots(1,2)
#     fig.set_size_inches(16,7)
    
#     # the 1st subplot is the silhouette plot 
#     # the silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1,1]
#     ax1.set_xlim([-0.1,1])
#     # the (n_clusters+1)*10 is for inserting blank space between silhoutte plots of individual clusters, to demarcate them clearly
#     ax1.set_ylim([0, len(X) + n_clusters * 10 ])

#     # Initialise the cluster with n_clusters value and a randon generator seed of 10 for reproducibility 
#     clusterer  = KMeans(n_clusters=n_clusters, random_state=1)
#     cluster_labels = clusterer.fit_predict(X)

#     # The silhoutte_score gives the average value for all the samples
#     # this give a perspective into the density and separation of the formed clusters
#     silhouette_avg  = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average Silhouette_score is: ",silhouette_avg)
    
#     #Compute silhoutte scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)

#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belongings to cluster i, and sort them
#         ith_cluster_sil_values = sample_silhouette_values[cluster_labels == i]
#         ith_cluster_sil_values.sort()

#         size_cluster_i = ith_cluster_sil_values.shape[0]
#         y_upper = y_lower + size_cluster_i

# #Plotting Silhouette values : Shows the silhouette scores for each sample and each cluster.
#         color = cm.nipy_spectral(float(i) / n_clusters)

#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                          0 , ith_cluster_sil_values,
#                          facecolor =color , edgecolor = color ,alpha=0.7)
#         # label the silhoutte plots with their cluster numbers at the middle 
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10 # 10 for the 0 samples

#     ax1.set_title("the silhoutte plot for various clusters.")
#     ax1.set_xlabel("The silhoutte coefficient values")
#     ax1.set_ylabel("Cluster label")

#     ax1.axvline(x=silhouette_avg , color= 'red', linestyle="--")
#     ax1.set_yticks([])      # clear the yaxis labels/ticks
#     ax1.set_xticks([-0.1, 0 , 0.2, 0.4, 0.6, 0.8, 1])

#     # 2nd plot showing the actual clusters formed 
#     colors = cm.nipy_spectral(cluster_labels.astype(float)/n_clusters)
#     ax2.scatter(X[:,0], X[:,1], marker='.', s=30, lw=0 , alpha= 0.7, c = colors , edgecolor ='k')

# #Scatter plot of cluster centers (ax2) : Shows a scatter plot of the cluster centers on the 2D feature space.
#     # labeling the clusters 
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter (centers[:, 0], centers[:, 1], marker='o',
#                  c='white',alpha=1, s=200, edgecolor ='k')
#     for i,c in enumerate(centers):
#         ax2.scatter(c[0],c[1], marker = '$%d$' % i, alpha=1, s=50, edgecolor = 'k')
    
#     ax2.set_title("the visualisation of the clustered data")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
#     plt.suptitle(("Silhoutte analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight = 'bold')
    
#     plt.show()

#Giving n_cluster 2 on kmeans model

#applying Kmeans_clustering algorithm 
kmeans_freq_mon = KMeans(n_clusters=2)
kmeans_freq_mon.fit(X)
y_kmeans = kmeans_freq_mon.predict(X)

# Find the clusters for the observation 
rfm_df['Cluster_based_freq_mon'] = kmeans_freq_mon.labels_
# print(rfm_df.head(10))

# centers of the clusters(coordinate)
centers = kmeans_freq_mon.cluster_centers_
# print(centers)

# # Plotting visualization clusters
# plt.figure(figsize=(15,6)) 
# plt.title('Customer Segmentation based on Frequency and Monetary')
# plt.scatter(X[:,0],X[:,1], c = y_kmeans , s=50 , cmap='winter')
# plt.scatter(centers[:,0], centers[:,1], c='red', s=300, alpha=0.8)
# plt.show()

#               Recency , Frequency and Monetary
rec_freq_mon_feat = ['Recency_log','Frequency_log','Monetary_log']
X = rfm_df[rec_freq_mon_feat].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# applying Elbow Method
# wcss = {}
# for k in range(1,15):       # try 1 to 14 clusters
#     km= KMeans(n_clusters= k , init= 'k-means++', max_iter= 1000)
#     km = km.fit(X)
#     wcss[k] = km.inertia_       # inertia_ = total within-cluster variance

# # Plot the graph 
# sns.pointplot(x = list(wcss.keys()),y= list(wcss.values()))
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Sum of Square Distances')
# plt.show()


# Observation : Optimal Value for cluster came out to be 2 

# Silhouette Score
# range_n_clusters = list(range(2,16))
# for n_clusters in range_n_clusters:
#     clusterer = KMeans(n_clusters=n_clusters , random_state=1)
#     preds = clusterer.fit_predict(X)
#     centers = clusterer.cluster_centers_

#     score = silhouette_score(X, preds)
#     print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))
# n_cluster = 2 silhouette score is better.

# plotting

# range2_n_cluster = list(range(2,11))
# for n_clusters in range2_n_cluster :
#     # subplot with 1 row and 2 columns
#     fig, (ax1 ,ax2) = plt.subplots(1,2)
#     fig.set_size_inches(16,7)
    
#     # the 1st subplot is the silhouette plot 
#     # the silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1,1]
#     ax1.set_xlim([-0.1,1])
#     # the (n_clusters+1)*10 is for inserting blank space between silhoutte plots of individual clusters, to demarcate them clearly
#     ax1.set_ylim([0, len(X) + n_clusters * 10 ])

#     # Initialise the cluster with n_clusters value and a randon generator seed of 10 for reproducibility 
#     clusterer  = KMeans(n_clusters=n_clusters, random_state=1)
#     cluster_labels = clusterer.fit_predict(X)

#     # The silhoutte_score gives the average value for all the samples
#     # this give a perspective into the density and separation of the formed clusters
#     silhouette_avg  = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average Silhouette_score is: ",silhouette_avg)
    
#     #Compute silhoutte scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)

#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belongings to cluster i, and sort them
#         ith_cluster_sil_values = sample_silhouette_values[cluster_labels == i]
#         ith_cluster_sil_values.sort()

#         size_cluster_i = ith_cluster_sil_values.shape[0]
#         y_upper = y_lower + size_cluster_i

# #Plotting Silhouette values : Shows the silhouette scores for each sample and each cluster.
#         color = cm.nipy_spectral(float(i) / n_clusters)

#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                          0 , ith_cluster_sil_values,
#                          facecolor =color , edgecolor = color ,alpha=0.7)
#         # label the silhoutte plots with their cluster numbers at the middle 
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10 # 10 for the 0 samples

#     ax1.set_title("the silhoutte plot for various clusters.")
#     ax1.set_xlabel("The silhoutte coefficient values")
#     ax1.set_ylabel("Cluster label")

#     ax1.axvline(x=silhouette_avg , color= 'red', linestyle="--")
#     ax1.set_yticks([])      # clear the yaxis labels/ticks
#     ax1.set_xticks([-0.1, 0 , 0.2, 0.4, 0.6, 0.8, 1])

#     # 2nd plot showing the actual clusters formed 
#     colors = cm.nipy_spectral(cluster_labels.astype(float)/n_clusters)
#     ax2.scatter(X[:,0], X[:,1], marker='.', s=30, lw=0 , alpha= 0.7, c = colors , edgecolor ='k')

# #Scatter plot of cluster centers (ax2) : Shows a scatter plot of the cluster centers on the 2D feature space.
#     # labeling the clusters 
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter (centers[:, 0], centers[:, 1], marker='o',
#                  c='white',alpha=1, s=200, edgecolor ='k')
#     for i,c in enumerate(centers):
#         ax2.scatter(c[0],c[1], marker = '$%d$' % i, alpha=1, s=50, edgecolor = 'k')
    
#     ax2.set_title("the visualisation of the clustered data")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
#     plt.suptitle(("Silhoutte analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight = 'bold')
    
#     plt.show()

# n_clusters =2  on KMeans Model

#applying Kmeans_clustering algorithm 
kmeans_rec_freq_mon = KMeans(n_clusters=2)
kmeans_rec_freq_mon.fit(X)
y_kmeans = kmeans_rec_freq_mon.predict(X)

# Find the clusters for the observation 
rfm_df['Cluster_based_rec_freq_mon'] = kmeans_rec_freq_mon.labels_
# print(rfm_df.head(10))

# centers of the clusters(coordinate)
centers = kmeans_rec_freq_mon.cluster_centers_
# print(centers)

# # Plotting visualization clusters
# plt.figure(figsize=(15,6)) 
# plt.title('Customer Segmentation based on Recency, Frequency and Monetary')
# plt.scatter(X[:,0],X[:,1], c = y_kmeans , s=50 , cmap='winter')
# plt.scatter(centers[:,0], centers[:,1], c='red', s=300, alpha=0.8)
# plt.show()

#Conclusion OF Project
# print(segmentation_RFM)
#   1. Clustering based on RFM analysis. We had 4 clusters/segments of customers
#       A Customers: 1188(less recency but high freq and heacy spendings)
#       B Customers: 1266 (good recency,frequency and monetary)
#       C Customers: 947 (high recency, low frequency and low spending)
#       D Customers: 937 ( very high recency but very less frequency and spendings)

# Clustering_based_RFM = rfm_df[['Recency','Frequency','Monetary','Cluster_based_rec_freq_mon']]
# Clustering_RFM = Clustering_based_RFM.groupby('Cluster_based_rec_freq_mon').agg({
#     'Recency': ['mean','min','max'],
#     'Frequency':['mean','min','max'],
#     'Monetary':['mean','min','max','count']
# })
# print(Clustering_RFM)
#   2. Implementation of M.L. Algorithm to cluster the customers 
#           giving optimal no. of clusters = 2
#       Cluster 0 : high recency rate but low freq and monetary. contains 2585 customers
#       Cluster 1 : low recency rate but high freq and monetary. conatins 1753 customers
