# Customer Segmentation Project

This project applies RFM (Recency, Frequency, Monetary) analysis and KMeans clustering to segment customers based on their purchasing behavior from an online retail dataset.

## 📊 Techniques Used
- Data Cleaning & Preprocessing using Numpy and Pandas
- RFM Analysis
- KMeans Clustering (scikit-learn)
- Elbow Method for Optimal Clusters
- Visualization with Seaborn & Matplotlib

## 📁 Project Structure

├── Customer-Segmentation/ # Dataset location
├── main.py/ # Python scripts
├── output/ # Generated visualizations

## 📌 Data Source
UCI Machine Learning Repository - Online Retail Dataset

## 📈 Results
Cluster summary statistics provided in the terminal.
1. Clustering based on RFM analysis. We had 4 clusters/segments of customers
        A Customers: 1188(less recency but high freq and heacy spendings)
        B Customers: 1266 (good recency,frequency and monetary)
        C Customers: 947 (high recency, low frequency and low spending)
        D Customers: 937 ( very high recency but very less frequency and spendings)

2. Implementation of M.L. Algorithm to cluster the customers 
        giving optimal no. of clusters = 2
        Cluster 0 : high recency rate but low freq and monetary. contains 2585 customers
        Cluster 1 : low recency rate but high freq and monetary. conatins 1753 customers

Visualization Plots are saved in the Output directory.
