# Customer Segmentation & Predictive Lifetime Value (CLV) Analysis

## 📌 Project Overview
This project performs end-to-end customer analytics on an online retail dataset. It begins with descriptive customer segmentation using RFM (Recency, Frequency, Monetary) analysis and KMeans clustering, and elevates to predictive analytics using Probabilistic Machine Learning (BG/NBD and Gamma-Gamma models) to forecast Customer Lifetime Value (CLV) and predict churn.

The goal is to identify distinct customer groups, predict their future purchasing behavior, and derive highly actionable business and marketing strategies.

---

## 📊 Dataset
- **Source**: UCI Machine Learning Repository  
- **Name**: Online Retail Dataset  
- **Description**: Transactional data from a UK-based online retail store including invoice details, product descriptions, customer IDs, and purchase amounts.

---

## 🔍 Methodology

### 1. Data Cleaning & Preprocessing
- Removed cancelled invoices and invalid transactions  
- Dropped rows with missing customer identifiers  
- Filtered negative and zero price values  
- Converted date columns to datetime format  
- Engineered additional time-based features

---

### 2. Exploratory Data Analysis (EDA)
- Identified top-selling products and highest-value customers
- Analyzed sales trends by country, month, weekday, and time of day  
- Observed strong seasonality (peaks in November) and customer concentration patterns

---

### 3. RFM Analysis
- **Recency**: Days since last purchase  
- **Frequency**: Number of unique transactions  
- **Monetary**: Total amount spent  

Customers were scored using quantiles and grouped into **four loyalty segments (A–D)**.
- **A**: Low recency, high frequency & high spending  
- **B**: Good recency, frequency & monetary value  
- **C**: High recency, low frequency & low spending  
- **D**: Very high recency with minimal engagement  

---

### 4. KMeans Clustering (Unsupervised ML)
- Applied **log transformation** to handle skewed distributions  
- Standardized features using `StandardScaler`  
- Evaluated multiple feature combinations using the **Elbow Method (WCSS)** and **Silhouette Score**.
- Optimal number of clusters: 2

### 5. Predictive CLV & Churn Modeling (Probabilistic ML)
- **Beta-Geometric/Negative Binomial Distribution (BG/NBD)**: Fitted a model to predict the expected number of purchases in the next 30 days and calculate the probability that a customer is still "alive" (churn risk).
- **Gamma-Gamma Submodel**: Modeled the distribution of average transaction values to predict future spend.
- **12-Month CLV**: Combined both models to project the exact monetary value each customer will bring to the business over the next year.

---

## 📈 Results & Business Insights 

### RFM-Based Segmentation
- **4 customer segments** with clearly differentiated purchasing behavior

### KMeans Clustering
- Cluster 0: High recency, low frequency & low spending (~59% of the base).
Strategy: Re-engagement and automated win-back campaigns.
- Cluster 1: Low recency, high frequency & high spending (~41% of the base). 
Strategy: VIP treatment and lookalike audience acquisition.

These clusters represent **low-value vs high-value customers**, enabling targeted business strategies.

### Predictive Analytics & Churn Risk
- Successfully projected the **12-Month Expected Revenue** and **30-Day Purchase Pacing** for all returning customers, allowing the business to shift from reactive to proactive marketing.
- **Actionable Insight**: The model flagged exactly **1 "At-Risk Whale"** (A Tier 'A' loyalty customer whose probability of returning dropped below 50%). This allows the business to immediately deploy a highly targeted retention campaign to save a high-value account.

---

## 📊 Visualizations
- Missing value heatmaps & sales trends
- RFM distributions (before & after log transformation)
- Elbow Method & Silhouette analysis for K-Means validation
- Final KMeans cluster scatter plots
- Predicted 12-Month CLV distribution by customer tier

All plots are saved in the `output/` directory.

---

## 🛠️ Tools & Technologies
- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (KMeans, StandardScaler)
- **Predictive CLV**: Lifetimes (BetaGeoFitter, GammaGammaFitter)
- **Data Visualization**: Matplotlib, Seaborn

---

## 📁 Project Structure
<pre>
├── online-retail.xlsx/   # Dataset 
├── main2.py/             # Main Execution Script 
├── output/               # Generated visualizations
└── README.md             # Project documentation
</pre>

---

## 📬 Contact
If you have suggestions or feedback, feel free to connect on LinkedIn!
