# Customer Segmentation using RFM Analysis & KMeans Clustering

## 📌 Project Overview
This project performs **customer segmentation** using **RFM (Recency, Frequency, Monetary) analysis** and **KMeans clustering** on an online retail dataset.  
The goal is to identify distinct customer groups based on purchasing behavior and derive actionable business insights.

---

## 📊 Dataset
- **Source**: UCI Machine Learning Repository  
- **Name**: Online Retail Dataset  
- **Description**: Transactional data from an online retail store including invoice details, product information, customer IDs, and purchase amounts.

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
- Identified top-selling products and customers  
- Analyzed sales trends by country, month, weekday, and time of day  
- Observed strong seasonality and customer concentration patterns

---

### 3. RFM Analysis
- **Recency**: Days since last purchase  
- **Frequency**: Number of unique transactions  
- **Monetary**: Total amount spent  

Customers were scored using quantiles and grouped into **four loyalty segments (A–D)**.

#### RFM Segments:
- **A**: Low recency, high frequency & high spending  
- **B**: Good recency, frequency & monetary value  
- **C**: High recency, low frequency & low spending  
- **D**: Very high recency with minimal engagement  

---

### 4. KMeans Clustering
- Applied **log transformation** to handle skewed distributions  
- Standardized features using `StandardScaler`  
- Evaluated multiple feature combinations:
  - Recency & Monetary  
  - Frequency & Monetary  
  - Recency, Frequency & Monetary  

#### Model Selection:
- Used the **Elbow Method (WCSS)** and **Silhouette Score**
- Optimal number of clusters: **2**

---

## 📈 Results

### RFM-Based Segmentation
- **4 customer segments** with clearly differentiated purchasing behavior

### KMeans Clustering
- **Cluster 0**: High recency, low frequency & low spending (~59%)  
- **Cluster 1**: Low recency, high frequency & high spending (~41%)

These clusters represent **low-value vs high-value customers**, enabling targeted business strategies.

---

## 📊 Visualizations
- Missing value heatmaps  
- Sales trends (time, country, product)  
- RFM distributions (before & after log transformation)  
- Elbow Method & Silhouette analysis  
- Final KMeans cluster scatter plots  

All plots are saved in the `output/` directory.

---

## 🛠️ Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 📁 Project Structure
<pre>
├── Customer-Segmentation/ # Dataset location
├── main.py/ # Python scripts
├── output/ # Generated visualizations
</pre>


## 🚀 Future Improvements
- Try advanced clustering techniques (DBSCAN, Hierarchical Clustering)
- Include customer lifetime value (CLV)
- Build a dashboard for interactive exploration

---

## 📬 Contact
If you have suggestions or feedback, feel free to connect on LinkedIn!
