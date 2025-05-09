# 🛍️ Customer Segmentation Project

A modern machine learning pipeline for segmenting retail customers using RFM analysis, KMeans clustering, and Random Forest classification. Includes an interactive Streamlit web app for real-time customer segmentation.

---

## 🚀 Overview

This project empowers retail businesses to understand and segment their customers based on purchasing behavior. By leveraging RFM (Recency, Frequency, Monetary) features and machine learning, you can:

- **Analyze** customer value and engagement
- **Segment** customers for targeted marketing
- **Predict** customer segments for new data via a user-friendly web interface

---

## 📁 Project Structure

```
Customer Segmentation Project/
│
├── Datasets/
│   └── online_retail_II.csv
├── Models/
│   ├── kmeans_model.joblib
│   └── rf_segmenter.joblib
├── Notebooks/
│   └── project.ipynb
├── src/
│   └── train.py
├── Streamlit/
│   └── app.py
└── README.md
```

---

## ✨ Features

- **RFM Feature Engineering**: Recency, Frequency, Monetary, plus advanced features like average basket size, average order value and interpurchase time.
- **KMeans Clustering**: Unsupervised segmentation of historical customers.
- **Random Forest Classification**: Predicts customer segments for new data.
- **Streamlit Web App**: Upload your CSV, get instant segmentation, and download results.
- **Reusable Models**: Trained models are saved and used for future predictions.

---

## ⚡ Quickstart

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Train the Models (One-Time)

```bash
python src/train.py
```
This will create `kmeans_model.joblib` and `rf_segmenter.joblib` in the `Models/` folder.

### 3. Launch the Streamlit App

```bash
cd Streamlit
streamlit run app.py
```

---

## 🖥️ Using the Web App

1. **Upload** a CSV file with columns:
   - `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`
2. **Process**: The app cleans, engineers features, and predicts customer segments using the trained Random Forest model.
3. **Download**: Get a CSV with each customer's predicted segment.

---

## 🧠 Model Details

- **Clustering**: KMeans (n=5, configurable)
- **Classification**: Random Forest (100 estimators)
- **Features Used**:
  - Recency
  - Frequency
  - Monetary
  - Interpurchase Time
  - Average Basket Size
  - Average Order Value

---

## 📊 Example Output

| CustomerID | Recency | Frequency | Monetary | Interpurchase_time | AvgBasketSize | AvgOrderValue | Predicted_Label      |
|------------|---------|-----------|----------|-------------------|---------------|---------------|----------------------|
| 12345      | 12      | 5         | 500.0    | 30                | 10.0          | 100.0         | Top VIP Customers    |
| 67890      | 60      | 2         | 80.0     | 45                | 5.0           | 40.0          | Occasional Shoppers  |

---

## 📚 Acknowledgments

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- Inspired by best practices in retail analytics and customer segmentation.

---

## 📬 Contact

**Author:** Aung Kyaw Win 

**GitHub:** [yourusername](https://github.com/AungKyawWin20)

---

_Designed with ❤️ for modern data-driven retail teams._
 
