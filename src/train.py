# train_model.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your historical data
data = pd.read_csv("../Datasets/online_retail_II.csv", parse_dates=['InvoiceDate'])

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return up_limit, low_limit

def replace_with_threshold(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def preprocess_data(data):
    # Rename columns if needed
    data.rename(columns={
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID"
    }, inplace=True)
    # Drop missing CustomerID and duplicates
    data.dropna(subset=["CustomerID"], inplace=True)
    data = data.drop_duplicates(keep='first')
    # Remove outliers
    replace_with_threshold(data, "Quantity")
    replace_with_threshold(data, "UnitPrice")
    # Remove cancelled transactions
    data = data[~data["InvoiceNo"].astype(str).str.contains('C', na=False)]
    # Feature engineering
    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    latest_date = data["InvoiceDate"].max()
    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': lambda x: x.nunique(),
        'Quantity': lambda x: x.sum(),
        "Revenue": lambda x: x.sum()
    })
    rfm.rename(columns={'InvoiceDate': 'Recency', 
                       'InvoiceNo': 'Frequency',
                       'Quantity': 'TotalQuantity', 
                       'Revenue': 'Monetary'}, inplace=True)
    rfm['AvgBasketSize'] = rfm['TotalQuantity'] / rfm['Frequency']
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    # Interpurchase time
    cycle = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: ((x.max() - x.min()).days)
    })
    rfm["Shopping_Cycle"] = cycle
    rfm["Interpurchase_time"] = rfm["Shopping_Cycle"] // rfm["Frequency"]
    rfmt = rfm[["Recency", "Frequency", "Monetary", "Interpurchase_time", "AvgBasketSize", "AvgOrderValue"]].copy()
    rfmt = rfmt.fillna(0)
    return rfmt

rfmt = preprocess_data(data)

# KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
rfmt['Cluster'] = kmeans.fit_predict(rfmt)

# Train Random Forest
X = rfmt[["Recency", "Frequency", "Monetary", "Interpurchase_time", "AvgBasketSize", "AvgOrderValue"]]
y = rfmt["Cluster"]
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Save models
joblib.dump(rf, "../Models/rf_segmenter.joblib")
joblib.dump(kmeans, "../Models/kmeans_model.joblib")