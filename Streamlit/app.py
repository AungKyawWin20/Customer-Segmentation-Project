import streamlit as st
import pandas as pd
import joblib

st.title("Customer Segmentation App")

st.write("""
Upload a CSV file in the same format as your Online Retail dataset.  
The app will clean the data, perform RFM feature engineering, cluster customers using KMeans,  
train a Random Forest classifier, and let you download a CSV with predicted customer segments.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load trained model
rf_classifier = joblib.load("../Models/rf_segmenter.joblib")

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

if uploaded_file:
    with st.spinner("Processing..."):
        data = pd.read_csv(uploaded_file, parse_dates=['InvoiceDate'])
        rfmt = preprocess_data(data)
        X = rfmt[["Recency", "Frequency", "Monetary", "Interpurchase_time", "AvgBasketSize", "AvgOrderValue"]]
        rfmt["Predicted_Label"] = rf_classifier.predict(X)
        output = rfmt.reset_index()[["CustomerID", "Recency", "Frequency", "Monetary", "Interpurchase_time", "AvgBasketSize", "AvgOrderValue", "Predicted_Label"]]
        st.success("Segmentation complete!")
        st.dataframe(output.head(20))
        csv = output.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Segmentation CSV",
            data=csv,
            file_name="customer_segments.csv",
            mime='text/csv'
        )