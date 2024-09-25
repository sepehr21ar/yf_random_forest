import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
BTC = yf.download("BTC-USD")

BTC.dropna(inplace=True)
imputer = SimpleImputer( strategy='mean')
BTC["body"] = BTC["Close"] - BTC["Open"]
BTC['SMA'] = ta.sma(BTC['Close'], length=14)
X = BTC[["Close", "Open", "High", "Low","SMA"]].values
X = imputer.fit_transform(X)
y = BTC["body"]

kmeans_model = KMeans(n_clusters=2)
kmeans_model.fit(X)

lables = kmeans_model.predict(X)

BTC["Cluster"] = lables

X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.2, random_state=0)

rf_model = RandomForestRegressor()
rf_model.fit(X_Train, y_Train)

y_pred = rf_model.predict(X_Test)
rmse = root_mean_squared_error(y_Test,y_pred)
print(f"rmse in model is : {rmse}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(range(len(X)), BTC["body"], s=10, c="blue")
plt.title("Scatter Plot - Without Clustering")

plt.subplot(1, 2, 2)
plt.scatter(range(len(X)), BTC["body"], s=10, c=lables)
plt.title("Scatter Plot - With KMeans Clustering")

plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_Test)), y_Test, label="Actual", color="blue", s=10)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", color="red", s=10)
plt.legend()
plt.title("Actual vs Predicted - Random Forest Regression")
plt.show()