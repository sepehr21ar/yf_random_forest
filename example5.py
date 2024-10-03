# import yfinance as yf
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.neural_network import MLPClassifier
# import pandas as pd
# import pandas_ta as ta
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 1. Download data
# BTC_data = yf.download("BTC-USD")
# ETH_data = yf.download("ETH-USD")
#
# # 2. Calculate price change
# BTC_data["Price-Change"] = (BTC_data["Close"] - BTC_data["Open"]).shift(-1)
#
# # 3. Add ETH close price to BTC data
# BTC_data["ETH"] = (ETH_data["Close"])
#
# # 4. Calculate SMA (Simple Moving Average)
# BTC_data["SMA_10"] = (ta.sma(BTC_data["Close"], length=10))
# BTC_data["RSI"] = ta.rsi(BTC_data["Close"], length=14)
# BTC_data["High"] = BTC_data["High"]
# BTC_data["Volume"] = BTC_data["Volume"]
# BTC_data["Close"] = BTC_data["Close"]
# BTC_data["Open"] = BTC_data["Open"]
#
# # 5. Drop NaN values
# BTC_data = BTC_data.dropna()
#
# # 6. Prepare features and target
# X = BTC_data[["Open", "Close", "High", "Volume", "ETH","RSI" ,"SMA_10"]]
# y = (BTC_data["Price-Change"] > 0).astype(int)
#
# # 7. Scale the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X = pd.DataFrame(X_scaled, columns=["Open", "Close", "High", "Volume", "ETH","RSI", "SMA_10"])
#
# # 8. Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
#
# # 9. Define models
# models = [
#     DecisionTreeClassifier(max_depth=70, min_samples_split=10, min_samples_leaf=16, random_state=42),
#     LogisticRegression(C=10, solver='lbfgs', max_iter=3000, random_state=42),
#     SVC(C=5, kernel='rbf', gamma=0.01, probability=True, random_state=42),
#     GradientBoostingClassifier(n_estimators=300, learning_rate=0.004, max_depth=64, random_state=42),
# ]
#
# # 10. Initialize weights for boosting
# weights = np.ones(len(y_train))
#
# # 11. Manual boosting loop - بدون اضافه کردن ستون‌های اضافی
# for model in models:
#     model.fit(X_train, y_train, sample_weight=weights)
#
#     # Predictions for test data
#     y_pred = model.predict(X_test)
#
#     # Evaluate accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"مدل {model.__class__.__name__} دقت: {accuracy:.2f}")
#
#     # Predictions for training data
#     y_train_pred = model.predict(X_train)
#
#     # Update weights
#     errors = (y_train != y_train_pred).astype(int)
#     weights += errors * (1 / (accuracy + 1e-10))
#
# # 12. Define and train the neural network
# mlp_model = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=2500, random_state=42)
# mlp_model.fit(X_train, y_train)
# last_pred = mlp_model.predict(X_test)
# last_score = accuracy_score(y_test, last_pred)
# print(f'دقت مدل نهایی با شبکه عصبی: {last_score:.2f}')
# x = np.array([[26000, 26300, 26500, 30000000000, 2200,2000, 2122]])  # Make it 2D
# x_scaled = scaler.transform(x)  # Scale the input data
# print(mlp_model.predict(x_scaled))  # Predict using the scaled input
#
# # 13. Create a DataFrame for predictions
# preds_df = pd.DataFrame(index=BTC_data.index[-len(last_pred):])
# preds_df['Predicted'] = last_pred
#
# # 14. Add color to predictions
# preds_df['Color'] = np.where(preds_df['Predicted'] == 0, 'red', 'green')

# 15. Plot with Seaborn
# plt.figure(figsize=(7, 10))
# sns.lineplot(data=BTC_data, x=BTC_data.index, y="Price-Change", label="Price Change", color='blue')
#
# # Adding colored lines based on predictions
# for index, row in preds_df.iterrows():
#     plt.axvline(x=index, color=row['Color'], alpha=0.3)
#
# # Plot settings
# plt.title('Bitcoin Price Change and Predictions Over Time')
# plt.xlabel('Date')
# plt.ylabel('Price Change')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
"------------------------------"

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta

# Step 1: Fetch Bitcoin historical data
def get_bitcoin_data():
    btc_data = yf.download('BTC-USD', start='2020-01-01', end='2023-09-30')
    btc_data['PriceChange'] = (btc_data['Close'] - btc_data['Open']).shift(-1)
    btc_data['PriceDirection'] = np.where(btc_data['PriceChange'] > 0, 1, 0)  # 1: +, 0: -
    btc_data.dropna(inplace=True)
    return btc_data

btc_data = get_bitcoin_data()

# Step 2: Feature Engineering
# Add Relative Strength Index (RSI)
btc_data['RSI'] = ta.rsi(btc_data['Close'], timeperiod=14)

# Add Moving Average Convergence Divergence (MACD)


# Add Simple and Exponential Moving Averages
btc_data['SMA5'] = btc_data['Close'].rolling(window=5).mean()  # 5-day SMA
btc_data['EMA12'] = ta.ema(btc_data['Close'], timeperiod=12)  # 12-day EMA

# Add lagged price returns
btc_data['Lag1'] = btc_data['Close'].shift(1)
btc_data['Lag2'] = btc_data['Close'].shift(2)
btc_data['Lag3'] = btc_data['Close'].shift(3)

# Add volatility as a feature
btc_data['Volatility'] = btc_data['Close'].rolling(window=10).std()

# Drop any NaN values after adding features
btc_data.dropna(inplace=True)

# Step 3: Define features (X) and target (y)
features = ['Close', 'RSI', 'SMA5', 'EMA12', 'Lag1', 'Lag2', 'Lag3', 'Volatility']
X = btc_data[features].values
y = btc_data['PriceDirection'].values

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = MLPClassifier(hidden_layer_sizes=(64,64,64),random_state=42)


# Use GridSearchCV for hyperparameter tuning

model.fit(X_train_scaled, y_train)



# Step 7: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model accuracy: {accuracy * 100:.2f}%")

# Predict the price direction for the next day
next_day_features = X_test_scaled[-1].reshape(1, -1)
prediction = model.predict(next_day_features)
direction = "+" if prediction[0] == 1 else "-"
print(f"Prediction for the next day: {direction}")
