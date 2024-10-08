import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import pandas_ta as ta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,f1_score
# 1. Download data
BTC_data = yf.download("BTC-USD")
# ETH_data = yf.download("ETH-USD")

# 2. Calculate price change
BTC_data["Price-Change"] = (BTC_data["Close"] - BTC_data["Open"])

# 3. Add ETH close price to BTC data
# BTC_data["ETH"] = (ETH_data["Close"])

BTC_data["DAYS"] = BTC_data.index.dayofweek

# 4. Calculate SMA (Simple Moving Average)
BTC_data["SMA_10"] = (ta.sma(BTC_data["Close"], length=10))
BTC_data["RSI"] = ta.rsi(BTC_data["Close"], length=14)

BTC_data["Candle-Change"] = (BTC_data["Price-Change"] > 0).astype(int)

# 5. Drop NaN values
BTC_data = BTC_data.dropna()

# 6. Prepare features and target
X = BTC_data[["Open", "Close","DAYS", "Price-Change", "Candle-Change","High", "Volume" ,"RSI" ,"SMA_10"]]
y = (BTC_data["Candle-Change"]).shift(-1)
print(BTC_data["DAYS"])
# 7. Scale the data


# حذف NaN‌ها از y
y = y.dropna()

# هم‌زمان X را نیز با طول جدید y هماهنگ کنید
X = X.iloc[:-1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=["Open", "Close", "DAYS","Price-Change" , "Candle-Change","High", "Volume","RSI", "SMA_10"])

# 8. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=False)

# 9. Define models
models = [
    DecisionTreeClassifier(max_depth=70, min_samples_split=10, min_samples_leaf=16,random_state=0),
    LogisticRegression(C=10, solver='lbfgs', max_iter=3000,random_state=0),
    SVC(C=5, kernel='rbf', gamma=0.01, probability=True, random_state=0),
    # GradientBoostingClassifier(n_estimators=300, learning_rate=0.004, max_depth=64, random_state=42),
]

# 10. Initialize weights for boosting
weights = np.ones(len(y_train))

# 11. Manual boosting loop - بدون اضافه کردن ستون‌های اضافی
for model in models:
    model.fit(X_train, y_train, sample_weight=weights)

    # Predictions for test data
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"مدل {model.__class__.__name__} دقت: {accuracy:.2f}")
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f1)
    # Predictions for training data
    y_train_pred = model.predict(X_train)

    # Update weights
    errors = (y_train != y_train_pred).astype(int)
    weights += errors * (1 / (accuracy + 1e-10))

# 12. Define and train the neural network
mlp_model = MLPClassifier(hidden_layer_sizes=(32, 32), activation='relu', solver='adam', max_iter=2500, random_state=0,shuffle=False)
mlp_model.fit(X_train, y_train)
last_pred = mlp_model.predict(X_test)
last_score = accuracy_score(y_test, last_pred)
print(f'دقت مدل نهایی با شبکه عصبی: {last_score:.2f}')
# x = np.array([[26000, 26300, 26500, 30000000000, 2200,2000, 2122]])  # Make it 2D
# x_scaled = scaler.transform(x)  # Scale the input data
# print(mlp_model.predict(x_scaled))  # Predict using the scaled input

# 13. Create a DataFrame for predictions
preds_df = pd.DataFrame(index=BTC_data.index[-len(last_pred):])
preds_df['Predicted'] = last_pred

# 14. Add color to predictions
preds_df['Color'] = np.where(preds_df['Predicted'] == 0, 'red', 'green')

#15. Plot with Seaborn
plt.figure(figsize=(7, 10))
sns.lineplot(data=BTC_data, x=BTC_data.index, y="Candle-Change", label=" Candle-Change")
sns.lineplot(data=BTC_data, x=BTC_data.index, y="Price-Change", label="Price-Change")
# Adding colored lines based on predictions
for index, row in preds_df.iterrows():
    plt.axvline(x=index, color=row['Color'], alpha=0.3)

# Plot settings
plt.title('Bitcoin Price Change and Predictions Over Time')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
