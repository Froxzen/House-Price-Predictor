import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('home_dataset.csv')

house_sizes = data['HouseSize'].values
house_prices = data['HousePrice'].values

fig, axis = plt.subplots(1, 2, figsize=(12, 6))

# Plotting original data
axis[0].scatter(house_sizes, house_prices, marker = 'o', color = 'blue')
axis[0].set_title('House Prices vs House Size')
axis[0].set_xlabel('House Size (sq.ft)')
axis[0].set_ylabel('House Price ($)')

X_train, X_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Plotting data with prediction
axis[1].scatter(X_test, y_test, marker='o', color='blue', label='Actual Prices')
axis[1].plot(X_test, predictions, color='red', linewidth=2, label='Predicted Prices')
axis[1].set_title('Property Price Prediction with Linear Regression')
axis[1].set_xlabel('House Size (sq.ft)')
axis[1].set_ylabel('House Price (millions $)')
axis[1].legend()

plt.tight_layout()
plt.show()