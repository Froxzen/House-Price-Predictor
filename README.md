# House Price Predictor

This project demonstrates a simple linear regression model to predict house prices based on house sizes. It uses Python libraries such as pandas, matplotlib, and scikit-learn to process data, visualize relationships, and build a predictive model.

## Prerequisites
To run this project, you need to have the following Python libraries installed:
- pandas
- matplotlib
- scikit-learn
  
You can install these libraries using pip:
```bash
pip install pandas matplotlib scikit-learn
```

## Dataset
The project uses a dataset named home_dataset.csv. This file should contain two columns:

- HouseSize: The size of the house in square feet
- HousePrice: The price of the house in dollars

Ensure that this CSV file is in the same directory as the script.

## Script Overview
The main script performs the following tasks:

1. Loads the dataset using pandas
2. Splits the data into training and testing sets
3. Trains a linear regression model on the training data
4. Makes predictions on the test data
5. Calculates the model's accuracy using R-squared score
6. Creates two visualizations:
...- A scatter plot of the original data (house sizes vs. prices)
...- A plot showing actual prices vs. predicted prices from the linear regression model

## Contributing
Contributions are welcome! If you'd like to improve the model, feel free to fork the repository and submit a pull request.
