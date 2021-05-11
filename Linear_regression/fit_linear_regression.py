"""
Main script witch create dataset, train logistic regression and visualise whole process.
"""
import numpy as np

from Linear_regression.create_linear_rand_dataset import create_dataset
from Linear_regression.LinearRegression import LinearRegression

if __name__ == '__main__':
    # Create (x, y) training pairs.
    train_x, train_y = create_dataset()

    # Create object with Logistic regression parameters and train/ visualise  functions.
    linear_regression = LinearRegression()

    # Train Linear regression on created dataset.
    linear_regression.optimize(train_x.T, train_y.T, print_cost=True, num_iterations=200, learning_rate=0.5)

    # Plot end results.
    linear_regression.plot_prediction(train_x, train_y)