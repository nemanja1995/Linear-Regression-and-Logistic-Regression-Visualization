"""
Logistic regression class. It contain parameters, functions for training, visualising and
initializing Logistic regression.
X - data
Y - y_true
Y^ - y_prediction
m - num_data
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, parameter_dimension=1):
        self.w = None
        self.b = None
        self._initialize_parameters(parameter_dimension)

    def _initialize_parameters(self, dim):
        """
        Initialize Linear regression parameters. Set W parameter to have dim dimensions
        :param dim: How much dimensions should W parameter have.
        :return: W, b parameters
        """
        # Initialize W to random values.
        self.w = np.random.rand(dim, 1)
        # Initialize b to zero.
        self.b = 0
        return self.w, self.b

    @staticmethod
    def gradient_dw(data, y_prediction, y_true):
        """ Return gradient value for W parameters. It uses data, predicted y, and true y values """
        num_data = data.shape[1]
        dw = (1 / num_data) * np.dot(data, (y_prediction - y_true).T)
        return dw

    @staticmethod
    def gradient_db(y_prediction, y_true):
        """ Return gradient value for b parameter. It uses predicted y, and true y values."""
        num_data = y_true.shape[1]
        db = (1 / num_data) * np.sum(y_prediction - y_true)
        return db

    @staticmethod
    def cost_function(y_prediction, y_true):
        """ Return Linear regression cost for predicted and true y values."""
        num_data = y_true.shape[1]
        cost = (1/ (2 * num_data)) * np.sum(np.square(y_prediction-y_true))

        return cost

    @staticmethod
    def propagate(w, b, data, y_true):
        """Do forward and backward propagation for linear regression."""
        # Compute y prediction with given parameters
        y_predictions = np.dot(w.T, data) + b
        # Compute cost for given predictions and true values.
        cost = LinearRegression.cost_function(y_predictions, y_true)
        # Compute gradient values for W and b parameters
        dw = LinearRegression.gradient_dw(data, y_predictions, y_true)
        db = LinearRegression.gradient_db(y_predictions, y_true)

        cost = np.squeeze(cost)

        grads = {"dw": dw,
                 "db": db}
        return grads, cost

    def optimize(self,
                 data,
                 y_true,
                 num_iterations,
                 learning_rate,
                 print_cost=False,
                 print_cost_steps=10,
                 plot_data_step=20
                 ):
        """
        Optimize W and b parameters for given data.
        :param data: Data for each attribute(feature). It should have equal dimensions as W parameter.
        :param y_true: True values for each data examples.
        :param num_iterations: How much iterations of gradient descent should do.
        :param learning_rate: Hyperparameter learning rate (a)
        :param print_cost: True if it should print cost.
        :param print_cost_steps: On how many gradient descent steps it should print cost
        :param plot_data_step:On how many gradient descent steps it should plot data, prediction, and linear
        regression model hyperplane.
        :return:
        """
        costs = []
        for i in range(num_iterations):
            if i % plot_data_step == 0:
                self.plot_prediction(data.T, y_true.T)

            # Do forward and backward propagation for linear regression. Get cost and gradient values for model.
            grads, cost = LinearRegression.propagate(self.w, self.b, data, y_true)
            dw = grads["dw"]
            db = grads["db"]

            # Change model parameters
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db

            if i % print_cost_steps == 0:
                costs.append(cost)

            if print_cost and i % print_cost_steps == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": self.w,
                  "b": self.b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs

    def predict(self, data):
        """ For public class parameters W and b predict values for given data."""
        m = data.shape[1]
        y_prediction = np.dot(self.w.T, data) + self.b
        assert (y_prediction.shape == (1, m))
        return y_prediction

    def plot_prediction(self, data, y_true):
        # plot data, prediction, and linear regression model hyperplane.
        y_estimate = self.predict(data.T)
        plt.close()
        plt.plot(data, y_estimate.T, '-', label='prediction')
        plt.plot(data, y_true, 'x', label='data')
        # Add a legend
        plt.legend()
        plt.show()
