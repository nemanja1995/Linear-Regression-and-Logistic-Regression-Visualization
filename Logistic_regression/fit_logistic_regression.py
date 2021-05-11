import numpy as np
import matplotlib.pyplot as plt

from Logistic_regression.create_2_class_rand_dataset import create_dataset
from Logistic_regression.LogisticRegression import LogisticRegression
from Logistic_regression.try_classification import try_predicting_given_value


w = np.random.rand(2, 1)
b = 0
train_x, train_y, class1, class2 = create_dataset()
# train_x = train_x[:, 1]
# train_x = np.reshape(train_x, newshape=(len(train_x), 1))

log_regression = LogisticRegression(parameter_dimension=2)
log_regression.optimize(train_x.T, train_y.T, 150, 0.5, print_cost=True)
log_regression.plot_decision_boundary(train_x, train_y, class1, class2, plot_detail=True)
log_regression.predict(train_x.T)

parameters = {}
parameters["w"] = log_regression.w
parameters["b"] = log_regression.b

try_predicting_given_value(parameters, train_x, train_y)
