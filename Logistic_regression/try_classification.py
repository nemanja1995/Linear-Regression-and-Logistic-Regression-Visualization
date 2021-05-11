from Logistic_regression.LogisticRegression import LogisticRegression
import numpy as np


def try_predicting_given_value(parameters, train_x, train_y):
    log_regression = LogisticRegression(parameter_dimension=2)
    w = parameters['w']
    b = parameters['b']
    log_regression.w = w
    log_regression.b = b
    class1 = [[0.4, 2]]
    prediction_data = []
    log_regression.plot_decision_boundary(train_x, train_y, [], [], plot_detail=False)
    exit = False
    while not exit:
        print("Enter 2 values (or exit):")
        print("x1: ")
        x1 = input()
        if x1 == "exit":
            exit = True
            break
        print("x2: ")
        x2 = input()
        a = np.empty(shape=(1))
        a[0] = x1
        b = np.empty(shape=(1))
        b[0] = x2
        prediction_data = [a, b]
        data = np.empty(shape=(1, 2))
        data[0, 0] = x1
        data[0, 1] = x2
        prediction, percent = log_regression.classify(data.T)
        print("Prediction: " + str(prediction[0, 0]) + ", with certainty: " + str(percent[0, 0]))
        if prediction[0, 0] == 1:
            log_regression.plot_decision_boundary(train_x, train_y, prediction_data, [], plot_detail=False)
        else:
            log_regression.plot_decision_boundary(train_x, train_y, [], prediction_data, plot_detail=False)


