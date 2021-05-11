import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


def create_dataset():
    data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
    data_y = np.sin(data_x) + 0.1 * np.power(data_x, 2) + 0.5 * np.random.randn(100, 1)
    data_x /= np.max(data_x)

    data_x = np.hstack((np.ones_like(data_x), data_x))
    order = np.random.permutation(len(data_x))
    portion = 20
    train_x = data_x[order[portion:]]
    train_y = data_y[order[portion:]]
    return train_x, train_y


def predict(w, b, x):
    pred = np.dot(w.T, x) + b
    return pred


def cost_function(w, b, x, y_true):
    y_pred = predict(w, b, x)
    num_examples = y_pred.shape[-1]
    cost = 1 / (2 * num_examples) * np.sum(np.square(np.subtract(y_pred, y_true)))
    return cost


def gradient_dw_j(w, b, x, y_true):
    y_pred = predict(w, b, x)
    num_examples = y_pred.shape[-1]
    w_gradient = 1 / num_examples * np.sum(np.multiply(np.subtract(y_pred, y_true), x))
    return w_gradient


def gradient_db_j(w, b, x, y_true):
    y_pred = predict(w, b, x)
    num_examples = y_pred.shape[-1]
    b_gradient = 1 / num_examples * np.sum(np.subtract(y_pred, y_true))
    return b_gradient


def gradient_descent(x, y, w, b, num_epoch):
    cost_history = []
    a = 0.5
    # mu = np.mean(x, 0)
    # sigma = np.std(x, 0)
    # x = (x - mu) / sigma

    for i in range(num_epoch):
        cost = cost_function(w, b, x, y)
        cost_history.append(cost)
        print("Cost: " + str(cost))

        dw = gradient_dw_j(w, b, x, y)
        db = gradient_db_j(w, b, x, y)

        w = w - a * dw
        b = b - a * db
    return w, b


w = np.random.rand(1, 1)
b = 0
train_x, train_y = create_dataset()
train_x = train_x[:, 1]
train_x = np.reshape(train_x, newshape=(len(train_x), 1))
# (x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
w, b = gradient_descent(train_x.T, train_y.T, w, b, 200)

y_estimate = predict(w, b, train_x.T)

plt.close()
plt.plot(train_x, y_estimate.T, 'o', label='preds')
plt.plot(train_x, train_y, 'x', label='linear')
# Add a legend
plt.legend()
plt.show()
