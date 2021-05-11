import numpy as np
import matplotlib.pyplot as plt
from Logistic_regression.create_2_class_rand_dataset import create_dataset


class LogisticRegression:

    def __init__(self, parameter_dimension=2):
        self.w = None
        self.b = None
        self._initialize_parameters(parameter_dimension)

    def _initialize_parameters(self, dim):
        self.w = np.random.rand(dim, 1)
        self.b = 0
        assert (self.w.shape == (dim, 1))
        assert (isinstance(self.b, float) or isinstance(self.b, int))
        return self.w, self.b

    @staticmethod
    def sigmoid(z):
        s = 1 / (1 + np.exp(-z))
        return s

    @staticmethod
    def gradient_dw(X, A, Y):
        m = X.shape[1]
        dw = (1 / m) * np.dot(X, (A - Y).T)
        return dw

    @staticmethod
    def gradient_db(A, Y):
        m = Y.shape[1]
        db = (1 / m) * np.sum(A - Y)
        return db

    @staticmethod
    def cost_function(A, Y):
        m = Y.shape[1]
        cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
        return cost

    @staticmethod
    def propagate(w, b, X, Y):

        m = X.shape[1]

        A = LogisticRegression.sigmoid(np.dot(w.T, X) + b)  # compute activation
        cost = LogisticRegression.cost_function(A, Y)

        dw = LogisticRegression.gradient_dw(X, A, Y)
        db = LogisticRegression.gradient_db(A, Y)

        assert (dw.shape == w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        grads = {"dw": dw,
                 "db": db}
        return grads, cost

    def plot_decision_boundary_without_separated_classes(self, X, Y):
        arr = Y[0, :]
        class_1_index = np.where(arr == 0)
        class_2_index = np.where(arr == 1)
        class1 = X[:, class_1_index]
        class1 = np.reshape(class1, newshape=(class1.shape[0], class1.shape[-1]))
        class2 = X[:, class_2_index]
        class2 = np.reshape(class2, newshape=(class2.shape[0], class2.shape[-1]))
        self.plot_decision_boundary(X.T, Y.T, [class1[0,:], class1[1,:]], [class2[0,:], class2[1,:]])

    def optimize(self, X, Y, num_iterations, learning_rate, print_cost=False):
        costs = []
        for i in range(num_iterations):
            if i % 20 == 0:
                self.plot_decision_boundary_without_separated_classes(X, Y)

            grads, cost = LogisticRegression.propagate(self.w, self.b, X, Y)
            dw = grads["dw"]
            db = grads["db"]

            self.w = self.w - learning_rate * dw  # need to broadcast
            self.b = self.b - learning_rate * db

            if i % 10 == 0:
                costs.append(cost)

            if print_cost and i % 10 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": self.w,
                  "b": self.b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs

    def predict(self, X):
        m = X.shape[1]
        Y_prediction = LogisticRegression.sigmoid(np.dot(self.w.T, X) + self.b)
        assert (Y_prediction.shape == (1, m))
        return Y_prediction

    def classify(self, X):
        A = self.predict(X)
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        Y_prediction_p = np.zeros((1, m))
        for i in range(A.shape[1]):
            Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
            if Y_prediction[0, i] == 1:
                Y_prediction_p[0, i] = A[0, i]
            else:
                Y_prediction_p[0, i] = 1 - A[0, i]
        return Y_prediction, Y_prediction_p

    def _calculate_decision_boundary(self, x):
        t = np.zeros(100)
        for i in range(100):
            t[i] = i * 0.01
        y2 = -(x[:, 0] * self.w[0] + self.b) / self.w[1]
        return x[:, 0], y2

    def plot_decision_boundary(self, data, y_true, class1_data, class2_data, plot_detail=False):
        if len(self.w.shape) > 2:
            print("Error, w shape greater than 2. Could't plot.")
            return
        x1, x2 = self._calculate_decision_boundary(data)
        plt.close()
        plt.plot(x1, x2, '--', label='decision boundary')
        if class1_data:
            plt.plot(class1_data[0], class1_data[1], 'x', label='class 1')
        if class2_data:
            plt.plot(class2_data[0], class2_data[1], 'o', label='class 2')
        # Add a legend
        plt.legend()
        plt.show()
        plt.close()

        if plot_detail:
            self.plot_detail_space(data, y_true)

    def plot_detail_space(self, data, y_true):
        xx, yy = np.mgrid[-1:2:.1, -4:12:.1]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = self.predict(grid.T)[0, :].T.reshape(xx.shape)

        f, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                              vmin=0, vmax=1)
        ax_c = f.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, .25, .5, .75, 1])

        ax.scatter(data[100:, 0], data[100:, 1], c=y_true[100:, 0], s=50,
                   cmap="RdBu", vmin=-.2, vmax=1.2,
                   edgecolor="white", linewidth=1)

        ax.set(
            # aspect="equal",
            xlim=(0, 1.2), ylim=(-4, 12),
            xlabel="$X_1$", ylabel="$X_2$")

        plt.show()