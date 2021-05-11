"""
Creating random dastaset (x, y) pairs. It is used to create simple, easily visualised dataset
for visualising Linear regression.
"""
import numpy as np
import matplotlib.pyplot as plt


def create_dataset(num_data=100, plot_data=False):
    """
    Create data pairs (x, y) for Logistic regression.
    :param num_data: How much data pairs it should generate.
    :param plot_data: After generating data plot it (show data) into 2d space.
    :return: Dataset (num_data pairs (x, y) as numpy array)
    """
    # Create data pairs (x, y)
    data_x = np.linspace(1.0, 10.0, num_data)[:, np.newaxis]
    data_y = np.sin(data_x) + 0.1 * np.power(data_x, 2) + 0.5 * np.random.randn(100, 1)
    data_x /= np.max(data_x)

    # Create random permutations witch you can use to shuffle data.
    order = np.random.permutation(len(data_x))
    portion = 20
    train_x = data_x[order[portion:]]
    train_y = data_y[order[portion:]]

    if plot_data:
        plt.plot(train_x, train_y, 'o', label='data')
        plt.legend()
        plt.show()

    return train_x, train_y


if __name__ == "__main__":
    create_dataset(100, True)