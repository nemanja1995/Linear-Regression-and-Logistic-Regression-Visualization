"""
Creating random dastaset (x1, x2) pairs for Logistic regression. It creates data for two classes,
with 2 features (x1 feature, and x2 feature).
It is used to create simple, easily visualised dataset for visualising Logistic regression.
"""
import numpy as np
import matplotlib.pyplot as plt


def create_dataset(num_data_per_class=100, plot_data=False, plot_features=False):
    """
    Create data with 2 features for 2 separated classes. That data can be used for training and visualising
    Logistic regression witch can separate 2 data clusters (data for two classes).
    :param num_data_per_class: How much data pairs it should generate for each class.
    Total number of data examples will be num_data_per_class * 2.
    :param plot_data: Plot data pairs (x1, x2). Plot features.
    :param plot_features: Plot data pairs (xi, class). Plot chart for (x1, class), (x2, class).
    :return:
        data - num_data_per_class * 2 shuffled (x1, x2 feature) pairs.
        classes - num_data_per_class * 2 shuffled labels (class for each (x1, x2) feature pair) for data.
        class1_pairs, class2_pairs - separated (x1, x2) pairs per class.
    """
    # Create data for class 1
    class_1_x1 = np.linspace(1.0, 10.0, num_data_per_class)[:, np.newaxis]
    class_1_x2 = np.sin(class_1_x1) + 0.1*np.power(class_1_x1, 2) + 0.5*np.random.randn(num_data_per_class, 1) + 2
    class_1_x1 /= np.max(class_1_x1)
    y1 = np.zeros(shape=(num_data_per_class, 1))

    # Create data for class 2
    class_2_x1 = np.linspace(1.0, 10.0, num_data_per_class)[:, np.newaxis]
    class_2_x2 = np.cos(class_2_x1) + 0.1*np.power(class_2_x1, 2) + 0.8*np.random.randn(num_data_per_class, 1) - 2
    class_2_x1 /= np.max(class_2_x1)
    y2 = np.ones(shape=(num_data_per_class, 1))

    # Plot x1, x2 pairs
    if plot_data:
        plt.plot(class_1_x1, class_1_x2, 'o', label='class 1')
        plt.plot(class_2_x1, class_2_x2, 'x', label='class 2')
        plt.legend()
        plt.show()

    # Plot feature with class. Plot (xi feature, y class) pairs.
    if plot_features:
        plt.close()
        plt.plot(class_1_x1, y1, 'o', label='class 1')
        plt.plot(class_2_x1, y2, 'x', label='class 2')
        plt.legend()
        plt.show()

        plt.close()
        plt.plot(class_1_x2, y1, 'o', label='class 1')
        plt.plot(class_2_x2, y2, 'x', label='class 2')
        plt.legend()
        plt.show()

    y = np.append(y1, y2)
    x1 = np.append(class_1_x1, class_2_x1)
    x2 = np.append(class_1_x2, class_2_x2)
    x = np.zeros(shape=(num_data_per_class*2, 2))
    x[:, 0] = x1
    x[:, 1] = x2

    idx = np.random.permutation(x.shape[0])
    data, classes = x[idx, :], y[idx]
    classes = np.reshape(classes, newshape=(200, 1))
    class1_pairs = [class_1_x1, class_1_x2]
    class2_pairs = [class_2_x1, class_2_x2]
    return data, classes, class1_pairs, class2_pairs


if __name__ == "__main__":
    create_dataset(100, True, True)
