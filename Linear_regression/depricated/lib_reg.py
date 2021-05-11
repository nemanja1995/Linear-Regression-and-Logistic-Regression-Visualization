import numpy as np
import matplotlib.pyplot as plt

# Prepare the data
# x = np.linspace(0, 10, 100)

data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
data_y = np.sin(data_x) + 0.1*np.power(data_x, 2) + 0.5*np.random.randn(100,1)
data_x /= np.max(data_x)

# Plot the data
plt.plot(data_x, data_y, 'o', label='linear')
# Add a legend
plt.legend()
# Show the plot
plt.show()
data_x = np.hstack((np.ones_like(data_x), data_x))
order = np.random.permutation(len(data_x))
portion = 20
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]


def gradient(w, x, y):
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    gradient = -(1.0/len(x)) * error.dot(x)
    return gradient, np.power(error, 2)


w = np.random.randn(2)
alpha = 0.5
tolerance = 1e-5

# Perform Gradient Descent
iterations = 1
while True:
    grad, error = gradient(w, train_x, train_y)
    new_w = w - alpha * grad

    # Stopping Condition
    if np.sum(abs(new_w - w)) < tolerance:
        print("Converged.")
        break

    # Print error every 50 iterations
    if iterations % 10 == 0:
        print("Iteration: %d - Error: %.4f" % (iterations, error[-1]))

    iterations += 1
    w = new_w

y_estimate = data_x.dot(w).flatten()
plt.close()
plt.plot(data_x[:, 1], y_estimate, 'o', label='preds')
plt.plot(data_x[:, 1], data_y, 'x', label='linear')
# Add a legend
plt.legend()
plt.show()
