# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
# define dataset
X = [[92], [56], [88], [70], [80], [49], [65], [35], [66], [67]]
y = [[98], [68], [81], [80], [83], [52], [66], [30], [68], [73]]
# Scatter plot
plt.scatter(X, y)
plt.xlabel('math')
plt.ylabel('cs')
# plt.show()
# mean of x and y vector
X, y = np.array(X), np.array(y)
print(type(X))
print(type(y))
print(X)
print(y)
mean_x = X.mean()
mean_y = y.mean()
# calculating cross-deviation and deviation about x
sum_yx = (X*y).sum()
x_sq = (X**2).sum()
ssxy = sum_yx - (len(X)*mean_x*mean_y)
ssxx = ((X - mean_x)**2).sum()
# calculating regression coefficients for the fisrt step
theta_1 = ssxy/ssxx
theta_0 = mean_y - (theta_1*mean_x)
# printing both the values
print('Theta 0: {:0.3f}'.format(theta_0))
print('Theta 1: {:0.3f}'.format(theta_1))
# calculate cost for the first time


def cost(theta, X, y):
    '''
    Calculates cost of the function.
    X & y have their usual meaning.
    theta - vector of coefficients.
    '''
    m = len(y)
    # Calculating Cost
    c = (1/2*m) * np.sum(np.square((X.dot(theta))-y))
    return c


"""
theta=[theta_1]
theta = np.array(theta)
print(theta)
cost1 = cost(theta,X,y)
print(cost1)
"""


def gradient_descent(X, y, theta, alpha, iterations):
    '''
    returns array of thetas, cost of every iteration
    X - X matrix with added bias.
    y - target variable matrix
    theta - matrix of regression coefficients
    alpha - learning rate
    iteration - number of iteration to be run
    '''
    # Getting number of observations.
    m = len(y)

    # Initializing cost and theta's arrays with zeroes.
    thetas = np.zeros((iterations, 2))
    costs = np.zeros(iterations)

    # Calculating theta for every iteration.
    cost_new = 1
    cost_previous = 0
    for i in range(iterations):
        if math.isclose(cost_new, cost_previous, rel_tol=1e-20) == False:
            theta = theta - (1/m)*alpha*(X.T.dot((X.dot(theta))-y))
            thetas[i, :] = theta.T
            costs[i] = cost(theta, X, y)
            cost_new = costs[i]
            if i == 1:
                cost_previous = 0
            els: cost_previous = costs[i-1]

    return theta, thetas, costs


# Learning Rate
alpha = 0.01
# Number of iterations
iterations = 3000
# Initializing a random value to give algorithm a base value.
theta = np.random.randn(2, 1)
# Adding a biasing constant of value 1 to the features array.
X_bias = np.c_[np.ones((len(X), 1)), X]
# Running Gradient Descent
theta, thetas, costs = gradient_descent(X_bias, y, theta, alpha, iterations)
# printing final values.
#print('Final Theta 0 value: {:0.3f}\nFinal Theta 1 value: {:0.3f}'.format(theta[0][0],theta[1][0]))
#print('Final Cost/MSE(L2 Loss) Value: {:0.3f}'.format(costs[-1]))

csv.writer(cost_function_variables.csv, dialect='excel')

with open('cost_function_variables.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(["coefficient and intercept ", "cost"])
    writer.writerow([1, theta])
    writer.writerow([2, costs])
