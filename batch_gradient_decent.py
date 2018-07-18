import numpy as np
import random
import matplotlib.pyplot as plt
import time

np.random.seed(0)

def generate_data_set(num_rows=100,num_columns=1):
	points = np.random.random((num_rows, num_columns)) * np.random.random_integers(0,10)
	weights = np.random.random_integers(1, 100, (num_columns, 1))
	bias = np.random.random_integers(0, 100, [1,1])
	y = np.dot(points, weights) + bias + np.random.normal(0, 5, (num_rows, 1))
	weights = np.append(weights, bias, axis=0)
	return points, y, weights


def calculate_mean_squared_error(y_pred, y):
	return np.sqrt(np.mean(np.square(y_pred - y)))


def evaluate_gradient(data, y , params):
	y_pred = np.dot(data, params)
	loss = y_pred - y
	cost = calculate_mean_squared_error(y_pred, y)
	gradients = np.dot(data.transpose(), loss) / x.shape[0]
	return gradients, cost


def update_cost_list(cost_list, cost):
    if len(cost_list) == 5:
        _ = cost_list.pop(0)
        cost_list.append(cost)
    else:
        cost_list.append(cost)
    return cost_list


def check_cost_list(cost_list, threshold):
    if abs(np.mean(np.diff(cost_list))) < threshold:
        return True
    else:
        return False


def gradient_decent_wihout_epoch(x, y, l_rate=0.01, cost_limit=0.00001):
	num_rows, num_columns = x.shape
	params = np.ones((num_columns + 1, 1))
	x = np.append(x, np.ones((num_rows, 1)), axis=1)
	cost_list = []
	epoch = 0
	while True:
		grad, cost = evaluate_gradient(x, y, params)
		cost_list = update_cost_list(cost_list, float(cost))
		if check_cost_list(cost_list, cost_limit):
			break
		else:
			params -= l_rate * grad
			epoch += 1
	return params, cost_list, epoch

def gradient_decent_with_epoch(x, y, n_epochs=100, l_rate=0.01):
	num_rows, num_columns = x.shape
	params = np.ones((num_columns + 1, 1))
	x = np.append(x, np.ones((num_rows, 1)), axis=1)
	cost_list = []
	for _ in range(n_epochs):
		grad, cost = evaluate_gradient(x, y, params)
		cost_list.append(cost)
		#print("cost is {}".format(cost))
		params -= l_rate * grad
	return params, cost_list

if __name__ == "__main__":
	x, y, weights = generate_data_set(200000,6)
	#epoch = 10000
	print("original weights")
	print(weights)
	start = time.time()
	new_weights, costs, epoch = gradient_decent_wihout_epoch(x, y, l_rate=0.01)
	print("new weights")
	print(new_weights)
	print("number of epochs: {}".format(epoch))
	print(time.time() - start)