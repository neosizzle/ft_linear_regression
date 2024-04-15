import csv
import matplotlib.pyplot as plt
import numpy as np

def r2(actual_data, b1, b0):
	x_set = actual_data[0]
	y_set = actual_data[1]
	mean_y = mean(y_set)
	predicted_ys = []
	a = 0
	b = 0

	for x in x_set:
		y = b1 * x + b0
		predicted_ys.append(y)
	for idx in range(len(predicted_ys)):
		predicted_y = predicted_ys[idx]
		actual_y = y_set[idx]

		a += (predicted_y - mean_y) * (predicted_y - mean_y)
		b += (actual_y - mean_y) * (actual_y - mean_y)
	return a / b

def stddev(data, ddof=0):
    # Calculate the mean of the data
    mean_data = sum(data) / len(data)

    # Calculate squared differences for each data point and mean
    squared_diffs = [(x - mean_data) ** 2 for x in data]

    # Calculate the average of the squared differences
    variance = sum(squared_diffs) / (len(squared_diffs) - ddof)

    # Calculate the square root of the variance
    stddev = variance ** 0.5
    return stddev

def mean(set):
	data_len = len(set)
	total = 0
	for i in set :
		total += i
	return total / data_len

def standardize_data(data):
	res = [[], []]
	x_set = data[0]
	y_set = data[1]
	mean_x = mean(x_set)
	mean_y = mean(y_set)
	stddev_x = stddev(x_set)
	stddev_y = stddev(y_set)

	for idx in range(len(x_set)):
		x = x_set[idx]
		y = y_set[idx]

		res[0].append(((x - mean_x) / stddev_x))
		res[1].append(((y - mean_y) / stddev_y))

	return res

def reverse_coefficients(data, old_b1, old_b0) :
	x_set = data[0]
	y_set = data[1]
	mean_x = mean(x_set)
	mean_y = mean(y_set)
	stddev_x = stddev(x_set)
	stddev_y = stddev(y_set)

	new_b1 = (stddev_y * old_b1) / stddev_x
	new_b0 = -(old_b1 * mean_x * stddev_y) / stddev_x + mean_y + (old_b0 * stddev_y)

	return (new_b1, new_b0)

# read and parse csv
def read_csv() :
	x_set = []
	y_set = []
	file=open("data.csv", "r")
	reader = csv.reader(file)
	for line in reader:
		t=line
		try:
			x_set.append(int(t[0]))
			y_set.append(int(t[1]))
		except:
			True
	return [x_set, y_set]

def derivitive_b1_squared_residuals(x, y, b1, b0):
	return (-2 * x *(y - (b1 * x) - b0))

def derivitive_b0_squared_residuals(x, y, b1, b0):
	return (-2 * (y - (b1 * x) - b0))

def step_size_out_of_range(number, range) :
	return number < -range or number > range

def gradiant_descent_step(data, init_b1, init_b0, learning_rate, steps):
	x_set = data[0]
	y_set = data[1]
	step_size_b0 = 6969
	step_size_b1 = 6969
	step_size_range = 0.00001
	
	while steps != 0 :
		should_change_b0 = False
		should_change_b1 = False

		sum_of_b0s = 0
		for idx in range(len(x_set)):
			new_b0 = derivitive_b0_squared_residuals(x_set[idx], y_set[idx], init_b1, init_b0)
			sum_of_b0s += new_b0
		step_size_b0 = sum_of_b0s * learning_rate
		should_change_b0 = True

		sum_of_b1s = 0
		for idx in range(len(x_set)):
			new_b1 = derivitive_b1_squared_residuals(x_set[idx], y_set[idx], init_b1, init_b0)
			sum_of_b1s += new_b1
		step_size_b1 = sum_of_b1s * learning_rate
		should_change_b1 = True

		# update b0 and b1 babsed on stepsize
		if should_change_b0:
			init_b0 = init_b0 - step_size_b0
		if should_change_b1:
			init_b1 = init_b1 - step_size_b1
		steps -= 1

	print(f"b1 {init_b1} b0 {init_b0}")
	return (init_b1, init_b0)

def gradiant_descent(data, init_b1, init_b0, learning_rate):
	x_set = data[0]
	y_set = data[1]
	step_size_b0 = 6969
	step_size_b1 = 6969
	# step_size_range = 0.0000001
	step_size_range = 0.001
	steps = 0
	
	while step_size_out_of_range(step_size_b0, step_size_range) or step_size_out_of_range(step_size_b1, step_size_range) :
		should_change_b0 = False
		should_change_b1 = False
		steps += 1

		# GD for loss function for b0
		if step_size_out_of_range(step_size_b0, step_size_range) :
			sum_of_b0s = 0
			for idx in range(len(x_set)):
				new_b0 = derivitive_b0_squared_residuals(x_set[idx], y_set[idx], init_b1, init_b0)
				sum_of_b0s += new_b0
			step_size_b0 = sum_of_b0s * learning_rate
			should_change_b0 = True

		# GD for loss function for b1
		if step_size_out_of_range(step_size_b1, step_size_range) :
			sum_of_b1s = 0
			for idx in range(len(x_set)):
				new_b1 = derivitive_b1_squared_residuals(x_set[idx], y_set[idx], init_b1, init_b0)
				sum_of_b1s += new_b1
			step_size_b1 = sum_of_b1s * learning_rate
			should_change_b1 = True

		# update b0 and b1 babsed on stepsize
		if should_change_b0:
			init_b0 = init_b0 - step_size_b0
		if should_change_b1:
			init_b1 = init_b1 - step_size_b1
		# print(f"step size b0 {step_size_b0}, step size b1 {step_size_b1}")
	print(f"GD b1 {init_b1} b0 {init_b0}, steps {steps}")
	return (init_b1, init_b0)

# least squares method
#
# b1 = total of ((x - mean x) * (y - mean y)) / total of (x - mean x) ^ 2
# b0 = mean y - (b1 * mean x)
#
def least_squares(data):
	x_set = data[0]
	y_set = data[1]
	mean_x = mean(x_set)
	mean_y = mean(y_set)
	
	total_x_m_mean_x_t_y_m_mean_y = 0
	total_x_m_mean_x_sq = 0
	for idx in range(len(x_set)):
		x = x_set[idx]
		y = y_set[idx]

		x_m_mean_x = x - mean_x
		y_m_mean_y = y - mean_y
		total_x_m_mean_x_t_y_m_mean_y += (x_m_mean_x * y_m_mean_y)

		x_m_mean_x_sq = x_m_mean_x * x_m_mean_x
		total_x_m_mean_x_sq += x_m_mean_x_sq
	b1 = total_x_m_mean_x_t_y_m_mean_y / total_x_m_mean_x_sq
	b0 = mean_y - (b1 * mean_x)

	print(f"LS b1 {b1}, b0 {b0}")
	return (b1, b0)

data = read_csv()
data_standardized = standardize_data(data)
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(2, 1, figsize=(15, 15))
ax[0].set_title("Actual scale")
ax[1].set_title("Standardized scale")
plt.tight_layout(pad=2)


# regression LS
slope, y_inter = least_squares(data_standardized)
print(f"LS r2: {r2(data_standardized, slope, y_inter)}")

x_regres = [-10, 10]
y_regres = [slope * - 10 + y_inter, slope * 10 + y_inter]
ax[1].plot(x_regres, y_regres, 'r', alpha=0.5)

slope_reverted, y_inter_reverted = reverse_coefficients(data, slope, y_inter)
x_regres = [10000, 250000]
y_regres = [slope_reverted * 10000 + y_inter_reverted, slope_reverted * 250000 + y_inter_reverted]
ax[0].plot(x_regres, y_regres, 'r', alpha=0.5)

# regression GD
slope, y_inter = gradiant_descent(data_standardized, 0, 0, 0.004)
print(f"GD r2: {r2(data_standardized, slope, y_inter)}")

x_regres = [-10, 10]
y_regres = [slope * - 10 + y_inter, slope * 10 + y_inter]
ax[1].plot(x_regres, y_regres, 'c', alpha=0.5)

slope_reverted, y_inter_reverted = reverse_coefficients(data, slope, y_inter)
x_regres = [10000, 250000]
y_regres = [slope_reverted * 10000 + y_inter_reverted, slope_reverted * 250000 + y_inter_reverted]
ax[0].plot(x_regres, y_regres, 'c', alpha=0.5)

# write legends
ax[0].legend(["Least Square", "Gradient Descent"])
ax[1].legend(["Least Square", "Gradient Descent"])

# write GD values to file
f = open("values.csv", "a")
f.write(f"{slope_reverted},{y_inter_reverted}")
f.close()


# plot data
ax[0].plot(data[0], data[1], 'bo')
ax[0].set(xlim=(min(data[0]) - 10000 , max(data[0]) + 10000),
       ylim=(min(data[1]) - 100 , max(data[1]) + 100))

ax[1].plot(data_standardized[0], data_standardized[1], 'bo')
ax[1].set(xlim=(min(data_standardized[0]) - 1 , max(data_standardized[0]) + 1),
       ylim=(min(data_standardized[1]) - 1 , max(data_standardized[1]) + 1))

plt.savefig("matplotlib.png")