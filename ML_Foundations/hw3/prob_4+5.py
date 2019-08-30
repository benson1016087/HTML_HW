import numpy as np
import random as rd
import xlwt
import matplotlib.pyplot as plt

train_data = np.loadtxt('hw3_train.dat')
test_data = np.loadtxt('hw3_test.dat')

dim = len(train_data[0])
data_size = len(train_data)
print('data_size =', data_size)
test_size = len(test_data)
add_1 = np.ones(data_size)
add_1_test = np.ones(test_size)
train_data = np.column_stack((add_1.T, train_data))
test_data = np.column_stack((add_1_test.T, test_data))

ita_SGD = 0.001
ita_GD = 0.01

w_SGD = np.zeros(dim)
w_GD = np.zeros(dim)

def update_SGD(i):
	index = train_data[i][-1]*w_SGD.dot(train_data[i][0:-1])
	return ((-train_data[i][-1]*train_data[i][0:-1]) / (1 + np.exp(index)))

def update_GD():
	add = 0
	for i in range(data_size):
		index = train_data[i][-1]*w_GD.dot(train_data[i][0:-1])
		add += ((-train_data[i][-1]*train_data[i][0:-1]) / (1 + np.exp(index)))
	add /= data_size
	return add

def err_train(w_in):
	wrong = 0
	for i in range(data_size):
		jdg = w_in.dot(train_data[i][0:-1])
		if(np.sign(jdg)*train_data[i][-1] < 0):
			wrong += 1
	return wrong / data_size

def err_test(w_in):
	wrong = 0
	for i in range(test_size):
		jdg = w_in.dot(test_data[i][0:-1])
		if(np.sign(jdg) != test_data[i][-1]):
			wrong += 1
	return wrong / test_size

index_num = []
result_SGD = []
result_GD = []
result_GD_test = []
result_SGD_test = []

for t in range(2000):
	up_num = t % data_size
	#print(rd_num)
	w_SGD = w_SGD - ita_SGD*update_SGD(up_num)
	w_GD = w_GD - ita_GD*update_GD()
	#print(w_SGD)
	
	index_num.append(t)
	er_SGD = err_train(w_SGD)
	result_SGD.append(er_SGD)
	er_GD = err_train(w_GD)
	result_GD.append(er_GD)
	result_SGD_test.append(err_test(w_SGD))
	result_GD_test.append(err_test(w_GD))
	if (t+1)%100 == 0:
		print('t done =',(t+1))

plt.plot(index_num, result_GD, label = 'result of GD')
plt.plot(index_num, result_SGD, label = 'result of SGD')
plt.xlabel('update time')
plt.ylabel('E_in')
plt.legend()
plt.show()

plt.figure()
plt.plot(index_num, result_GD_test, label = 'result of GD')
plt.plot(index_num, result_SGD_test, label = 'result of SGD')
plt.xlabel('update time')
plt.ylabel('E_out')
plt.legend()
plt.show()