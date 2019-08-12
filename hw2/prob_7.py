import numpy as np
import random as rd
import sys
import xlwt
import matplotlib.pyplot as plt

book = xlwt.Workbook()
sheet1 = book.add_sheet("Sheet1", cell_overwrite_ok=True)
result = []

for times in range(1000):
	#generate the data
	data_in = np.zeros([20, 2])
	for i in range(20):
		data_in[i, 0] = rd.uniform(-1, 1)
		noise = rd.random()
		data_in[i, 1] = np.sign(data_in[i, 0])
		if noise > 0.8:
			data_in[i, 1] = -data_in[i, 1]
		
	
	#generate the best choise
	min_err = 1000000; min_s = 0; min_theta = 0;
	for i in range(20):
		theta = data_in[i][0]
		for s in [-1,1]:#range(-1, 2, 2): # s = -1 or 1
			err = 0
			for j in range(20):
				if i != j and s*(data_in[j][0] - theta)*data_in[j][1] < 0:
					err += 1
			if err < min_err:
				min_err = err
				min_s = s
				min_theta = theta
			#print(data_in[i], "; s = ", s, "err = ", err)

	out_err = 0.5 + 0.3*min_s*(abs(min_theta) - 1)
	err = (min_err/20 - out_err) 

	result.append(err)

#print(result)
plt.hist(result, bins = 20)
plt.show()