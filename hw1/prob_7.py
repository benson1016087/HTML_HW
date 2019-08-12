import numpy as np 
import math
import sys
import xlwt

data = np.loadtxt('hw1_7_train.dat')
data = np.insert(data, 0, 1.0, axis=1)
time = np.zeros([1,100]);

for test in range(1126): 
	a = np.random.choice(400, 400, replace=False)
	update = 0
	if data[0][5] > 0:
		np_pla = data[0][0:5]
	else: 
		np_pla = -data[0][0:5]
	for t in range(3):
		for i in a:
			if np.sign(np.dot(np_pla,data[i][0:5])) == np.sign(data[i][5]):
				continue
			else:
				if data[i][5] > 0:
					np_pla = np_pla + data[i][0:5]
				else:
					np_pla = np_pla - data[i][0:5]
				update += 1
	time[0][update] += 1
print(time)

book = xlwt.Workbook()
sheet1 = book.add_sheet("Sheet1", cell_overwrite_ok=True)
row = 0
for i in time[0]:
	sheet1.write(row, 0, row)
	sheet1.write(row, 1, i)
	row += 1
book.save('7.xls')