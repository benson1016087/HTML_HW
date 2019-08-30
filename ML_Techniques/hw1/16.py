import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

train_data = np.loadtxt('train')
test_data = np.loadtxt('test')

X = train_data[:, 1:]
Y = (train_data[:, 0] == 0)
data = np.hstack((Y.reshape(-1, 1), X))

gamma_rec = [-2, -1, 0, 1, 2]
min_gamma_rec = [0, 0, 0, 0, 0] #dist = 1 / len(w)


for time in range(100):
	np.random.shuffle(data)
	X_train = data[:-1000, 1:]
	Y_train = data[:-1000, 0]
	X_vali = data[-1000:, 1:]
	Y_vali = data[-1000:, 0]
	print(time+1, "time:")
	min_E_val = 100
	min_gamma = -10
	for gma in gamma_rec:
		model = SVC(kernel = 'rbf', gamma = 10 ** gma, C = 0.1)
		result = model.fit(X_train, Y_train)
		train_acc = model.score(X_vali, Y_vali)
		E_val = 1 - train_acc
		print(gma, ", E_val =",E_val)
		if E_val < min_E_val : 
			min_E_val = E_val
			min_gamma = gma
	print("minimum gamma:",min_gamma)
	min_gamma_rec[(min_gamma+2)%5] += 1


plt.bar(gamma_rec, min_gamma_rec)
plt.title("Gaussian Kernel SVM - find minimum E_val")
plt.xlabel('log(gamma)')
plt.ylabel('times')
plt.show()


