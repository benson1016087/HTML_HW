import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

train_data = np.loadtxt('train')
test_data = np.loadtxt('test')

X = train_data[:, 1:]
Y = (train_data[:, 0] == 4)
X_test = test_data[:, 1:]
Y_test = (test_data[:, 0] == 4)
print("X_shape :", X.shape, "Y_shape :", Y.shape)

C_rec = [-5, -3, -1, 1, 3]
E_in_rec = []

for constrain in C_rec:
	print("log_constrain = ", constrain)
	model = SVC(kernel = 'poly', degree = 2, gamma = 1, coef0 = 1, C = 10 ** constrain)
	result = model.fit(X, Y)
	print(result)
	train_acc = model.score(X, Y)
	E_in = 1 - train_acc
	print(E_in)
	E_in_rec.append(E_in)

plt.plot(C_rec, E_in_rec)
plt.title("Polynomial Kernel SVM")
plt.xlabel('logC')
plt.ylabel('E_in')
plt.show()


