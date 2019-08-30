import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

train_data = np.loadtxt('train')
test_data = np.loadtxt('test')

X = train_data[:, 1:]
Y = (train_data[:, 0] == 0)
print("X_shape :", X.shape, "Y_shape :", Y.shape)

C_rec = [-2, -1, 0, 1, 2]
dist_rec = [] #dist = 1 / len(w)

for constrain in C_rec:
	print("log_constrain = ", constrain)
	model = SVC(kernel = 'rbf', gamma = 80, C = 10 ** constrain)
	result = model.fit(X, Y)
	print(result)
	x_support = model.support_vectors_
	alpha = model.dual_coef_[0]
	print(alpha)
	w_2 = 0
	for i in range(x_support.shape[0]):
		for j in range(x_support.shape[0]):
			w_2 += alpha[i]*alpha[j]*np.exp(-80*(x_support[i, :]-x_support[j, :]).T.dot(x_support[i, :]-x_support[j, :]))
	dist = 1 / np.sqrt(w_2)
	'''
	alpha_mul_x = model.dual_coef_.T*model.support_vectors_
	w = np.sum(alpha_mul_x, axis = 0)
	print("w_shape : ", w.shape)
	dist = 1 / np.sqrt(w.T.dot(w))
	print("dist = ", dist)
	'''
	dist_rec.append(dist)

plt.plot(C_rec, dist_rec)
plt.title("Gaussian Kernel SVM")
plt.xlabel('logC')
plt.ylabel('distance')
plt.show()


