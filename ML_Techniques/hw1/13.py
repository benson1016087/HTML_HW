import numpy as np
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt


train_data = np.loadtxt('train')
test_data = np.loadtxt('test')

X = train_data[:, 1:]
Y = (train_data[:, 0] == 2)
print("X_shape :", X.shape, "Y_shape :", Y.shape)

C_rec = [-5, -3, -1, 1, 3]
w_len_rec = []

for constrain in C_rec:
	print("log_constrain = ", constrain)
	model = SVC(kernel = 'linear', C = 10 ** constrain)
	result = model.fit(X, Y)
	print(result)
	w = model.coef_
	w_len = float(np.sqrt(w.dot(w.T)))
	print(w_len)
	w_len_rec.append(w_len)


plt.plot(C_rec, w_len_rec)
plt.title("Linear SVM")
plt.xlabel('logC')
plt.ylabel('||w||')
plt.show()