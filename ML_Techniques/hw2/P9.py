import numpy as np

data = np.loadtxt('./hw2_lssvm_all.dat')
data = np.hstack((np.ones([data.shape[0], 1]), data))
X_train = data[:400, :-1]
X_test = data[400:, :-1]
Y_train = data[:400, -1].reshape(-1, 1)
Y_test = data[400:, -1].reshape(-1, 1)
lamba = [0.05, 0.5, 5, 50, 500]

def E_in(w):
	Y_pred = np.sign(X_train.dot(w))
	err = np.sum(Y_pred != Y_train) / Y_train.shape[0]
	return err

def E_out(w):
	Y_pred = np.sign(X_test.dot(w))
	#print(np.min(np.abs(X_test.dot(w))))
	err = np.sum(Y_pred != Y_test) / Y_test.shape[0]
	return err

for ld in lamba:
	w = np.linalg.inv(ld*np.identity(X_train.shape[1]) + X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
	print(ld, E_in(w), E_out(w))

