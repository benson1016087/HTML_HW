import numpy as np
np.random.seed(6101317)

data = np.loadtxt('./hw2_lssvm_all.dat')
data = np.hstack((np.ones([data.shape[0], 1]), data))
X_train = data[:400, :-1]
X_test = data[400:, :-1]
Y_train = data[:400, -1].reshape(-1, 1)
Y_test = data[400:, -1].reshape(-1, 1)
lamba = [0.05, 0.5, 5, 50, 500]

def E_in(w_rec):
	Y_tot = 0
	for w in w_rec:
		Y_pred = np.sign(X_train.dot(w))
		Y_tot += Y_pred
	Y_tot = np.sign(Y_tot + 0.1)
	err = np.sum(Y_tot != Y_train) / Y_train.shape[0]
	return err

def E_out(w_rec):
	Y_tot = 0
	for w in w_rec:
		Y_pred = np.sign(X_test.dot(w))
		Y_tot += Y_pred
	Y_tot = np.sign(Y_tot + 0.1)
	err = np.sum(Y_tot != Y_test) / Y_test.shape[0]
	return err	


for ld in lamba:
	w_rec = []
	for itr in range(250):
		u = np.zeros(400)
		for times in range(400):
			a = np.random.randint(400);
			u[a] += 1
		u = u.reshape([400, 1])
		new_X_train = u * X_train
		w = np.linalg.inv(ld*np.identity(new_X_train.shape[1]) + new_X_train.T.dot(new_X_train)).dot(new_X_train.T).dot(Y_train)
		w_rec.append(w)
	print(ld, E_in(w_rec), E_out(w_rec))