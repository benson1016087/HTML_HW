import numpy as np
import matplotlib.pyplot as plt
data_train = np.loadtxt('hw4_train.dat')
data_test = np.loadtxt('hw4_test.dat')

X_train = data_train[:, :-1]
Y_train = data_train[:, -1]
X_test = data_test[:, :-1]
Y_test = data_test[:, -1]
print('finish dealing with data')

# prob_11
k_rec = [1, 3, 5, 7, 9]
Ein_rec = []
for k in k_rec:
	wrong = 0
	for i in range(X_train.shape[0]):
		dist = np.zeros(X_train.shape[0])
		for j in range(X_train.shape[0]):
			dist[j] = (X_train[i] - X_train[j]).T.dot(X_train[i] - X_train[j])
		top_k = np.argsort(dist)[:k]
		pred = np.sign(np.sum(Y_train[top_k]))
		if(pred != Y_train[i]):
			wrong += 1
	Ein_rec.append(wrong / X_train.shape[0])

plt.figure()
plt.plot(k_rec, Ein_rec)
plt.xlabel('k')
plt.ylabel('$E_{in}(g_{knbor})$')
plt.legend()
plt.savefig('Prob_11.png')
print('finishing prob_11')

# prob_12
k_rec = [1, 3, 5, 7, 9]
Eout_rec = []
for k in k_rec:
	wrong = 0
	for i in range(X_test.shape[0]):
		dist = np.zeros(X_train.shape[0])
		for j in range(X_train.shape[0]):
			dist[j] = (X_test[i] - X_train[j]).T.dot(X_test[i] - X_train[j])
		top_k = np.argsort(dist)[:k]
		pred = np.sign(np.sum(Y_train[top_k]))
		if(pred != Y_test[i]):
			wrong += 1
	Eout_rec.append(wrong / X_test.shape[0])

plt.figure()
plt.plot(k_rec, Eout_rec)
plt.xlabel('k')
plt.ylabel('$E_{out}(g_{knbor})$')
plt.legend()
plt.savefig('Prob_12.png')
print('finishing prob_12')


gamma_rec = [0.001, 0.1, 1, 10, 100]
log_gamma = [-3, -1, 0, 1, 2]
# prob_13
Ein_rec = []
for gamma in gamma_rec:
	wrong = 0
	for i in range(X_train.shape[0]):
		dist_2 = np.sum((X_train - X_train[i]) ** 2, axis = 1)
		pred = np.sign(np.sum(Y_train*np.exp(-gamma * dist_2)))
		if(pred != Y_train[i]):
			wrong += 1
	Ein_rec.append(wrong / X_train.shape[0])

plt.figure()
plt.plot(log_gamma, Ein_rec, 'bo')
plt.xlabel('$log_{10}(\gamma)$')
plt.ylabel('$E_{in}(g_{uniform})$')
plt.legend()
plt.savefig('Prob_13.png')
print('finishing prob_13')

# prob_14
Ein_rec = []
for gamma in gamma_rec:
	wrong = 0
	for i in range(X_test.shape[0]):
		dist_2 = np.sum((X_train - X_test[i]) ** 2, axis = 1)
		pred = np.sign(np.sum(Y_train*np.exp(-gamma * dist_2)))
		if(pred != Y_test[i]):
			wrong += 1
	Ein_rec.append(wrong / X_test.shape[0])

plt.figure()
plt.plot(log_gamma, Ein_rec, 'bo')
plt.xlabel('$log_{10}(\gamma)$')
plt.ylabel('$E_{out}(g_{uniform})$')
plt.legend()
plt.savefig('Prob_14.png')
print('finishing prob_14')


