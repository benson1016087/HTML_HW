import numpy as np
import matplotlib.pyplot as plt
np.random.seed(61017)

data_train = np.loadtxt('./hw2_adaboost_train.dat')
data_test = np.loadtxt('./hw2_adaboost_test.dat')
X_test = data_test[:, :-1]
Y_test = data_test[:, -1] #.reshape(-1, 1)

arg_sort = np.argsort(data_train, axis = 0)
X_train_1 = data_train[arg_sort[:, 0]]
X_train_2 = data_train[arg_sort[:, 1]]

Y_train_1 = X_train_1[:, -1] #.reshape(-1, 1)
Y_train_2 = X_train_2[:, -1] #.reshape(-1, 1)
X_train_1 = X_train_1[:, :-1]
X_train_2 = X_train_2[:, :-1]


def E_in_G(w_rec, alpha_rec):
	Y_pred = np.zeros(Y_train_1.shape[0]) + 1e-3
	for i in range(len(w_rec)):
		w = w_rec[i]
		alpha = alpha_rec[i]
		Y_pred += alpha*w[2]*np.sign(X_train_1[:, int(w[1])] - (w[0] - 1e-10))
	Y_pred = np.sign(Y_pred)
	err = np.sum(Y_pred != Y_train_1) / Y_train_1.shape[0]
	return err

def E_out_G(w_rec, alpha_rec):
	Y_pred = np.zeros(Y_test.shape[0]) + 1e-3
	for i in range(len(w_rec)):
		w = w_rec[i]
		alpha = alpha_rec[i]
		Y_pred += alpha*w[2]*np.sign(X_test[:, int(w[1])] - (w[0] - 1e-10))
	Y_pred = np.sign(Y_pred)
	err = np.sum(Y_pred != Y_test) / Y_test.shape[0]
	return err

def E_in_g(w, u):
	#print(w)
	dim = int(w[1])
	cmp_num = w[0] - 1e-5
	X = X_train_1[:, dim]
	pred_arr = w[2] * np.sign(X - cmp_num)
	cmp_arr = (pred_arr != Y_train_1)
	err = np.sum(u * cmp_arr)
	return err, cmp_arr

def update(w_in, u_in):
	err_up, modi = E_in_g(w_in, u_in)
	err_up /= np.sum(u_in)
	diamond_t = np.sqrt((1 - err_up)/err_up)
	alpha = np.log(diamond_t)
	#print(np.sum(modi), w_in)
	#exit()
	for i in range(u_in.shape[0]):
		if modi[i] == 1:
			u_in[i] *= diamond_t
		else: 
			u_in[i] /= diamond_t
	#print(np.sum(u_in))
	return u_in, alpha


def choose_w(u):
	min_err = 10000
	min_w = -1
	for i in range(X_train_1.shape[0]):
		w = np.array([X_train_1[i, 0], 0, 1])
		err = E_in_g(w, u)[0]
		if  err <= min_err:
			min_err = err
			min_w = w
		w = np.array([X_train_1[i, 0], 0, -1])
		err = E_in_g(w, u)[0]
		if  err <= min_err:
			min_err = err
			min_w = w
		w = np.array([X_train_2[i, 1], 1, 1])
		err = E_in_g(w, u)[0]
		if  err <= min_err:
			min_err = err
			min_w = w
		w = np.array([X_train_2[i, 1], 1, -1])
		err = E_in_g(w, u)[0]
		if  err <= min_err:
			min_err = err
			min_w = w
	return min_w, min_err

u_now = np.ones(data_train.shape[0]) / X_train_1.shape[0]
w_rec = []
u_rec = []
E_in_g_rec = []
E_in_G_rec = []
E_out_rec = []
alpha_rec = []
for ti in range(300):
	w_get, E_in = choose_w(u_now)
	E_in = np.sum(E_in_g(w_get, u_now)[1])
	w_rec.append(w_get)
	u_rec.append(np.sum(u_now))

	u_now, alpha_now = update(w_get, u_now)

	alpha_rec.append(alpha_now)
	E_in_g_rec.append(E_in / X_train_1.shape[0])
	E_out_rec.append(E_out_G(w_rec, alpha_rec))
	E_in_G_rec.append(E_in_G(w_rec, alpha_rec))
	
	print(u_rec[-1], E_in_g_rec[-1], E_in_G_rec[-1], E_out_rec[-1])
	print('finish', ti)
	

plt.plot(range(300), E_in_g_rec)
plt.xlabel('t')
plt.ylabel('$E_{in}(g_t)$')
plt.title('P13 t v.s. $E_{in}(g_t)$')
plt.legend()
plt.savefig('E_in_g.png')

plt.figure()
plt.plot(range(300), E_out_rec)
plt.xlabel('t')
plt.ylabel('$E_{out}(G_t)$')
plt.title('P16 t v.s. $E_{out}(G_t)$')
plt.legend()
plt.savefig('E_out_G.png')

plt.figure()
plt.plot(range(300), E_in_G_rec)
plt.xlabel('t')
plt.ylabel('$E_{in}(G_t)$')
plt.title('P14 t v.s. $E_{in}(G_t)$')
plt.legend()
plt.savefig('E_in_GG.png')

plt.figure()
plt.plot(range(300), u_rec)
plt.xlabel('t')
plt.ylabel('$U_t$')
plt.title('P15 t v.s. $U_t$')
plt.legend()
plt.savefig('U_t.png')





