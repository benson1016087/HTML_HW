import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('hw4_nolabel_train.dat')
data_num = X.shape[0]
np.random.seed(902066)
def min_type(centers, point):
	s = np.sum(((centers - point)**2), axis = 1)
	return np.argmin(s)

def cal_E_in(centers):
	err = 0
	for i in range(data_num):
		types = min_type(centers, X[i])
		err += (X[i] - centers[types]).T.dot(X[i] - centers[types])
	return err / data_num

k_rec = [2, 4, 6, 8, 10]
means_rec = []
var_rec = []
for k in k_rec:
	E_in_total = 0
	E_in2_total = 0
	for times in range(1, 501):
		center = []
		choosed = np.zeros(data_num)
		for i in range(k):
			num = 1e10
			while 1:
				num = np.random.randint(data_num)
				if choosed[num] == 0:
					choosed[num] += 1
					break;
			center.append(X[num])
		center = np.asarray(center)
		last_center = np.zeros([center.shape[0], center.shape[1]])
		t = 0
		while np.sum(np.abs(last_center - center)) > 1e-30:
			last_center = np.copy(center)
			rec_vec = np.zeros([k, X.shape[1]])
			rec_num = np.zeros(k)
			for i in range(data_num):
				index = min_type(center, X[i])
				rec_vec[index] += X[i]
				rec_num[index] += 1
			center = np.copy(rec_vec / (rec_num.reshape(-1, 1) + 1e-30))
			t += 1
		E_in = cal_E_in(center)
		E_in_total += E_in
		E_in2_total += (E_in*E_in)
		if times % 50 == 0:
			print('finishing {}, t = {}'.format(times, t))
	means_rec.append(E_in_total / 500)
	var_rec.append((E_in2_total / 500) - ((E_in_total / 500)**2))
	print(E_in_total, E_in2_total)
print(means_rec)
print(var_rec)
# Prob 15
plt.figure()
plt.plot(k_rec, means_rec)
plt.xlabel('k')
plt.ylabel('$Means(E_{in})$')
plt.legend()
plt.savefig('Prob_15.png')
print('finishing prob_15')

# Prob 16
plt.figure()
plt.plot(k_rec, var_rec)
plt.xlabel('k')
plt.ylabel('$Var(E_{in})$')
plt.legend()
plt.savefig('Prob_16.png')
print('finishing prob_16')
