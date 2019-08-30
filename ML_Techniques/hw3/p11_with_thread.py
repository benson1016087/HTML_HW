import numpy as np
import matplotlib.pyplot as plt
import threading

def gini_impurity(posi, nega):
	if posi + nega == 0:
		return 0
	return 1 - (posi / (posi + nega)) ** 2 - (nega / (posi + nega)) ** 2

def find_best_theta(in_sorted):
	num = in_sorted[0].shape[0]
	min_val = 1e5
	min_i = -1
	min_n = -1
	for n in range(2): # dimension
		data = in_sorted[n]
		label = data[:, 2].astype('int32')
		r = [np.sum(label == 1), np.sum(label == -1)] # posi, nega
		l = [0, 0]
		if r[0] == 0:
			return -1, -1, -1
		if r[1] == 0:
			return 1, -1, -1
		for i in range(num):
			add = 0
			if label[i] == -1: 
				add = 1
			l[add] += 1
			r[add] -= 1
			gini_im = (l[0] + l[1])*gini_impurity(l[0], l[1]) + (r[0] + r[1])*gini_impurity(r[0], r[1])
			if gini_im < min_val:
				min_val = gini_im
				min_i = i # cut between i and i+1
				min_n = n
	return (min_n, (in_sorted[min_n][min_i, min_n] + in_sorted[min_n][min_i+1, min_n])/2), in_sorted[min_n][:min_i+1, :], in_sorted[min_n][min_i+1:]


def build_tree(inp, max_H):
	if max_H == 0:
		num = [np.sum(inp == 1), np.sum(inp == -1)]
		if num[1] > num[0]:
			return -1
		else:
			return 1
	arg_sort = np.argsort(inp, axis = 0)
	X_train = [inp[arg_sort[:, 0]], inp[arg_sort[:, 1]]]
	min_Node, l_data, r_data = find_best_theta(X_train) # (n, theta)
	if min_Node == -1 or min_Node == 1:
		return min_Node
	L_node = build_tree(l_data, max_H - 1)
	R_node = build_tree(r_data, max_H - 1)
	return min_Node, L_node, R_node

def output_the_tree(inp, tree):
	lst = []
	for i in inp:
		rec_str = ''
		Node = tree
		while Node != -1 and Node != 1:
			cl_n = Node[0][0]
			cl_v = Node[0][1]
			if i[cl_n] < cl_v:
				Node = Node[1]
				rec_str += 'L'
			else:
				Node = Node[2]
				rec_str += 'R'
		lst.append(rec_str)
	print(lst)

def Error(inp, tree):
	total_num = inp.shape[0]
	err = 0
	for i in inp:
		Node = tree
		while Node != -1 and Node != 1:
			cl_n = Node[0][0]
			cl_v = Node[0][1]
			if i[cl_n] < cl_v:
				Node = Node[1]
			else:
				Node = Node[2]
		if i[2] != Node:
			err += 1
	return err / total_num

def Error_forest(pred, ans):
	total_pred = np.sum(pred, axis = 1)
	res = np.sign(total_pred + 0.001) # judge 0 to 1
	err = (res != ans)
	return np.sum(err) / err.shape[0]
	

def make_prediction(inp, forest):
	tree_num = len(forest)
	rec = [] 
	for i in inp:
		i_pred = []
		for tree in forest:
			Node = tree
			while Node != -1 and Node != 1:
				cl_n = Node[0][0]
				cl_v = Node[0][1]
				if i[cl_n] < cl_v:
					Node = Node[1]
				else:
					Node = Node[2]
			jdg = Node
			i_pred.append(jdg)
		rec.append(i_pred)
	ret = np.asarray(rec)
	return ret



# loading data
train_data = np.loadtxt('hw3_train.dat')
test_data = np.loadtxt('hw3_test.dat')

# construct decicion tree
Tree = build_tree(train_data, 1000)


# prob_11
output_the_tree(train_data, Tree)

# prob_12
err_in = Error(train_data, Tree)
err_out = Error(test_data, Tree)
print('E_in = {}, E_out = {}'.format(err_in, err_out))

# prob 13
Height = 0
err_in_rec = []
err_out_rec = []
while True:	
	pruned_Tree = build_tree(train_data, Height)
	err_in = Error(train_data, pruned_Tree)
	err_out = Error(test_data, pruned_Tree)
	err_in_rec.append(err_in)
	err_out_rec.append(err_out)
	Height += 1
	if err_in == 0:
		break;

plt.plot(range(Height), err_in_rec, label = '$E_{in}$')
plt.plot(range(Height), err_out_rec, label = '$E_{out}$')
plt.xlabel('tree height')
plt.ylabel('0/1 Error')
plt.title('tree height v.s. 0/1 Error')
plt.legend()
plt.savefig('Prob_13.png')

# prob 14 ~ 16
bagging_num = int(0.8 * train_data.shape[0])
forest = []
err_in_rec = []
err_in_forest_rec = []
err_out_forest_rec = []

def job():
	for ti in range(1000):
		bagged_data = []
		for times in range(bagging_num):
			a = np.random.randint(train_data.shape[0]);
			bagged_data.append(train_data[a])
		bagged_data = np.asarray(bagged_data)
		Tree = build_tree(bagged_data, 1000)

		err_in = Error(train_data, Tree)
		err_in_rec.append(err_in)

		forest.append(Tree)

		if ti % 10 == 9:
			print('finishing {} times'.format(ti+1))

t = []
for i in range(30):
	t.append(threading.Thread(target = job))
	t[i].start()
for i in t:
	i.join()
print('finishing building, tree_num = {}'.format(len(forest)))


train_decision = make_prediction(train_data, forest)
test_decision = make_prediction(test_data, forest)
print(train_decision.shape, test_decision.shape)
for ti in range(1, 30001):
	err_in_forest = Error_forest(train_decision[:, :ti].reshape(-1, ti), train_data[:, 2])
	err_out_forest = Error_forest(test_decision[:, :ti].reshape(-1, ti), test_data[:, 2])

	err_in_forest_rec.append(err_in_forest)
	err_out_forest_rec.append(err_out_forest)

	if ti % 10 == 9:
		print('finishing recording err {} times'.format(ti+1))


# Notice that prob_14~16 need to run this code three times to produce three pictures
# Have to annotate others for each time

plt.hist(err_in_rec, bins = 20)
plt.xlabel('t')
plt.ylabel('$E_{in}(g_{t})$')
plt.title('t v.s. $E_{in}(g_{t})$')
plt.legend()
plt.savefig('prob_14.png')
print('finishing prob_14')


plt.plot(range(30000), err_in_forest_rec)
plt.xlabel('t')
plt.ylabel('$E_{in}(G_{t})$')
plt.title('t v.s. $E_{in}(G_{t})$')
plt.legend()
plt.savefig('prob_15.png')
print('finishing prob_15')


plt.plot(range(30000), err_out_forest_rec)
plt.xlabel('t')
plt.ylabel('$E_{out}(G_{t})$')
plt.title('t v.s. $E_{out}(G_{t})$')
plt.legend()
plt.savefig('prob_16.png')
print('finishing prob_16')

