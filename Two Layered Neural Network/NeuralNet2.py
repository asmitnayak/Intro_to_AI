import numpy as np
from scipy.special import expit
from sklearn.utils import shuffle

# Training data:
data = open('mnist_train.csv', "r")
X = []
y = []
for line in data:
    label = int(line.split(',')[0].strip())
    if label == 8 or label == 1:
        y.append(0 if label == 8 else 1)
        r = line.split(',')[1:]
        x = []
        for val in r:
            x.append(int(val.strip()))
        X.append(x)

y = np.array(y)
X = np.array(X)
X = X / 255
X_actual = X

test = open('test.txt', 'r')
X_test = []
for line in test:
    r = line.split(',')[:]
    x_t = []
    for val in r:
        x_t.append(int(val.strip()))
    X_test.append(x_t)

X_test = np.array(X_test)
X_test = X_test.T

n = X.shape[0]

max_iter = 500
alpha = 1e-1
b = np.random.rand()

act = []

cost = []
eps = 1e-6

num_train = len(y)
train_x = X_actual  # np.hstack((X_actual, np.ones(num_train).reshape(-1,1)))
num_input_units = train_x.shape[1]
num_hidden_units = int (num_input_units/2)
prev_error = 0
bias_h = np.ones(num_hidden_units).reshape((392, 1))
bias_o = 1
cb1 = np.zeros((392, 1))
cb2 = 0
cw1 = np.zeros((num_hidden_units, num_input_units))
cw2 = np.zeros((1, 392))
w1 = np.random.uniform(low=-1, high=1, size=(num_hidden_units, num_input_units)) #392*784
w2 = np.random.uniform(low=-1, high=1, size=(1, num_hidden_units)) # 1 * 392
for epoch in range(10):
    print("epoch: =", epoch)
    y_rand = y
    act = []
    X, y_rand = shuffle(train_x, y_rand)
    for ind in range(num_train):
        yi = y[ind]
        xi = train_x[ind].reshape(train_x[ind].shape[0], 1)
        z1 = np.matmul(w1, xi) + bias_h
        a1 = expit(z1)
        z2 = np.matmul(w2, a1) + bias_o
        a2 = np.squeeze(expit(z2))

        cw2 = (a2 - yi) * a2 * (1 - a2) * a1
        cb2 = (a2 - yi) * a2 * (1 - a2)

        t1 = cb2 * w2.T
        a11 = np.ones(z1.shape[0]).reshape(z1.shape[0], 1) - a1
        t2 = np.multiply(a1, a11)
        cw1 = np.multiply(t1, t2) @ xi.T
        cb1 = np.multiply(t1, t2)

        w1 = w1 - alpha * cw1
        bias_h = bias_h - alpha * cb1
        w2 = w2 - alpha * cw2.T
        bias_o = bias_o - alpha * cb2

        act.append(a2)

    acc = sum((np.array(act) > 0.5).astype(int) == y_rand)*100/len(y_rand)
    if acc > 95.0:
        break

s = ""
i = 0
for line in w1.T:
    i = 0
    for w in line:
        format_4f = '%.4f' % w
        i += 1
        s = s + str(format_4f)
        if i != 784:
            s = s + ", "
    s += "\n"
i = 0
for b in bias_h:
    i += 1
    format_4f = '%.4f' % b
    s = s + str(format_4f)
    if i != 784:
        s += ','
file1 = open("q5.txt", "w")
file1.write(s)
file1.close()

s = ""
for w in w2.T:
    format_4f = '%.4f' % w
    s = s + str(format_4f)
    i += 1
    if i != 392:
        s += ','
file1 = open("q6.txt", "w")
file1.write(s)
file1.close()

nn_act = []

for i in range(len(X_test[0])):
    xi = X_test[:, i].reshape((X_test.shape[0], 1))
    z_i_h = (w1 @ xi) + bias_h
    aij = expit(z_i_h)
    z_h_o = (w2 @ aij) + bias_o
    ai = expit(z_h_o)[0][0]
    nn_act.append(ai)

nn_pred = np.round(nn_act)
print(nn_pred)


nn_actual = np.concatenate((np.zeros(100), np.ones(100)))
for i in range(len(nn_actual)):
    if nn_actual[i] != nn_pred[i]:
        p = X_test[:, i]
        feat = ['%.2f' % ele for ele in p]
        s = ""
        i = 0
        for num in feat:
            s = s + str(num)
            i += 1
            if i != len(feat):
                s = s + ", "
        file1 = open("q9.txt", "w")
        file1.write(s)
        file1.close()


format_2f = ['%.2f' % ele for ele in nn_act]
s = ""
i = 0
for num in format_2f:
    s = s + str(num)
    i += 1
    if i != len(format_2f):
         s = s + ", "
file1 = open("q7.txt", "w")
file1.write(s)
file1.close()

format_2f = ['%.0f' % ele for ele in nn_pred]
s = ""
i = 0
for num in format_2f:
    s = s + str(num)
    i += 1
    if i != len(format_2f):
         s = s + ", "
file1 = open("q4.txt", "w")
file1.write(s)
file1.close()