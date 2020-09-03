import copy

import numpy as np


class Node:
    def __init__(self, split_val):
        self.fea, self.threshold = split_val
        self.left = None
        self.right = None
        self.size = 0
        self.benign = 0
        self.malign = 0

    def set_threshold(self, t):
        self.threshold = t


def entropy(data):
    count = len(data)
    p0 = sum(b[-1] == 2 for b in data) / count
    if p0 == 0 or p0 == 1: return 0
    p1 = 1 - p0
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def infogain(data, f, t):
    left = data[data[:, f - 1] <= t]
    right = data[data[:, f - 1] > t]
    if len(left) == 0 or len(right) == 0: return 0
    return entropy(data) - (len(left) / len(data) * entropy(left) + len(right) / len(data) * entropy(right))


def get_best_split(data, fea):
    b = data[:, -1].tolist().count(2)
    m = data[:, -1].tolist().count(4)
    if b == len(data):
        return 2, None
    elif m == len(data):
        return 4, None
    infos = []  # (f, t, max_info_gain)
    for j in range(len(fea)):
        info = []
        for i in range(1, 11):
            info.append(infogain(data, fea[j], i))
        infos.append([fea[j], info.index(max(info)) + 1, max(info)])
    inf = np.array(infos)[:, -1]
    max_inf = max(inf.tolist())
    if max_inf == 0:
        if b > m:
            return 2, None
        else:
            return 4, None
    return infos[inf.tolist().index(max_inf)][0], infos[inf.tolist().index(max_inf)][1]


def split(data, node):
    fea, thresh = node.fea, node.threshold
    d1 = data[data[:, fea - 1] <= thresh]
    d2 = data[data[:, fea - 1] > thresh]
    return d1, d2


def create_tree(data, node, fea):
    d1, d2 = split(data, node)
    f1, t1 = get_best_split(d1, fea)
    f2, t2 = get_best_split(d2, fea)
    # print("Adding branches ", f1, f2, " to node ", node.fea)
    if t1 is None:
        node.left = Node((f1, t1))
        node.left.size = len(d1)
    else:
        node.left = Node((f1, t1))
        node.left.benign = d1.tolist().count(2)
        node.left.malign = d1.tolist().count(4)
        create_tree(d1, node.left, fea)
    if t2 is None:
        node.right = Node((f2, t2))
        node.right.size = len(d2)
    else:
        node.right = Node((f2, t2))
        node.right.benign = d2.tolist().count(2)
        node.right.malign = d2.tolist().count(4)
        create_tree(d2, node.right, fea)
    return node


def get_label(node, row):
    f, t = node.fea, node.threshold
    label = 0
    if t is not None:
        test_f = row[f - 1]
        if test_f <= t:
            label = get_label(node.left, row)
        else:
            label = get_label(node.right, row)
    else:
        label = f
    return label


def predict(node, data, q=False, num=""):
    s = ""
    i = 0
    for row in data:
        i += 1
        l = get_label(node, row)
        s += str(l) + (", " if i != len(data) else "")
    # print(s)
    if q:
        file = open("q" + num + ".txt", "w")
        file.write(s)
        file.close()
    return s


def accuracy(node, val_set):
    s = predict(node, val_set)
    labels = np.fromstring(s, dtype=int, sep=",")
    l1 = labels[0:100]
    l2 = labels[100:]
    return (l1.tolist().count(2) + l2.tolist().count(4)) / 200


root_p = None


def pruning(curr, val_set, depth=0):
    if curr.threshold is None:
        return None
    lval = pruning(curr.left, val_set, depth + 1)
    rval = pruning(curr.right, val_set, depth + 1)

    if lval and rval is None and depth >= 8:
        prev_acc = accuracy(root_p, val_set)
        temp_t = curr.threshold
        temp_f = curr.fea
        curr.threshold = None
        curr.fea = 2 if curr.left.size > curr.right.size else 4
        acc = accuracy(root_p, val_set)
        if acc < prev_acc:
            curr.threshold = temp_t
            curr.fea = temp_f
    if depth == 8:
        curr.threshold = None
        curr.fea = 2 if curr.benign > curr.malign else 4
        return curr.threshold
    return curr.threshold


def preOrder(curr, tabcount=0):
    space = " "
    if curr.threshold is not None:
        l = curr.left
        r = curr.right
        s = "\n" + tabcount * space + "if (x" + str(curr.fea) + " <= " + str(curr.threshold) + ")"
        tabcount += 1
        s = s + tabcount * space + preOrder(l, tabcount)
        tabcount -= 1
        s = s + "\n" + tabcount * space + "else"
        tabcount += 1
        s = s + tabcount * space + preOrder(r, tabcount)
    else:
        s = "return " + str(curr.fea)

    return s


def maxDepth(node):
    if node.threshold is None:
        return 0

    else:
        # Compute the depth of each subtree
        lDepth = maxDepth(node.left)
        rDepth = maxDepth(node.right)
        # Use the larger one
        return max(lDepth, rDepth) + 1


def p2_part1(A):
    y = A[:, -1]

    ben = y.tolist().count(2)
    mal = y.tolist().count(4)
    print("1. ", (ben, mal))
    H_y = entropy(A)
    print("2. ", H_y)

    info = []
    for i in range(1, 11):
        info.append(infogain(A, 9, i))
    threshold = info.index(max(info)) + 1
    pos = A[A[:, 9 - 1] > threshold]
    neg = A[A[:, 9 - 1] <= threshold]
    ab = 0
    bb = 0
    am = 0
    bm = 0
    for i in pos:
        if i[-1] == 2:
            ab += 1
        else:
            am += 1
    for i in neg:
        if i[-1] == 2:
            bb += 1
        else:
            bm += 1
    print("3. ", (ab, bb, am, bm))
    print("4. ", infogain(A, 9, threshold))


if __name__ == '__main__':
    with open('breast-cancer-wisconsin.data', 'r') as f:
        a = [l.strip('\n').split(',') for l in f if '?' not in l]
    A = np.array(a).astype(int)
    p2_part1(A)
    global root
    feat = [3, 10, 9, 8, 7]
    s1 = get_best_split(A, feat)
    root = Node(s1)
    root = create_tree(A, root, feat)

    file1 = open("q5.txt", "w")
    file1.write(preOrder(root))
    file1.close()

    # print(preOrder(root))
    print("6. Max Depth: ", maxDepth(root))

    with open('test.txt', 'r') as f:
        a = [l.strip('\n').split(',') for l in f if '?' not in l]
    test = np.array(a).astype(int)
    predict(root, test, q=True, num="7")

    # root.parent = root
    root_p = copy.deepcopy(root)
    mnb = pruning(root, test)
    print("Pruned Depth: ", maxDepth(root))

    file1 = open("q8.txt", "w")
    file1.write(preOrder(root))
    file1.close()
    print("Accuracy of un-pruned tree: ", accuracy(root_p, test))
    predict(root_p, test, q=True, num="9")
    print("Accuracy of pruned tree: ", accuracy(root, test))
