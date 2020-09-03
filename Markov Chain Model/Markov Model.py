# Using script: "Memento"
# A lot of help from stackoverflow and geeksforgeeks!!

import math
import re
import string
from collections import Counter
from itertools import product
import time
from random import choices


def file_print(s, name, multi=False):
    f = open(name + ".txt", "w")
    f.write(s)
    f.close()


script = ""
letters = ''
bigram = {}
bigram_prob = {}
bigram_prob_smooth = {}
trigram = {}
trigram_prob_smooth = {}

def ngram(n):
    global script, letters
    # all possible n-grams
    d = dict.fromkeys([''.join(i) for i in product(letters, repeat=n)], 0)      # cartesian product
    # update counts
    d.update(Counter([''.join(j) for j in zip(*[script[i:] for i in range(n)])]))
    return d


def gen_bi(c):
    w = [bigram_prob[c + i] for i in letters]
    return choices(letters, weights=w)[0]


def gen_tri(ab):
    w_tri = [trigram_prob_smooth[ab + i] for i in letters]
    return choices(letters, weights=w_tri)[0]


def gen_sen(c, num):
    res = c + gen_bi(c)
    for i in range(num - 2):
        if bigram[res[-2:]] == 0:
            t = gen_bi(res[-1])
        else:
            t = gen_tri(res[-2:])
        res += t
    return res


def print_dict(dic, n=1):
    key = dic.keys()
    s = ""
    i = 0
    for k in key:
        i += 1
        s += str(dic[k]) + (',' if i < 27 else ('' if n == 1 else '\n'))
        if i == 27:
             i = 0
    return s


if __name__ == '__main__':
    start_time = time.time()
    print("1. Memento")
    with open('Memento.txt', 'r') as file:
        script = file.read().replace('\n', '')

    script = script.lower()                             # convert to lowercase
    script = re.sub(r'[^a-zA-Z ]+', '', script)         # keep only letters and spaces
    script = " ".join(script.split())                   # remove multiple consecutive spaces

    letters = ' ' + string.ascii_lowercase
    uni = Counter(script)
    unigram = {lt: round((uni[lt] / len(script)), 4) for lt in letters}
    # uni_list = [unigram[lt] for lt in letters]

    file_print(print_dict(unigram), "q2")

    '''
    for bigram transition probs, I am using the following method:
    To find: P(A|B) = P(AB)/P(B)    
    '''
    bigram = ngram(2)
    bigram_prob = {i: (bigram[i] / uni[i[0]]) for i in bigram}
    file_print(print_dict(bigram_prob, 2), "q3")

    bigram_prob_smooth = {i: math.ceil(((bigram[i] + 1) / (uni[i[0]] + 27))*(10**4))/(10**4) for i in bigram}
    file_print(print_dict(bigram_prob_smooth, 2), "q4")

    trigram = ngram(3)
    trigram_prob_smooth = {i: math.ceil(((trigram[i] + 1) / (bigram[i[:2]] + 27))*(10**4)) / (10**4) for i in trigram}

    # sntc = ''
    # for ch in string.ascii_lowercase:
    #     sntc += gen_sen(ch, 1000) + "\n"
    #
    # print(file_print(sntc, "q5"))

    with open('script.txt', encoding='utf-8') as f:
        young = f.read()

    young_uni = Counter(young)
    young_unigram = {lt: round((young_uni[lt] / len(young)), 4) for lt in letters}

    file_print(print_dict(young_unigram), "q7")

    # young_unigram = D and unigram = not D
    # P{D|A} = P{A|D}*P{D}/(P{A|D}*P{D} + P{A|not D}*P{not D})

    post = {lt: round(((young_unigram[lt]) / (young_unigram[lt] + unigram[lt])), 4) for lt in letters}
    post_movie = {lt: round(((unigram[lt]) / (young_unigram[lt] + unigram[lt])), 4) for lt in letters}
    file_print(print_dict(post), "q8")

    post_yw_list = {lt: math.log(post[lt]) for lt in letters}
    post_mv_list = {lt: math.log(post_movie[lt]) for lt in letters}

    f1 = open('q5.txt', 'r')
    Lines = f1.readlines()
    labels = ""
    i = 0
    for line in Lines:
        line = line.rstrip("\n")
        i += 1
        line_sum_yw = 0
        line_sum = 0
        for ele in line:
            line_sum += post_mv_list[ele]
            line_sum_yw += post_yw_list[ele]
        if line_sum > line_sum_yw:
            labels += '0' + (',' if i < 26 else '')
        else:
            labels += '1' + (',' if i < 26 else '')
    print("9. ", labels)
    print("--- %s seconds ---" % (time.time() - start_time))
