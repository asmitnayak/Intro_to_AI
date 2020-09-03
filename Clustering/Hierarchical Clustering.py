import math

import pandas as pd
import math

def file_print(s, name, multi=False, s1=[], is_string=False):
    if not is_string:
        s = ','.join(map(str, s))
        if multi:
            for ele in s1:
                temp = ','.join(map(str, ele))
                s += "\n" + temp
    f = open(name + ".txt", "w")
    f.write(s)
    f.close()


def euclidean_dist(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))


country_dict = {}


def sld(c1, c2):
    d = []
    for i in c1:
        di = country_dict[i]
        for j in c2:
            dj = country_dict[j]
            d.append(euclidean_dist(di, dj))
    return min(d)


def cld(c1, c2):
    d = []
    for i in c1:
        di = country_dict[i]
        for j in c2:
            dj = country_dict[j]
            d.append(euclidean_dist(di, dj))
    return max(d)


if __name__ == '__main__':
    data = pd.read_csv("time_series_covid19_deaths_global.csv")
    data = data.dropna(axis='columns')
    data = data.groupby(['Country/Region']).sum()
    data = data.drop(columns=['Lat', 'Long'])

    US = list(data.loc['US'].values)
    Can = list(data.loc['Canada'].values)
    file_print(US, "q1", True, [Can])
    US_diff = [t - s for s, t in zip(US, US[1:])]
    Can_diff = [t - s for s, t in zip(Can, Can[1:])]
    file_print(US_diff, "q2", True, [Can_diff])

    for c in list(data.index):
        country = list(data.loc[c].values)
        country = list([i for i in country if i != 0])
        p1 = p2 = p3 = p4 = p5 = 0
        init = 0
        if len(country) > 0:
            init = country[0]
        if init == 0:
            country_dict[c] = [p1, p2, p3, p4]
            continue
        pos = 0
        for i in range(1, len(country)):
            if country[i] / init >= 2.0:
                if init == 1:
                    p1 = i - pos
                    pos = i
                    init = 2
                elif init == 2:
                    p2 = i - pos
                    pos = i
                    init = 4
                elif init == 4:
                    p3 = i - pos
                    pos = i
                    init = 8
                elif init == 8:
                    p4 = i - pos
                    pos = i
                    init = 16
        country_dict[c] = [p1, p2, p3, p4]

    parameters = ""
    for k in country_dict.keys():
        param = country_dict[k]
        param_s = ','.join(map(str, param))
        parameters += param_s + "\n"
    file_print(parameters, "q4", is_string=True)

    k = 8
    if False:
        hac = data
        hac['Cluster'] = [i for i in range(len(data))]
        for _ in range(len(data) - k + 1):  # changed
            print(_)
            dist = float('inf')
            # min_ck = []
            clu = list(set(hac['Cluster'].tolist()))
            # print(clu)
            print("Current number of clusters: ", len(clu))
            if len(clu) <= 8:
                break
            dist_dict = {}
            for i in range(len(clu) - 1):
                clu1 = hac.index[hac['Cluster'] == clu[i]].tolist()
                for j in range(i + 1, len(clu)):
                    clu2 = hac.index[hac['Cluster'] == clu[j]].tolist()
                    if sld(clu1, clu2) in dist_dict.keys():
                        dist_dict[sld(clu1, clu2)] += [clu[i], clu[j]]
                    else:
                        dist_dict[sld(clu1, clu2)] = [clu[i], clu[j]]
            min_cc = min(list(dist_dict.keys()))
            min_ck = list(set(dist_dict[min_cc]))
            smallest = min(min_ck)
            min_ck.remove(smallest)
            for ele in min_ck:
                hac.loc[hac.Cluster == ele, 'Cluster'] = smallest
            print()
        i = 0
        for i in range(len(clu)):
            hac.loc[hac.Cluster == clu[i], 'Cluster'] = i
        s = ','.join(map(str, hac.Cluster.values))
        f = open("q5.txt", "w")
        f.write(s)
        f.close()
        print(hac.Cluster)

    hac = data
    hac['Cluster'] = [i for i in range(len(data))]
    clu = []
    for _ in range(len(data) - k + 1):  # changed
        print(_)
        dist = float('inf')
        min_ck = []
        clu = list(set(hac['Cluster'].tolist()))
        # print(clu)
        if len(clu) <= k:
            break
        print("Current number of clusters: ", len(clu))
        for i in range(len(clu) - 1):
            clu1 = hac.index[hac['Cluster'] == clu[i]].tolist()
            for j in range(i + 1, len(clu)):
                clu2 = hac.index[hac['Cluster'] == clu[j]].tolist()
                # print(i,j, sld(clu1, clu2))
                if cld(clu1, clu2) < dist:
                    dist = cld(clu1, clu2)
                    min_ck = [clu[i], clu[j]]
        print("min_ck: ", min_ck)
        print()
        hac.loc[hac.Cluster == max(min_ck), 'Cluster'] = min(min_ck)
    i = 0
    for i in range(len(clu)):
        hac.loc[hac.Cluster == clu[i], 'Cluster'] = i
    s = ','.join(map(str, hac.Cluster.values))
    f = open("q6.txt", "w")
    f.write(s)
    f.close()

    print("Done")

