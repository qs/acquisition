import numpy as np
from sklearn import linear_model, cross_validation
import csv
from decimal import Decimal
from random import randint
from collections import Counter


def read_data():
    train_data = []
    result = []
    cats = {}
    with open('Train_contest.csv', 'r') as file:
        for row in csv.reader(file):
            data = row[0].split("; ")
            result_row = int(data[-2])
            result.append(result_row)
            data_row = [float(0 if i == 'NaN' else i) for i in data[:-3]]
            train_data.append(data_row)
            cat_name = data[-3]
            if cat_name in cats:
                cats[cat_name]['data'].append(data_row)
                cats[cat_name]['result'].append(result_row)
            else:
                cats[cat_name] = {'data': [data_row, ], 'result': [result_row, ]}
    return train_data, result, cats


train_data, result, cats = read_data()

X = np.array(train_data)
Y = np.array(result)

clf = linear_model.SGDClassifier()
clf.fit(X, Y)

scores = cross_validation.cross_val_score(clf, X, Y, cv=20)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clfs = {}
scores = {}
for k,v in cats.items():
    print k, len(v['result'])
    cntr = Counter(v['result'])
    if  len([True for v1 in cntr.values() if v1 > 5]) < 2:
        print 'too small values'
        continue
    clfs[k] = linear_model.SGDClassifier()
    X = np.array(v['data'])
    Y = np.array(v['result'])
    clfs[k].fit(X, Y)
    scores[k] = cross_validation.cross_val_score(clfs[k], X, Y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores[k].mean(), scores[k].std() * 2))


'''
SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
        fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
        loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
        random_state=None, rho=None, shuffle=False,
        verbose=0, warm_start=False)
'''