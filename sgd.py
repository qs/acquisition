import numpy as np
from sklearn import linear_model
import csv
from decimal import Decimal
from random import randint

def read_data():
    train_data = []
    result = []
    with open('Train_contest.csv', 'r') as file:
        for row in csv.reader(file):
            data = row[0].split("; ")
            result.append(int(data[-2]))
            train_data.append([float(0 if i == 'NaN' else i) for i in data[:-3]])
    return train_data, result


train_data, result = read_data()

X = np.array(train_data)
Y = np.array(result)

clf = linear_model.SGDClassifier()
clf.fit(X, Y)

'''
SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
        fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
        loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
        random_state=None, rho=None, shuffle=False,
        verbose=0, warm_start=False)
'''

for i in range(30):
    index = randint(0, len(result))
    print(clf.predict([ train_data[index] ])), result[index]

match = 0.0
for i in range(len(result)):
    if clf.predict([ train_data[i] ]) == result[i]:
        match += 1

print match, len(result), match / len(result)