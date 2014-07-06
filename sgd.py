import numpy as np
from sklearn import linear_model, cross_validation
import csv
from decimal import Decimal
from datetime import datetime
from collections import Counter
from sklearn import metrics


def load_data():
    lst_fin_names = ['Cash and cash equivalents', 'Inventories', 'Total Current Assets', 'Total Current Liabilities', 'Total Assets', 'Property, Plant and Equipment, Net', 'Goodwill', 'Short-Term Debt', 'Long-Term Debt', 'Net Debt', 'Total Liabilities', 'Depreciation  and amortization', 'CAPEX', 'Net Sales', 'Gross Margin', 'EBITDA', 'Dividend yield', 'Market Capitalization', 'Gross Income', 'Financial Costs', 'Net Income', 'Book Value', 'Free Cash Flow']
    lst_names = ["%s %s" % (f, y) for f in lst_fin_names for y in range(1994, 2015)]
    result_data = []
    with open('Train_contest.csv', 'r') as file:
        for row in csv.reader(file):
            data = row[0].split("; ")
            data_row = dict(zip(lst_names, [Decimal(i) for i in data[:-3]]))
            data_row['Sector'] = data[-3]
            data_row['Result'] = bool(int(data[-2]))
            data_row['Result date'] = datetime.strptime(data[-1], "%d.%m.%Y") if data_row['Result'] else None
            # calculated values
            for y in range(1994, 2015):
                data_row['EV %s' % y] = data_row['Market Capitalization %s' % y] + data_row['Net Debt %s' % y]
                data_row['EV / EBITDA %s' % y] = data_row['EV %s' % y] / data_row['EBITDA %s' % y]
                data_row['EV / Sales %s' % y] = data_row['EV %s' % y] / data_row['Net Sales %s' % y]
                data_row['Net Debt / EBITDA %s' % y] = data_row['Net Debt %s' % y] / data_row['EBITDA %s' % y]
                data_row['CAPEX / Net Sales %s' % y] = data_row['CAPEX %s' % y] / data_row['Net Sales %s' % y]
                data_row['Interest Coverage ratio %s' % y] = data_row['Financial Costs %s' % y] / (data_row['EBITDA %s' % y] - data_row['Depreciation  and amortization %s' % y])
                data_row['Debt / Equity %s' % y] = (data_row['Net Debt %s' % y] + data_row['Cash and cash equivalents %s' % y]) / (data_row['Total Assets %s' % y] - data_row['Total Liabilities %s' % y])
                data_row['Total Current Assets / Total Assets %s' % y] = data_row['Total Current Assets %s' % y] / data_row['Total Assets %s' % y]
            result_data.append(data_row)
    return result_data

exit()

train_data, result, cats = read_data()

X = np.array(train_data)
Y = np.array(result)

cntr = Counter(result)
print cntr
exit()

clf = linear_model.SGDClassifier()
clf.fit(X, Y)

pred = np.array([])
for x in X:
    pred = np.append(pred, x[0])

fpr, tpr, thresholds = metrics.roc_curve(Y, pred)

print metrics.auc(fpr, tpr)

y_bin = np.array([bool(i) for i in Y])
p_bin = np.array([bool(i) for i in pred])

print metrics.precision_score(y_bin, p_bin)
print metrics.recall_score(y_bin, p_bin)

#print metrics.precision_score(y_bin, p_bin, average='micro')
#print metrics.precision_score(y_bin, p_bin, average='weighted')

#scores = cross_validation.cross_val_score(clf, X, Y, cv=20)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

exit()

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