import numpy as np
import csv
from decimal import Decimal
from datetime import datetime
import math

from collections import Counter, OrderedDict
from sklearn import metrics

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA


def load_data():
    lst_fin_names = ['Cash and cash equivalents', 'Inventories', 'Total Current Assets', 'Total Current Liabilities', 'Total Assets', 'Property, Plant and Equipment, Net', 'Goodwill', 'Short-Term Debt', 'Long-Term Debt', 'Net Debt', 'Total Liabilities', 'Depreciation  and amortization', 'CAPEX', 'Net Sales', 'Gross Margin', 'EBITDA', 'Dividend yield', 'Market Capitalization', 'Gross Income', 'Financial Costs', 'Net Income', 'Book Value', 'Free Cash Flow']
    lst_names = ["%s %s" % (f, y) for f in lst_fin_names for y in range(1994, 2015)]
    result_data = []
    with open('Train_contest.csv', 'r') as file:
        for row in csv.reader(file):
            data = row[0].split("; ")
            data_row = OrderedDict(zip(lst_names, [float(i if i != 'NaN' else 0) for i in data[:-3]]))
            # calculated values
            for y in range(1994, 2015):
                data_row['EV %s' % y] = data_row['Market Capitalization %s' % y] + data_row['Net Debt %s' % y]
                data_row['EV / EBITDA %s' % y] = (data_row['EV %s' % y] / data_row['EBITDA %s' % y]) if data_row['EBITDA %s' % y] != 0 else 0
                data_row['EV / Sales %s' % y] = (data_row['EV %s' % y] / data_row['Net Sales %s' % y]) if data_row['Net Sales %s' % y] != 0 else 0
                data_row['Net Debt / EBITDA %s' % y] = (data_row['Net Debt %s' % y] / data_row['EBITDA %s' % y]) if data_row['EBITDA %s' % y] != 0 else 0
                data_row['CAPEX / Net Sales %s' % y] = (data_row['CAPEX %s' % y] / data_row['Net Sales %s' % y]) if data_row['Net Sales %s' % y] != 0 else 0
                data_row['Interest Coverage ratio %s' % y] = (data_row['Financial Costs %s' % y] / (data_row['EBITDA %s' % y] - data_row['Depreciation  and amortization %s' % y])) if (data_row['EBITDA %s' % y] - data_row['Depreciation  and amortization %s' % y]) != 0 else 0
                data_row['Debt / Equity %s' % y] = ((data_row['Net Debt %s' % y] + data_row['Cash and cash equivalents %s' % y]) / (data_row['Total Assets %s' % y] - data_row['Total Liabilities %s' % y])) if (data_row['Total Assets %s' % y] - data_row['Total Liabilities %s' % y]) != 0 else 0
                data_row['Total Current Assets / Total Assets %s' % y] = (data_row['Total Current Assets %s' % y] / data_row['Total Assets %s' % y]) if data_row['Total Assets %s' % y] != 0 else 0
            data_row['Sector'] = data[-3]
            data_row['Result'] = bool(int(data[-2]))
            data_row['Result date'] = datetime.strptime(data[-1], "%d.%m.%Y") if data_row['Result'] else None
            result_data.append(data_row)
    return result_data

def compute_ndcg(ans, ideal):
    #ans = sorted(ans, reverse=True)
    #ideal = sorted(ideal, reverse=True)
    ans_summ = sum([ (1.0 / math.log(p + 2, 2)) if v else 0 for p, v in enumerate(ans)])
    ideal_summ = sum([ (1.0 / math.log(p + 2, 2)) if v else 0 for p, v in enumerate(ideal)])
    return ans_summ / ideal_summ

def print_metrics(X, Y, pred):
    # metrics
    fpr, tpr, thresholds = metrics.roc_curve(Y, pred)
    print ' * auc: ', metrics.auc(fpr, tpr)
    y_bin = np.array([bool(i) for i in Y])
    p_bin = np.array([bool(i) for i in pred])
    print ' * precision: ', metrics.precision_score(y_bin, p_bin)
    print ' * recall: ', metrics.recall_score(y_bin, p_bin)
    print ' * ndcg: ', compute_ndcg(p_bin, y_bin)
    print 

def build_classifier(clf, X_train, X_test, y_train, y_test):
    print '### ' + clf.__class__.__name__
    clf.fit(X_train, y_train)
    pred = np.array([])
    for x in X_test:
        pred = np.append(pred, bool(round(clf.predict(x))))
    print_metrics(X_test, y_test, pred)


# preparing data
data = load_data()
X = np.array([[v for k, v in i.items() if k not in ['Result', 'Result date', 'Sector']] for i in data])
Y = np.array([i['Result'] for i in data])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# classifier
# http://scikit-learn.org/stable/auto_examples/plot_classifier_comparison.html

classifiers = [
    SGDClassifier(),
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    AdaBoostClassifier(),
    AdaBoostClassifier(base_estimator=SVC(kernel="linear", C=0.025), algorithm='SAMME'),
    GradientBoostingClassifier(),
    GaussianNB(),
]

for classifier in classifiers:
    build_classifier(classifier, X_train, X_test, y_train, y_test)
