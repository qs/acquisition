import numpy as np
from sklearn import cross_validation
import csv
from decimal import Decimal
from datetime import datetime
from collections import Counter, OrderedDict
from sklearn import metrics

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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

def print_metrics(X, Y, pred):
    # metrics
    fpr, tpr, thresholds = metrics.roc_curve(Y, pred)
    print ' * auc: ', metrics.auc(fpr, tpr)
    y_bin = np.array([bool(i) for i in Y])
    p_bin = np.array([bool(i) for i in pred])
    print ' * precision: ', metrics.precision_score(y_bin, p_bin)
    print ' * recall: ', metrics.recall_score(y_bin, p_bin)
    print 

def build_classifier(clf, X, Y):
    print '### ' + clf.__class__.__name__
    clf.fit(X, Y)
    pred = np.array([])
    for x in X:
        pred = np.append(pred, bool(round(clf.predict(x))))
    print_metrics(X, Y, pred)


# preparing data
data = load_data()
X = np.array([[v for k, v in i.items() if k not in ['Result', 'Result date', 'Sector']] for i in data])
Y = np.array([i['Result'] for i in data])

# classifier
# http://scikit-learn.org/stable/auto_examples/plot_classifier_comparison.html

classifiers = [
    SGDClassifier(),
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    #LDA(),
    #QDA()
]

for classifier in classifiers:
    build_classifier(classifier, X, Y)
