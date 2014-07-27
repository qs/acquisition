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


def load_data(fname):
    lst_fin_names = ['Cash and cash equivalents', 'Inventories', 'Total Current Assets', 'Total Current Liabilities', 'Total Assets', 'Property, Plant and Equipment, Net', 'Goodwill', 'Short-Term Debt', 'Long-Term Debt', 'Net Debt', 'Total Liabilities', 'Depreciation  and amortization', 'CAPEX', 'Net Sales', 'Gross Margin', 'EBITDA', 'Dividend yield', 'Market Capitalization', 'Gross Income', 'Financial Costs', 'Net Income', 'Book Value', 'Free Cash Flow']
    lst_all_names = lst_fin_names + ['EV', 'EV / EBITDA', 'EV / Sales', 'Net Debt / EBITDA', 'CAPEX / Net Sales', 'Interest Coverage ratio', 'Debt / Equity', 'Total Current Assets / Total Assets']
    lst_names = ["%s %s" % (f, y) for f in lst_fin_names for y in range(1994, 2015)]
    result_data = []
    with open(fname, 'r') as file:
        for row in csv.reader(file):
            data = row[0].split("; ")
            if len(data) == 486:
                data_row = OrderedDict(zip(lst_names, [float(i if i != 'NaN' else 0) for i in data[:-3]]))
            else:
                data_row = OrderedDict(zip(lst_names, [float(i if i != 'NaN' else 0) for i in data[:-1]]))

            # sector and result
            
            if len(data) == 486:
                data_row['Sector'] = data[-3]
                data_row['Result'] = bool(int(data[-2]))
                data_row['Result date'] = datetime.strptime(data[-1], "%d.%m.%Y") if data_row['Result'] else None
            else:
                data_row['Sector'] = data[-1]

            # calculated values recomented
            for y in range(1994, 2015):
                data_row['EV %s' % y] = data_row['Market Capitalization %s' % y] + data_row['Net Debt %s' % y]
                data_row['EV / EBITDA %s' % y] = (data_row['EV %s' % y] / data_row['EBITDA %s' % y]) if data_row['EBITDA %s' % y] != 0 else 0
                data_row['EV / Sales %s' % y] = (data_row['EV %s' % y] / data_row['Net Sales %s' % y]) if data_row['Net Sales %s' % y] != 0 else 0
                data_row['Net Debt / EBITDA %s' % y] = (data_row['Net Debt %s' % y] / data_row['EBITDA %s' % y]) if data_row['EBITDA %s' % y] != 0 else 0
                data_row['CAPEX / Net Sales %s' % y] = (data_row['CAPEX %s' % y] / data_row['Net Sales %s' % y]) if data_row['Net Sales %s' % y] != 0 else 0
                data_row['Interest Coverage ratio %s' % y] = (data_row['Financial Costs %s' % y] / (data_row['EBITDA %s' % y] - data_row['Depreciation  and amortization %s' % y])) if (data_row['EBITDA %s' % y] - data_row['Depreciation  and amortization %s' % y]) != 0 else 0
                data_row['Debt / Equity %s' % y] = ((data_row['Net Debt %s' % y] + data_row['Cash and cash equivalents %s' % y]) / (data_row['Total Assets %s' % y] - data_row['Total Liabilities %s' % y])) if (data_row['Total Assets %s' % y] - data_row['Total Liabilities %s' % y]) != 0 else 0
                data_row['Total Current Assets / Total Assets %s' % y] = (data_row['Total Current Assets %s' % y] / data_row['Total Assets %s' % y]) if data_row['Total Assets %s' % y] != 0 else 0
            
            # calculated values our
            for name in lst_all_names:
                for period in [1, 2, 3, 4, 5]:
                    delta_name = 'Delta %s p:%s' % (name, period)
                    if 'Result' in data_row and data_row['Result']:
                        sold_year = data_row['Result date'].year
                        delta_value = data_row['%s %s' % (name, sold_year)] - data_row['%s %s' % (name, sold_year - period)] \
                                if '%s %s' % (name, sold_year - period) in data_row else 0
                    else:
                        delta_value = 0
                    data_row[delta_name] = delta_value

            result_data.append(data_row)
    return result_data

def compute_ndcg(ans, ideal):
    ans_t = np.array([i[1] for i in ans])
    ideal_num = np.array([int(i) for i in ideal])

    ans_srt = sorted(zip(ans_t, ideal_num), key=lambda x: x[0], reverse=True)
    ideal_srt = sorted(ideal_num, reverse=True)
    ans_summ = sum([ (1.0 / math.log(p + 2, 2)) if v[1] else 0 for p, v in enumerate(ans_srt)])
    ideal_summ = sum([ (1.0 / math.log(p + 2, 2)) if v else 0 for p, v in enumerate(ideal_srt)])

    return ans_summ / ideal_summ

def print_metrics(X, Y, pred):
    # metrics
    pred_bool = np.array([True if t > f else False for f, t in pred])
    fpr, tpr, thresholds = metrics.roc_curve(Y, pred_bool)
    print ' * auc: ', metrics.auc(fpr, tpr)
    print ' * precision: ', metrics.precision_score(Y, pred_bool)
    print ' * recall: ', metrics.recall_score(Y, pred_bool)
    print ' * ndcg: ', compute_ndcg(pred, Y)
    print 

def build_classifier(clf, X_train, X_test, y_train, y_test):
    print '### ' + clf.__class__.__name__, clf.base_estimator.__class__.__name__
    clf.fit(X_train, y_train)
    return clf


# preparing data
data = load_data('Train_contest.csv')
X = np.array([[v for k, v in i.items() if k not in ['Result', 'Result date', 'Sector']] for i in data])
#print len(X[0]), len(data[0])
#exit()
Y = np.array([i['Result'] for i in data])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

# classifier
# http://scikit-learn.org/stable/auto_examples/plot_classifier_comparison.html

classifiers = [
    #SGDClassifier(),
    #KNeighborsClassifier(2),
    #SVC(kernel="linear", C=0.025),
    #DecisionTreeClassifier(max_depth=5),
    #AdaBoostClassifier(),
    AdaBoostClassifier(base_estimator=RandomForestClassifier()),
    #AdaBoostClassifier(base_estimator=SVC(kernel="linear", C=0.025), algorithm='SAMME'),
    #GradientBoostingClassifier(),
    #GaussianNB(),
]

for classifier in classifiers:
    clf = build_classifier(classifier, X_train, X_test, y_train, y_test)
    pred = clf.predict_proba(X_test)
    print_metrics(X_test, y_test, pred)

    data = load_data('Valid_contest.csv')
    X = np.array([[v for k, v in i.items() if k not in ['Result', 'Result date', 'Sector']] for i in data])
    pred = clf.predict_proba(X)
    print pred
    with open('Result.csv', 'w') as file:
        for f, t in pred:
            file.write("%s\n" % t)