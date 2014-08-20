# coding:utf-8
import sys
from random import random
import numpy as np
import csv
from decimal import Decimal
from datetime import datetime
import math

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from collections import Counter, OrderedDict
from sklearn import metrics
from sklearn import cross_validation
from sklearn import linear_model


'''
1) Используемые этапы предобработки данных (нормализация, отбор признаков и т.п.)
2) Описание итогового алгоритма (композиции алгоритмов, если таковая использовалась)
3) Язык программирования и библиотеки, используемые для реализации конечного алгоритма
4) Описание проведенных экспериментов
5) Предполагаемые варианты дальнейшего повышения качества итогового алгоритма
'''

class Algo:
    def load_data(self, file_path, data_type='train'):
        ''' loading data, compute deltas '''
        cat_weiths = {
            'Commercial Services': 11,
             'Communications': 18,
             'Consumer Durables': 1,
             'Consumer Non-Durables': 10,
             'Consumer Services': 14,
             'Distribution Services': 12,
             'Electronic Technology': 16,
             'Energy Minerals': 9,
             'Finance': 5,
             'Health Services': 19,
             'Health Technology': 13,
             'Industrial Services': 6,
             'Miscellaneous': 0,
             'Non-Energy Minerals': 3,
             'Process Industries': 7,
             'Producer Manufacturing': 4,
             'Retail Trade': 8,
             'Technology Services': 15,
             'Transportation': 2,
             'Utilities': 17
        }
        lst_fin_names = [
            'Cash and cash equivalents', 
            'Inventories', 
            'Total Current Assets', 
            'Total Current Liabilities', 
            'Total Assets', 
            'Property, Plant and Equipment, Net', 
            'Goodwill', 
            'Short-Term Debt', 
            'Long-Term Debt', 
            'Net Debt', 
            'Total Liabilities', 
            'Depreciation  and amortization', 
            'CAPEX', 
            'Net Sales', 
            'Gross Margin', 
            'EBITDA', 
            'Dividend yield', 
            'Market Capitalization', 
            'Gross Income', 
            'Financial Costs', 
            'Net Income', 
            'Book Value', 
            'Free Cash Flow']
        lst_all_names = lst_fin_names + [
            'EV', 'EV / EBITDA', 'EV / Sales', 'Net Debt / EBITDA', 
            'CAPEX / Net Sales', 'Interest Coverage ratio', 'Debt / Equity', 
            'Total Current Assets / Total Assets']
        lst_names = ["%s %s" % (f, y) for f in lst_fin_names for y in range(1994, 2015)]
        result_data = []
        with open(file_path, 'r') as file_obj:
            for row in csv.reader(file_obj):
                data = row[0].split("; ")
                if data_type == 'train':
                    data_row = OrderedDict(zip(lst_names, [float(i if i != 'NaN' else 0) for i in data[:-3]]))
                    data_row['Sector'] = cat_weiths[data[-3]]
                    data_row['Result'] = bool(int(data[-2]))
                    data_row['Result date'] = datetime.strptime(data[-1], "%d.%m.%Y") if data_row['Result'] else None
                else:
                    data_row = OrderedDict(zip(lst_names, [float(i if i != 'NaN' else 0) for i in data[:-1]]))
                    data_row['Sector'] = cat_weiths[data[-1]]

                # calculated values recomented
                for y in range(1994, 2015):
                    data_row['EV %s' % y] = data_row['Market Capitalization %s' % y] + data_row['Net Debt %s' % y]
                    data_row['EV / EBITDA %s' % y] = (data_row['EV %s' % y] / data_row['EBITDA %s' % y]) if data_row['EBITDA %s' % y] != 0 else 0
                    data_row['EV / Sales %s' % y] = (data_row['EV %s' % y] / data_row['Net Sales %s' % y]) if data_row['Net Sales %s' % y] != 0 else 0
                    data_row['Net Debt / EBITDA %s' % y] = (data_row['Net Debt %s' % y] / data_row['EBITDA %s' % y]) if data_row['EBITDA %s' % y] != 0 else 0
                    data_row['CAPEX / Net Sales %s' % y] = (data_row['CAPEX %s' % y] / data_row['Net Sales %s' % y]) if data_row['Net Sales %s' % y] != 0 else 0
                    #data_row['Interest Coverage ratio %s' % y] = (data_row['Financial Costs %s' % y] / (data_row['EBITDA %s' % y] - data_row['Depreciation  and amortization %s' % y])) if (data_row['EBITDA %s' % y] - data_row['Depreciation  and amortization %s' % y]) != 0 else 0
                    data_row['Debt / Equity %s' % y] = ((data_row['Net Debt %s' % y] + data_row['Cash and cash equivalents %s' % y]) / (data_row['Total Assets %s' % y] - data_row['Total Liabilities %s' % y])) if (data_row['Total Assets %s' % y] - data_row['Total Liabilities %s' % y]) != 0 else 0
                    data_row['Total Current Assets / Total Assets %s' % y] = (data_row['Total Current Assets %s' % y] / data_row['Total Assets %s' % y]) if data_row['Total Assets %s' % y] != 0 else 0
                
                if 'Result' in data_row and data_row['Result']:
                    sold_year = data_row['Result date'].year
                else:
                    sold_year = 2014

                data_row['Year'] = sold_year

                # calculated values our
                for name in lst_all_names:
                    for period in [1, 2, 3, 5]:
                        delta_name = 'Delta %s p:%s' % (name, period)
                        if 'Result' in data_row and data_row['Result']:
                            sold_year = data_row['Result date'].year
                            delta_value = data_row['%s %s' % (name, sold_year)] - data_row['%s %s' % (name, sold_year - period)] \
                                    if '%s %s' % (name, sold_year - period) in data_row else 0
                        else:
                            sold_year = 2014
                            delta_value = data_row['%s %s' % (name, sold_year)] - data_row['%s %s' % (name, sold_year - period)] \
                                    if '%s %s' % (name, sold_year - period) in data_row else 0
                            #delta_value = 0
                        data_row[delta_name] = delta_value
                for i in lst_fin_names:
                    data_row['%s (var)' % i] = np.var([v for k, v in data_row.items() if i in k])
                for i in lst_names:
                    data_row.pop(i)
                result_data.append(data_row)
        return result_data


    def split_train_data(self, dataset, train_pct=0.8):
        ''' uniformly split data with sold/unsold status '''
        dataset = sorted(dataset, key=lambda x: x['Result'])
        testset = []
        trainset = []
        for row in dataset:
            if random() > train_pct:
                testset.append(row)
            else:
                trainset.append(row)
        return trainset, testset

    def build_classifier(self, trainset):
        ''' returns clf '''
        X_train, y_train = self._split_class_data(trainset)
        clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=30), n_estimators=100)
        clf.fit(X_train, y_train)
        return clf

    def cross_val(self, clf, dataset):
        ''' cross validation '''
        X_train, y_train = self._split_class_data(dataset)
        scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=10)
        return scores

    def compute_metrics(self, testset, clf):
        ''' auc, precision, recall, ndcg '''
        X_test, y_test = self._split_class_data(testset)
        pred = clf.predict_proba(X_test)
        pred_bool = np.array([True if t > f else False for f, t in pred])
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_bool)
        print ' * auc: ', metrics.auc(fpr, tpr)
        print ' * precision: ', metrics.precision_score(y_test, pred_bool)
        print ' * recall: ', metrics.recall_score(y_test, pred_bool)
        print ' * ndcg: ', self.compute_ndcg(pred, y_test)
        print 

    def snp_load(self, fname):
        data = open(fname).readlines()
        data = [d.split(';') for d in data]
        data = [(datetime.strptime(y, "%d.%m.%Y"), float(s[:-2].replace(',', '.'))) for y, s in data]
        return data

    def _split_class_data(self, data):
        remove_cols = ["Financial Costs %s" % i for i in range(1994, 2015)]
        remove_cols = ["Goodwill %s" % i for i in range(1994, 2015)]
        remove_cols = ["Free Cash Flow %s" % i for i in range(1994, 2015)]
        remove_cols = ["Dividend yield %s" % i for i in range(1994, 2015)]
        remove_cols += ['Result', 'Result date']
        X = np.array([[v for k, v in i.items() if k not in remove_cols] for i in data])
        Y = np.array([i['Result'] for i in data])
        return X, Y

    def compute_result(self, data, clf):
        remove_cols = ["Financial Costs %s" % i for i in range(1994, 2015)]
        remove_cols = ["Goodwill %s" % i for i in range(1994, 2015)]
        remove_cols = ["Free Cash Flow %s" % i for i in range(1994, 2015)]
        remove_cols = ["Dividend yield %s" % i for i in range(1994, 2015)]
        remove_cols += ['Result', 'Result date']
        Y = []
        X = np.array([[v for k, v in i.items() if k not in remove_cols] for i in data])
        for x in X:
            Y.append(clf.predict_proba(x)[0])
        return np.array(Y)

    def save_result(self, file_path, result):
        with open(file_path, 'w') as file_obj:
            for i in result:
                file_obj.write("%s\n" % i[1])

    @staticmethod
    def compute_ndcg(ans, ideal):
        ans_t = np.array([i[1] for i in ans])
        ideal_num = np.array([int(i) for i in ideal])

        ans_srt = sorted(zip(ans_t, ideal_num), key=lambda x: x[0], reverse=True)
        ideal_srt = sorted(ideal_num, reverse=True)
        ans_summ = sum([ (1.0 / math.log(p + 2, 2)) if v[1] else 0 for p, v in enumerate(ans_srt)])
        ideal_summ = sum([ (1.0 / math.log(p + 2, 2)) if v else 0 for p, v in enumerate(ideal_srt)])

        return ans_summ / ideal_summ


if __name__ == "__main__":
    train_file, predict_file = sys.argv[1:]
    algo = Algo()
    
    dataset = algo.load_data(train_file)
    trainset, testset = algo.split_train_data(dataset)
    clf = algo.build_classifier(dataset)

    res = algo.cross_val(clf, dataset)
    print res, dir(res)

    trainset_kf = algo._split_class_data(trainset) 
    testset_kf = algo._split_class_data(testset) 

    algo.compute_metrics(testset, clf)

    dataset = algo.load_data(predict_file, 'result')
    res = algo.compute_result(dataset, clf)
    algo.save_result('Result.csv', res)
    #print res