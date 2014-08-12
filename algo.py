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
                    data_row['Sector'] = data[-3]
                    data_row['Result'] = bool(int(data[-2]))
                    data_row['Result date'] = datetime.strptime(data[-1], "%d.%m.%Y") if data_row['Result'] else None
                else:
                    data_row = OrderedDict(zip(lst_names, [float(i if i != 'NaN' else 0) for i in data[:-1]]))
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
                            sold_year = 2014
                            delta_value = data_row['%s %s' % (name, sold_year)] - data_row['%s %s' % (name, sold_year - period)] \
                                    if '%s %s' % (name, sold_year - period) in data_row else 0
                        data_row[delta_name] = delta_value

                result_data.append(data_row)
        return result_data


    def split_train_data(self, dataset, train_pct=0.7):
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

    def build_classifier(self, trainset, testset):
        ''' returns clf '''
        X_train, y_train = self._split_class_data(trainset)
        X_test, y_test = self._split_class_data(testset)
        clf = AdaBoostClassifier(base_estimator=RandomForestClassifier())
        clf.fit(X_train, y_train)
        return clf

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

    def compute_cross_metrics(self):
        ''' cross_validation '''

    def _split_class_data(self, data):
        X = np.array([[v for k, v in i.items() if k not in ['Result', 'Result date', 'Sector']] for i in data])
        Y = np.array([i['Result'] for i in data])
        return X, Y

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
    clf = algo.build_classifier(trainset, testset)

    metricstats = algo.compute_metrics(testset, clf)
    print metricstats
    exit()

    metricstats = algo.compute_cross_metrics(dataset, clf)
    print metricstats

    dataset = algo.load_data()
    algo.compute_result(dataset)