from datetime import datetime
# coding:utf-8
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from collections import Counter, OrderedDict
from sklearn import metrics
from sklearn import cross_validation
from sklearn import linear_model



if __name__ == "__main__":
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

    lst_attrs = ['Cash and cash equivalents', 'Inventories', 'Total Current Assets', 'Total Current Liabilities', 'Total Assets', 'Property, Plant and Equipment, Net', 'Goodwill', 'Short-Term Debt', 'Long-Term Debt', 'Net Debt', 'Total Liabilities', 'Depreciation  and amortization', 'CAPEX', 'Net Sales', 'Gross Margin', 'EBITDA', 'Dividend yield', 'Market Capitalization', 'Gross Income', 'Financial Costs', 'Net Income', 'Book Value', 'Free Cash Flow']
    result_data = OrderedDict()

    with open('Train_contest2.csv') as f:
        company_id = 0
        while True:
            line = f.readline()
            if not line:
                break
            data = line[:-1].split('; ')
            if data[-1] != '-':
                sold_date = datetime.strptime(data[-1], "%d.%m.%Y")
            else:
                sold_date = 0
            sector = cat_weiths[data[-3]]
            for i, attr_name in enumerate(lst_attrs):
                for year in range(1995, 2015):
                    key = '%s_%s' % (company_id, year)
                    if key not in result_data:
                        result_data[key] = OrderedDict()
                    result_data[key][attr_name] = float(data[i*(year-1995)] if data[i*(year-1995)] != 'NaN' else 0)
            for year in range(1995, 2015):
                key = '%s_%s' % (company_id, year)
                result_data[key]['Sector'] = sector
                result_data[key]['Merged'] = True if sold_date and sold_date.year - 1 == year else False
            company_id += 1

    for ckey in result_data.keys():
        result_data[ckey]['EV'] = result_data[ckey]['Market Capitalization'] + result_data[ckey]['Net Debt']
        result_data[ckey]['EV / EBITDA'] = (result_data[ckey]['EV'] / result_data[ckey]['EBITDA']) \
            if result_data[ckey]['EBITDA'] != 0 else 0
        result_data[ckey]['EV / Sales'] = (result_data[ckey]['EV'] / result_data[ckey]['Net Sales']) \
            if result_data[ckey]['Net Sales'] != 0 else 0
        result_data[ckey]['Net Debt / EBITDA'] = (result_data[ckey]['Net Debt'] + result_data[ckey]['EBITDA']) \
            if result_data[ckey]['EBITDA'] != 0 else 0
        result_data[ckey]['CAPEX / Net Sales'] = (result_data[ckey]['CAPEX'] + result_data[ckey]['Net Sales']) \
            if result_data[ckey]['Net Sales'] != 0 else 0
        result_data[ckey]['Debt / Equity'] = \
            ( (result_data[ckey]['Net Debt'] + result_data[ckey]['Cash and cash equivalents']) / 
             (result_data[ckey]['Total Assets'] - result_data[ckey]['Total Liabilities']) ) \
            if (result_data[ckey]['Total Assets'] - result_data[ckey]['Total Liabilities']) != 0 else 0
        result_data[ckey]['Total Current Assets / Total Assets'] = \
            (result_data[ckey]['Total Current Assets'] / result_data[ckey]['Total Assets']) \
            if result_data[ckey]['Total Assets'] != 0 else 0

    for ckey in result_data.keys():
        cid, year = [int(i) for i in ckey.split('_')]
        for attr in result_data[ckey].keys():
            if attr in ['Merged', 'Sector']:
                continue
            for period in [1, 2, 3, 5, 10]:
                delta_name = 'Delta %s p:%s' % (attr, period)
                p_year = year - period
                p_key = '%s_%s' % (cid, p_year)
                if p_key in result_data:
                    delta_value = result_data[ckey][attr] - result_data[p_key][attr]
                else:
                    delta_value = 0
                result_data[ckey][delta_name] = delta_value

    print 'data loaded', datetime.now()

    X = np.array([[v for k, v in i.items() if k != 'Merged'] for i in result_data.values()])
    Y = np.array([i['Merged'] for i in result_data.values()])

    clf = AdaBoostClassifier(base_estimator=RandomForestClassifier())
    clf.fit(X, Y)
    print 'clf build', datetime.now()

    scores = cross_validation.cross_val_score(clf, X, Y, cv=5)

    print scores

    exit()



    valid_data = OrderedDict()

    with open('Train_contest2.csv') as f:
        company_id = 0
        while True:
            line = f.readline()
            if not line:
                break
            data = line[:-1].split('; ')
            if data[-1] != '-':
                sold_date = datetime.strptime(data[-1], "%d.%m.%Y")
            else:
                sold_date = 0
            sector = cat_weiths[data[-3]]
            for i, attr_name in enumerate(lst_attrs):
                for year in range(1995, 2015):
                    key = '%s_%s' % (company_id, year)
                    if key not in result_data:
                        valid_data[key] = OrderedDict()
                    valid_data[key][attr_name] = float(data[i*(year-1995)] if data[i*(year-1995)] != 'NaN' else 0)
            for year in range(1995, 2015):
                key = '%s_%s' % (company_id, year)
                valid_data[key]['Sector'] = sector
                valid_data[key]['Merged'] = True if sold_date and sold_date.year - 1 == year else False
            company_id += 1

    for ckey in result_data.keys():
        result_data[ckey]['EV'] = result_data[ckey]['Market Capitalization'] + result_data[ckey]['Net Debt']
        result_data[ckey]['EV / EBITDA'] = (result_data[ckey]['EV'] / result_data[ckey]['EBITDA']) \
            if result_data[ckey]['EBITDA'] != 0 else 0
        result_data[ckey]['EV / Sales'] = (result_data[ckey]['EV'] / result_data[ckey]['Net Sales']) \
            if result_data[ckey]['Net Sales'] != 0 else 0
        result_data[ckey]['Net Debt / EBITDA'] = (result_data[ckey]['Net Debt'] + result_data[ckey]['EBITDA']) \
            if result_data[ckey]['EBITDA'] != 0 else 0
        result_data[ckey]['CAPEX / Net Sales'] = (result_data[ckey]['CAPEX'] + result_data[ckey]['Net Sales']) \
            if result_data[ckey]['Net Sales'] != 0 else 0
        result_data[ckey]['Debt / Equity'] = \
            ( (result_data[ckey]['Net Debt'] + result_data[ckey]['Cash and cash equivalents']) / 
             (result_data[ckey]['Total Assets'] - result_data[ckey]['Total Liabilities']) ) \
            if (result_data[ckey]['Total Assets'] - result_data[ckey]['Total Liabilities']) != 0 else 0
        result_data[ckey]['Total Current Assets / Total Assets'] = \
            (result_data[ckey]['Total Current Assets'] / result_data[ckey]['Total Assets']) \
            if result_data[ckey]['Total Assets'] != 0 else 0