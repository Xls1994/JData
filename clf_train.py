#coding:utf-8
from gen_feat import make_train_set
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
from gen_feat import report
import os
import numpy as np
def gdbt_train():
    from sklearn.ensemble import RandomForestClassifier
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2,
                                                        random_state=0)
    # np.savetxt('train.txt',X_train,fmt='%.2f',delimiter=' ')
    # clf = GradientBoostingClassifier(n_estimators=220)
    clf =RandomForestClassifier(n_estimators=220,
                 criterion="gini",
                 max_depth=10,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None)
    print type(X_train)
    clf.fit(X_train, y_train)
    pre_y_test = clf.predict_proba(X_test)
    print pre_y_test
    print("GBDT Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))

    print u'保存结果.....'
    f_result = open('result.txt', 'w')
    for i in range(0, len(pre_y_test)):
        if i == 0:
            print str(pre_y_test[i][0])
        if i == len(pre_y_test) - 1:
            print str(pre_y_test[i][0])
        f_result.write(str(pre_y_test[i][0]) + '\n')
    # np.savetxt('label_train.txt',y_train,fmt='%i',delimiter=' ')

def model(X,y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    param_test1 = {'n_estimators': range(70, 150, 10)}
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                             min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                             random_state=10),
                            param_grid=param_test1, scoring='roc_auc', cv=5)
    gsearch1.fit(X, y)

if __name__=='__main__':
    gdbt_train()