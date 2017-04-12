__author__ = 'foursking'
from gen_feat import make_train_set
from gen_feat import make_test_set
from gen_feat import get_actions
from sklearn.model_selection import train_test_split
import xgboost as xgb
from gen_feat import report
from gen_feat import get_sku_ids_in_P
import datetime
import os
from gen_feat import  get_labels
import eval



def xgboost_train(offline_test = False):
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'
    if offline_test:
        train_start_date = '2016-03-05'
        train_end_date = '2016-04-06'
        test_start_date = '2016-04-06'
        test_end_date = '2016-04-11'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2,
                                                        random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 200
    param['nthread'] = 4
    # param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)
    bst.save_model('./cache/bstmodel.bin')
    return bst


def xgboost_make_submission(retrain = False):
    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'
    if os.path.exists('./cache/bstmodel.bin') and not retrain:
        bst = xgb.Booster({'ntheard':4})
        bst.load_model('./cache/bstmodel.bin')
    else:
        bst = xgboost_train()
    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date, )
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    pred = sub_user_index[sub_user_index['label'] >= 0.03]
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    dt = datetime.datetime.now()
    sdt = str(dt.date())+str(dt.hour)+str(dt.minute)+str(dt.second)
    pred.to_csv('./sub/submission_%s.csv' % sdt, index=False, index_label=False)
    # P = get_sku_ids_in_P()

def xgboost_test_offline():
    bst = xgboost_train(True)
    labels = get_labels('2016-04-11','2016-04-16')
    sub_user_index, sub_trainning_data = make_test_set('2016-04-11', '2016-04-16', )
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    pred = sub_user_index[sub_user_index['label'] >= 0.01]
    # pred = sub_user_index
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    labels = labels[labels['label']==1]
    labels['user_id'] = labels['user_id'].astype(int)
    labels = labels[['user_id','sku_id']]

    eval.eval(pred,labels)

    pass

def xgboost_cv():
    train_start_date = '2016-03-05'
    train_end_date = '2016-04-06'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-02-05'
    sub_end_date = '2016-03-05'
    sub_test_start_date = '2016-03-05'
    sub_test_end_date = '2016-03-10'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 4000
    param['nthread'] = 4
    param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)

    sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
                                                                   sub_test_start_date, sub_test_end_date)
    test = xgb.DMatrix(sub_trainning_date)
    # y = bst.predict(test)

    pred = sub_user_index.copy()
    y_true = sub_user_index.copy()
    pred['label'] = y
    y_true['label'] = label
    report(pred, y_true)


if __name__ == '__main__':
    # xgboost_cv()
    # xgboost_make_submission(False)
    xgboost_test_offline()