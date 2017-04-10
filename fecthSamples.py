
# -*- coding: utf-8-*-
import csv

'''
# train_one_label_data_path, train_one_train_feature_path, 'one_'
# ['user_id', 'sku_id', 'time', 'model_id', 'type', 'cate', 'brand']
# 1.浏览（指浏览商品详情页）；2.加入购物车；3.购物车删除；4.下单；5.关注；6.点击
'''

# 获得正负样本，并且为每个样本添加label，分别保存在正样本、负样本文本中
def fetch_sample(test_data_path, feature_data_path, negative_data_path,positive_data_path):

    buy = set()
    for line in csv.reader(file(test_data_path, 'r')):
        if line[4] == '4':
            buy.add((line[0], line[1]))  # 正例集

    negative_file = file(negative_data_path, 'w')
    negative_writer = csv.writer(negative_file)

    positive_file = file(positive_data_path, 'w')
    positive_writer = csv.writer(positive_file)

    print 'open ',feature_data_path,'to add label'
    print len(buy)
    for line in csv.reader(file(feature_data_path, 'r')):
        if line[0]=='user_id':
            line.extend('r')
            negative_writer.writerow((line))
            positive_writer.writerow((line))
        elif (line[0], line[1]) not in buy:
            line.append(0)
            negative_writer.writerow(line)  # 负例特征
        elif (line[0], line[1]) in buy:
            line.append(1)
            positive_writer.writerow(line)  # 正例特征
    print u'正负样本分类并打好标签，分别存入:',negative_data_path,positive_data_path


# 抽取部分负样本作为训练数据集 抽取行数%200==0的负样本
def fetch_negative_sample(negative_data_path, new_negative_data_path):
    num = 1
    csvfile = file(new_negative_data_path, 'w')
    writer = csv.writer(csvfile)
    for line in csv.reader(file(negative_data_path, 'r')):
        if num==1:
            writer.writerow(line)
        elif num % 200 == 0:
            writer.writerow(line)
        num = num + 1
    print num
    print u'挑选部分负样本，存入:',new_negative_data_path

# 正负样本融合在一起构成训练集
def combine_neg_and_posi(negative_data_path, positive_data_path, train_dataSet_path):
    negative_data = open(negative_data_path, 'r')
    positive_data = open(positive_data_path, 'r')
    train_dataSet = open(train_dataSet_path, 'w')
    train_dataSet.write(negative_data.readline())
    for line in negative_data.readlines():
        if line.strip().split(',')[0] =='user_id':
            continue
        else:
            train_dataSet.write(line)
    for line in positive_data.readlines():
        if line.strip().split(',')[0] =='user_id':
            continue
        else:
            train_dataSet.write(line)
    print u'正负样本融合，存入:',train_dataSet_path