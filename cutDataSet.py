# -*- coding: utf-8 -*-

from datetime import  *
import pandas as pd

def parse_date(raw_date):
    entry_date = raw_date
    year, month, day = entry_date.split(" ")[0].split("-")
    return int(year), int(month), int(day)

def cut_data_as_time(dataSet_path, new_dataSet_path , begin_day, end_day):
    #Action.csv [user_id, sku_id, time, model_id, type, cate, brand,]
    raw_file = open(dataSet_path)
    t_all = open(new_dataSet_path, 'w')
    column_name = raw_file.readline()  # 读出栏位名
    t_all.write(column_name)
    for line in raw_file:
        line=line.strip()
        entry = line.split(",")
        # parse_date
        entry_date = date(*parse_date(entry[2]))
        if entry_date <= end_day and entry_date >= begin_day:
            t_all.write(line)
    t_all.close()
    raw_file.close()
    print u'根据起始和结束时间将数据集取出完成，存入:',new_dataSet_path