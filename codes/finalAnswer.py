import pandas as pd


if __name__=='__main__':
    path = 'data/three_answer_rf.csv'
    productpath ='input/JData_Product.csv'
    dataFrame = pd.read_csv(path)
    dataFrame['user_id']=int(dataFrame['user_id'])
    # product =pd.read_csv(productpath)
    dataFrame.to_csv('answer.csv', sep=',', index=False)
    print dataFrame.head()