# coding=utf-8

import os
import shutil
import sys
from collections import Counter
from datetime import datetime

import keras
import pandas as pd
import sklearn.metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import model
from LogUtils import LogUtils

# create a log object

logger=LogUtils('logger').get_log()

# load the input file
def load_data(cell_name):
    code_file1 = './data/'+cell_name+'/4mer_datavec.csv'
    code_file2 = './data/'+cell_name+'/5mer_datavec.csv'
    code_file3 = './data/'+cell_name+'/6mer_datavec.csv'
    label = './data/'+cell_name+'/'+cell_name+'.csv'
    input_4mer=pd.read_csv(code_file1,header=None,index_col=[0])
    input_5mer=pd.read_csv(code_file2,header=None,index_col=[0])
    input_6mer=pd.read_csv(code_file3,header=None,index_col=[0])
    x=pd.concat([input_4mer,input_5mer,input_6mer],axis=1)
    y=pd.read_csv(label)
    y.loc[y.label=='NO','label']=0
    y.loc[y.label=='YES','label']=1
    return x.values,y['label'].values

# balanced the dataset
def dataSample(x,y):
    logger.info("doing the data sampling...")
    logger.info('Original dataset shape:%s'%Counter(y))
    count=dict(Counter(y))
    sm=SMOTE(sampling_strategy={0:int(count[0]),1:int(count[1])*10},random_state=42)
    rus=RandomUnderSampler(sampling_strategy=1,random_state=42)
    x, y = sm.fit_resample(x, y)
    x, y = rus.fit_resample(x,y)
    logger.info('Sampled dataset shape:%s' % Counter(y))
    return x,y

def split_data(train_cells):
    dirs=['./data/integrative_data/train','./data/integrative_data/test']
    for dir in dirs:
        if os.listdir(dir):
            print("clear the directory!")
            shutil.rmtree(dir)
            os.mkdir(dir)
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for cell in train_cells:
        X, Y = load_data(cell)
        X = X.astype('float')
        Y = Y.astype('int')
        K = 1
        for train_index, test_index in skf.split(X, Y):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            x_train = pd.DataFrame(x_train)
            y_train = pd.Series(y_train)
            cell_name = pd.DataFrame([cell] * x_test.__len__())
            x_test = pd.concat([cell_name, pd.DataFrame(x_test)], axis=1)
            y_test = pd.concat([cell_name, pd.DataFrame(y_test)], axis=1)

            x_train.to_csv('./data/integrative_data/train/datavec-' + str(K) + 'fold.csv', index=False, header=False,
                           mode='a')
            y_train.to_csv('./data/integrative_data/train/label-' + str(K) + 'fold.csv', index=False, header=False,
                           mode='a')
            x_test.to_csv('./data/integrative_data/test/datavec-' + str(K) + 'fold.csv', index=False, header=False,
                          mode='a')
            y_test.to_csv('./data/integrative_data/test/label-' +str(K) + 'fold.csv', index=False, header=False,
                          mode='a')
            K += 1

def load_trainfoldData(fold):
    x_train=pd.read_csv('./data/integrative_data/train/datavec-'+str(fold)+'fold.csv',header=None)
    y_train=pd.read_csv('./data/integrative_data/train/label-'+str(fold)+'fold.csv',header=None)
    x_test=pd.read_csv('./data/integrative_data/test/datavec-' +str(fold)+'fold.csv',header=None)
    y_test=pd.read_csv('./data/integrative_data/test/label-'+str(fold)+'fold.csv',header=None)
    return  x_train.values,y_train[0].values,x_test.values,y_test.values

if __name__ == '__main__':

    cell_name=str(sys.argv[1])
    learning_rate=float(sys.argv[2])
    epoch=int(sys.argv[3])

    t1=datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logger.info ('Start time：'+t1)

    if(cell_name=='hg'):
        train_cells = ['mESC', 'myotube', 'macrophage', 'Th-cell', 'proB-cell']
    else:
        train_cells=['H2171','U87','MM1.S']
    split_data(train_cells)
    for K in range(1,11):
        x_train,y_train,x_test,y_test=load_trainfoldData(K)
        x_train,y_train=dataSample(x_train,y_train)

        scaler = StandardScaler()
        scaler = scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test[:,1:] = scaler.transform(x_test[:,1:])

        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        x_train = x_train.values

        test_x = x_test.iloc[:,1:].values
        test_y = y_test[1].values

        x_train = x_train.reshape(x_train.shape[0], 300, 1).astype('float32')
        test_x = test_x.reshape(test_x.shape[0], 300, 1).astype('float32')


        cnn=model.get_model()
        cnn.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(lr=learning_rate),
                    metrics=['binary_accuracy'])
        train_history=cnn.fit(x_train, y_train, epochs=epoch,batch_size=64, verbose=2,validation_data=[test_x,test_y])

        x_test=pd.DataFrame(x_test)
        y_test=pd.DataFrame(y_test)
        for test_cell in train_cells:
            test_x = x_test[x_test[0] == test_cell].iloc[:, 1:].values
            test_y = list(y_test[y_test[0] == test_cell][1].values)
            test_x = test_x.reshape(test_x.shape[0], 300, 1).astype('float32')

            y_test_pred = cnn.predict(test_x)
            test_pred_class =y_test_pred >=0.5
            test_pred_class=pd.Series(test_pred_class[:,0]).values

            test_acc = accuracy_score(test_y,test_pred_class)
            test_auc = roc_auc_score(test_y, y_test_pred)
            test_aupr = average_precision_score(test_y, y_test_pred)
            test_recall = recall_score(test_y, test_pred_class, pos_label=1)
            test_precision = precision_score(test_y, test_pred_class, pos_label=1)

            logger.info(sklearn.metrics.confusion_matrix(test_y, test_pred_class))
            logger.info('\r test_cell: %s test_acc: %s test_auc: %s test_aupr: %s test_recall: %s test_precision: %s' % (
                                                                                                            str(test_cell),
                                                                                                            str(round(test_acc, 4)),
                                                                                                            str(round(test_auc, 4)),
                                                                                                            str(round(test_aupr, 4)),
                                                                                                            str(round(test_recall, 4)),
                                                                                                            str(round(test_precision, 4))))
            model_path='./model/intergrative_model1/integrative_model_'+str(K)+'.h5'
            cnn.save(model_path)

    t2=datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print ("End Time："+t2)

