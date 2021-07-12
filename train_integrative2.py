import sys
import os
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LogUtils import LogUtils
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import classification_report
import sklearn.metrics
import matplotlib.pyplot as plt
from collections import Counter
from keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score
import model
import itertools
import warnings
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# create a log object
logger = LogUtils('logger').get_log()


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
    x, y = sm.fit_sample(x, y)
    x, y = rus.fit_sample(x,y)
    logger.info('Sampled dataset shape:%s' % Counter(y))
    return x,y

if __name__ == '__main__':

    cell_name=str(sys.argv[1])
    learning_rate=float(sys.argv[2])
    epoch=int(sys.argv[3])

    t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logger.info('Start time：' + t1)
    if(cell_name=='hg'):
        cells = ['mESC', 'myotube', 'macrophage', 'Th-cell', 'proB-cell']
    else:
        cells=['H2171','U87','MM1.S']

    for test_cell in cells:
        train_cells=set(cells)-set(test_cell)
        x_train=pd.DataFrame()
        y_train=pd.Series()
        x_test,y_test=load_data(test_cell)
        for train_cell in train_cells:
            X,Y = load_data(train_cell)
            X = pd.DataFrame(X)
            Y = pd.Series(Y)
            x_train = pd.concat([x_train, X], axis=0, ignore_index=True)
            y_train = pd.concat([y_train, Y], axis=0, ignore_index=True)

        x_train,y_train=dataSample(x_train,y_train)
        scaler = StandardScaler()
        scaler = scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_train = x_train.reshape(x_train.shape[0], 300, 1).astype('float32')
        x_test=x_test.reshape(x_test.shape[0],300,1).astype('float32')

        cnn = model.get_model()

        cnn.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=learning_rate),
                    metrics=['binary_accuracy'])

        train_history = cnn.fit(x_train, y_train, epochs=epoch,
                                batch_size=64, verbose=2,validation_data=[x_test,y_test])
        y_test_pred = cnn.predict(x_test)
        test_pred_class = y_test_pred >= 0.5
        test_acc = accuracy_score(y_test, test_pred_class)
        test_auc = roc_auc_score(y_test, y_test_pred)
        test_aupr = average_precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, test_pred_class, pos_label=1)
        test_precision = precision_score(y_test, test_pred_class, pos_label=1)

        logger.info('\r test_cell: %s test_acc: %s test_auc: %s test_aupr: %s test_recall: %s test_precision: %s' % (
                                                                                                                    str(test_cell),
                                                                                                                    str(round(test_acc, 4)),
                                                                                                                    str(round(test_auc, 4)),
                                                                                                                    str(round(test_aupr, 4)),
                                                                                                                    str(round(test_recall, 4)),
                                                                                                                    str(round(test_precision, 4))))
        model_path = './model/integrative_model2/' + str(test_cell) + '.h5'
        cnn.save(model_path)

    t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logger.info("End time：" + t2)

