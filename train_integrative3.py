import sys
import warnings
from collections import Counter
from datetime import datetime

import keras
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import model
from LogUtils import LogUtils

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
    y=np.array(y)
    print(y)
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

    mm_cells=['mESC','myotube','macrophage','Th-cell','proB-cell']
    hg_cells=['H2171','U87','MM1.S']

    t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logger.info('Start time：' + t1)

    if(cell_name=='hg'):
        train_cells = hg_cells
        test_cells=mm_cells
    else:
        train_cells=mm_cells
        test_cells=hg_cells

    x_train=pd.DataFrame()
    y_train=pd.Series()
    x_test=pd.DataFrame()
    y_test=pd.Series()
    count=1
    for train_cell in train_cells:
        X,Y = load_data(train_cell)
        # X,Y=dataSample(X,Y)
        X=pd.DataFrame(X)
        Y=pd.Series(Y)
        x_train=pd.concat([x_train,X],axis=0,ignore_index=True)
        y_train=pd.concat([y_train,Y],axis=0,ignore_index=True)

    x_train,y_train=dataSample(x_train,y_train)
    print(Counter(y_train))

    for test_cell in test_cells:
        X, Y = load_data(test_cell)
        X=pd.DataFrame(X)
        Y=pd.Series(Y)
        x_test = pd.concat([x_test, X], axis=0, ignore_index=True)
        y_test = pd.concat([y_test, Y], axis=0, ignore_index=True)
    print(Counter(y_test))

    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # x_train=x_train.values
    # x_test=x_test.values
    x_train = x_train.reshape(x_train.shape[0], 300, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 300, 1).astype('float32')

    cnn = model.get_model()

    cnn.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=learning_rate),
                metrics=['binary_accuracy'])

    train_history = cnn.fit(x_train, y_train, validation_data=[x_test,y_test], epochs=epoch,
                              batch_size=64, verbose=2)


    y_test_pred = cnn.predict(x_test)
    val_pred_class = y_test_pred >= 0.5
    val_acc = accuracy_score(y_test, val_pred_class)
    val_auc_val = roc_auc_score(y_test, y_test_pred)
    val_aupr_val = average_precision_score(y_test, y_test_pred)
    val_recall_val = recall_score(y_test, val_pred_class, pos_label=1)
    val_precision_val = precision_score(y_test, val_pred_class, pos_label=1)

    print('\r val_acc: %s ' % (str(round(val_acc, 4))))
    print('\r val_auc_val: %s ' % (str(round(val_auc_val, 4))))
    print('\r val_aupr_val: %s ' % (str(round(val_aupr_val, 4))))
    print('\r val_recall_val: %s ' % (str(round(val_recall_val, 4))))
    print('\r val_precision_val: %s ' % (str(round(val_precision_val, 4))))

    model_path = './model/integrative_model3/' + str(cell_name) + '.h5'
    cnn.save(model_path)

    t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print("End time：" + t2)



