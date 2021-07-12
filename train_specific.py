import sys
import warnings
from collections import Counter
from datetime import datetime
import keras
import pandas as pd
import sklearn.metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import model
from LogUtils import LogUtils

warnings.filterwarnings("ignore")

# create a log object
logger = LogUtils('logger').get_log()


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
def dataSample(x, y):
    logger.info("doing the data sampling...")
    logger.info('Original dataset shape:%s' % Counter(y))
    # 将少数集过采样扩大十倍，多数集下采样和少数集1：1
    count = dict(Counter(y))
    sm = SMOTE(sampling_strategy={0: int(count[0]), 1: int(count[1]) * 10}, random_state=42)
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    x, y = sm.fit_sample(x, y)
    x, y = rus.fit_sample(x, y)
    logger.info('Sampled dataset shape:%s' % Counter(y))
    return x, y


if __name__ == '__main__':
    mm_cells=['mESC_constituent','mESC','myotube','macrophage','Th-cell','proB-cell']
    hg_cells=['H2171','U87','MM1.S']
    cell_name=str(sys.argv[1])
    learning_rate=float(sys.argv[2])
    epoch=int(sys.argv[3])
    if cell_name in mm_cells or cell_name in hg_cells:
        t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        logger.info('开始时间：' + t1)

        X, Y = load_data(cell_name)

        logger.info('learning_rate= % s, epoch = %s' % (str(learning_rate), str(epoch)))

        skf = StratifiedKFold(n_splits=10)
        K=1
        for train_index, test_index in skf.split(X, Y):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            x_train, y_train = dataSample(x_train, y_train)

            scaler = StandardScaler()
            scaler = scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            x_train = pd.DataFrame(x_train)
            x_test = pd.DataFrame(x_test)
            y_train = pd.DataFrame(y_train)
            y_test = pd.DataFrame(y_test)

            x_train = x_train.values
            x_test = x_test.values
            x_train = x_train.reshape(x_train.shape[0], 300, 1).astype('float32')
            x_test = x_test.reshape(x_test.shape[0], 300, 1).astype('float32')

            cnn = model.get_model()
            cnn.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=learning_rate),
                        metrics=['binary_accuracy'])
            train_history = cnn.fit(x_train, y_train, validation_data=[x_test, y_test], epochs=epoch,
                                    batch_size=64, verbose=2)

            y_train_pred = cnn.predict(x_train)
            train_pred_class = y_train_pred >= 0.5

            train_acc = accuracy_score(y_train, train_pred_class)
            train_auc = roc_auc_score(y_train, y_train_pred)
            train_aupr = average_precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, train_pred_class, pos_label=1)
            train_precision = precision_score(y_train, train_pred_class, pos_label=1)

            logger.info('\r train_acc: %s train_auc: %s train_aupr: %s train_recall: %s train_precision: %s' % (
                                                                                                                str(round(train_acc, 4)),
                                                                                                                str(round(train_auc, 4)),
                                                                                                                str(round(train_aupr, 4)),
                                                                                                                str(round(train_recall, 4)),
                                                                                                                str(round(train_precision, 4))))


            y_test_pred = cnn.predict(x_test)
            test_pred_class = y_test_pred >= 0.5


            test_acc = accuracy_score(y_test, test_pred_class)
            test_auc = roc_auc_score(y_test, y_test_pred)
            test_aupr = average_precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, test_pred_class, pos_label=1)
            test_precision = precision_score(y_test, test_pred_class, pos_label=1)

            logger.info('\r test_acc: %s test_auc: %s test_aupr: %s test_recall: %s test_precision: %s' % (
                                                                                                            str(round(test_acc, 4)),
                                                                                                            str(round(test_auc, 4)),
                                                                                                            str(round(test_aupr, 4)),
                                                                                                            str(round(test_recall, 4)),
                                                                                                            str(round(test_precision, 4))))
            model_path='./model/specific/'+cell_name+'_specific_'+str(K)+'.h5'
            cnn.save(model_path)
            K+=1

        t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        logger.info("结束时间：" + t2)
    else:
        print("Please verify your cell name!")
