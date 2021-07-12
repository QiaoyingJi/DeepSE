import csv
import pandas as pd
import gensim
import Bio
from Bio import SeqIO
import numpy as np
import os
from multiprocessing import Pool,Process
import time

NULL_vec=np.zeros((100))
seq_file='./data/myotube/myotube.fasta'
# code_file='./data/mESC_constituent/6mer_datavec.csv'
genome='mm9'
embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format("./data/" + genome + "_embedding.w2v")
def get_kmer(dnaSeq,K):
    dnaSeq=dnaSeq.upper()
    l=len(dnaSeq)
    return [dnaSeq[i:i+K] for i in range(0,l-K+1,K)]

def seq_to_vec(seq_records,K):
    print('The process %d is running'%K)
    code_file='./data/myotube/'+str(K)+'mer_datavec.csv'
    seqid=1
    for seq_record in seq_records:
        dnaSeq=str(seq_record.seq)
        kmers=get_kmer(dnaSeq,K)
        code=[]
        for kmer in kmers:
            if ('n' not in kmer) and ('N' not in kmer):
                code.append(embedding_matrix[kmer])
                # code=np.concatenate((code,embedding_matrix[kmer]),axis=0)
            else:
                code.append(NULL_vec)
                # code=np.concatenate((code,NULL_vec),axis=0)
        #计算ac系数
        # ac = np.zeros(10)  # 储存ac系数
        array = np.array(code)  # 每一个list是一条dna序列的编码，一个n×100的数组，n是kmer的个数
        ave=array.sum(axis=0)
        # ave = array.sum(axis=0) / array.shape[0]  # 按列累加除以行数（kmer个数）
        count = 0
        # for lg in range(0, 10):  # 计算lg_high-lg_low+1个ac系数
        #     sum = 0
        #     for i in range(0, array.shape[0] - lg):
        #         # sum +=np.inner(array[i],array[i+lg])
        #         sum += np.inner((array[i] - ave/array.shape[0]), (array[i + lg] - ave/array.shape[0]))
        #     ac[count] = sum / (array.shape[0] - lg)  # 每一条dna序列的ac
        #     count += 1

        ave = pd.DataFrame(ave).T
        # ac = pd.DataFrame(ac).T
        id = pd.DataFrame([seqid]).T
        # acc_code = pd.concat([id, ave, ac], axis=1, ignore_index=True)
        ave=pd.concat([id,ave],axis=1,ignore_index=True)
        ave.to_csv(code_file,index=False,mode='a',header=False)
        # acc_code.to_csv(code_file, index=False, mode='a', header=False)
        print('the %dth seq is done' % seqid)
        seqid+=1
    # codes=pd.DataFrame(codes)
    # codes.to_csv('./data/neg_datavec.csv',index=False)
    # print(codes)


if __name__=='__main__':
    seq_records=list(Bio.SeqIO.parse(seq_file,'fasta'))
    records_num=len(seq_records)
    print(records_num)
    print('The main process %s is running...'%os.getpid())

    ps=[]
    for K in range(4,7):
        p=Process(target=seq_to_vec,args=(seq_records,K))
        ps.append(p)
    for i in range(3):
        ps[i].start()
    for i in range(3):
        ps[i].join()

    # p = Pool(3)
    # for K in range(4,7):
    #     p.apply_async(seq_to_vec, args=(seq_records, K))
    #
    # p.close()
    # p.join()

    print('The main process %s is done...' % os.getpid())


    # data=data.sort_index()
    # print(data)
    # data.to_csv(code_file)

# if __name__=='__main__':
#     seq_records=list(Bio.SeqIO.parse(seq_file,'fasta'))
#     records_num=len(seq_records)
#     print(records_num)
#     print('The main process %s is running...'%os.getpid())
#
#     for i in range(0, records_num):
#         dnaSeq=str(seq_records[i].seq)
#         seq_to_vec(dnaSeq, i, 6)
#
#
#     print('The main process %s is done...' % os.getpid())
#     data=pd.read_csv(code_file,header=None,index_col=[0])
#     print(data)