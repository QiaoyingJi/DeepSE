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


def get_kmer(dnaSeq,K):
    dnaSeq=dnaSeq.upper()
    l=len(dnaSeq)
    return [dnaSeq[i:i+K] for i in range(0,l-K+1,K)]

def seq_to_vec(cell_name,seq_records,embedding_matrix,K):
    print('The process %d is running'%K)
    code_file='./data/%s/'%cell_name+str(K)+'mer_datavec.csv'
    seqid=1
    for seq_record in seq_records:
        dnaSeq=str(seq_record.seq)
        kmers=get_kmer(dnaSeq,K)
        code=[]
        for kmer in kmers:
            if ('n' not in kmer) and ('N' not in kmer):
                code.append(embedding_matrix[kmer])
            else:
                code.append(NULL_vec)
        array = np.array(code)
        ave=array.sum(axis=0)
        ave = pd.DataFrame(ave).T
        id = pd.DataFrame([seqid]).T
        ave=pd.concat([id,ave],axis=1,ignore_index=True)
        ave.to_csv(code_file,index=False,mode='a',header=False)

        print('the %dth seq is done' % seqid)
        seqid+=1


if __name__=='__main__':
    cell_name=str(sys.argv[1])
    genome=str(sys.argv[2])
    seq_file = './data/%s/%s.fasta'%(cell_name,cell_name)
    seq_records=list(Bio.SeqIO.parse(seq_file,'fasta'))
    embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format("./data/" + genome + "_embedding.w2v")
    records_num=len(seq_records)
    print(records_num)
    print('The main process %s is running...'%os.getpid())
    ps=[]
    for K in range(4,7):
        p=Process(target=seq_to_vec,args=(cell_name,seq_records,embedding_matrix,K))
        ps.append(p)
    for i in range(3):
        ps[i].start()
    for i in range(3):
        ps[i].join()
    print('The main process %s is done...' % os.getpid())
