# DeepSE(Detecting super-enhancers among typical enhancers using only sequence feature embeddings)

DeepSE is a model based on a two-layer CNN model using computational methods to predict the SEs from TEs.

In this work, we aimed at predicting SEs and their constituents based on the DNA sequences only. As far as we can tell, this is the first attempt to identify SEs without ChIP-seq data, which will be useful in computational genome annotations. 

## **File descriptions**

### LogUtils.py

Create a log file during training.

### get_datevec.py

Encode the DNA sequence of TEs and SEs with the pre-trained k-mer feature vectors.

### model.py

A two-layer CNN model was implemented by Keras.

### train_specific.py

Train a cell-line specific model.

### train_integrative1.py

Trained a universal model DeepSE-integrative by pooling all cell line datasets together, and testing on every cell line respectively.

### train_integrative2.py

Trained a model DeepSE-integrative by pooling several cell line datasets, and testing in a new cell line. 

### train_integrative3.py

Trained a model DeepSE-integrative on one species datasets and test on another.



## **How to use DeepSE**

### **Step 1**: **Install third-party packages.**

DeepSE requires the runtime enviroment with Python >=3.6. The modules requried by DeepSE is provied in the file requirements.txt, and you can install them by

```python
pip install requirements.txt
```

### **Step 2**: **Encode the DNA sequence.**

Convert the DNA sequence of TEs and SEs into feature vectors with the pre-trained k-mer feature vectors. You should prepared the DNA sequence file with 'fasta' format, and put it in your directory according to the cell name. You can type the command line like follows:

```python
python get_datavec.py cell_name genome
```

The term 'cell_name' is the dataset you want to encode, and 'genome' is the DNA genome of the cell. We have provided two pre-trained embedding matrixs, including human (version hg19) and mouse (mm9). Meanwhile, you can train the dna2vec (https://ngpatrick.com/dna2vec/) model and get your specific embedding matrixs.

We have uploaded all the encoded cell data used in our experiments, including mESC_constituent, mESC, myotube, macrophage, pro-B cell, Th-cell, H2171, MM1.S, and U87. 

### **Step 3: Choose the model.**

There are two kinds of DeepSE, DeepSE-specific and DeepSE-integrative.

#### Train a cell-line specific model

To predict SEs in cell-line specific way, you can type the command line like follows:

```python
python train_specific.py cell_name learning_rate epochs
```

The term 'cell_name' is the cell line you choose from mESC_constituent, mESC, myotube, macrophage, pro-B cell, Th-cell, H2171, MM1.S, and U87. And the parameters 'learning_rate' and 'epochs' are the learning rate and epochs setting for the training model.

#### Train an integrative model

- Create a universal prediction model by pooling all kinds of cell line data together. You can type like follows:

```python
python train_integrative1.py species learning_rate epochs
```

- Train an integrative model by pooling several cell line datasets, and testing in a new cell line. You can type like follows:

```python
python train_integrative2.py species learning_rate epochs
```

- Train an integrative model on one species and test on another. You can type like follows:

```python
python train_integrative3.py species learning_rate epochs
```

The parameter 'species' is the species setting as the training set, and can be hg or mm, representing 'human' and 'mouse' respectively. And the parameters 'learning_rate' and 'epochs' are the learning rate and epochs setting for the training model.
