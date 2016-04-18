import pandas as pd
import math
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import time
import numpy as np

df = pd.read_csv('data/transactions_full.csv',encoding = "ISO-8859-1")

uniq_stock = list(df.StockCode.unique())
uniq_stock.sort()

uniq_customer = list(df.CustomerID.unique())
uniq_customer.sort()

# stock_df: used to retrieve the stock name and description
stock_df = pd.DataFrame(data = np.matrix(df[['StockCode','Description']]),index = df['StockCode'].values,columns=['StockCode','Description'])
stock_df = stock_df.drop_duplicates(subset = ['StockCode'])

# 90% of the data is selected as training data
msk = np.random.rand(len(df)) < 0.9
train_df = df[msk]
test_df = df[~msk]

# sparse_df: the relationship btw each customer and product based on the training data
# col: stock
# row: customer
row = train_df.CustomerID.astype('category', categories=uniq_customer).cat.codes
col = train_df.StockCode.astype('category', categories=uniq_stock).cat.codes
data = listofones = [1] * train_df.shape[0]
sparse_matrix = csr_matrix((data, (row, col)), shape=(len(uniq_customer), len(uniq_stock)))
sparse_matrix.data = np.array([1]*len(sparse_matrix.data))

sparse_df = pd.DataFrame(data = sparse_matrix.todense(),columns = uniq_stock,index = uniq_customer)

# coocc_stock_df: product similarity matrix
coocc_stock = sparse_matrix.T.dot(sparse_matrix)

coocc_stock_d = np.sqrt(coocc_stock.diagonal())
coocc_stock = (lil_matrix(np.diag(1/coocc_stock_d))*coocc_stock).T
coocc_stock = (lil_matrix(np.diag(1/coocc_stock_d))*coocc_stock).T
coocc_stock.setdiag(0)

coocc_stock_df = pd.DataFrame(data = coocc_stock.T.todense(),index = uniq_stock,columns = uniq_stock)

# K_similar: store the K most similiar products
K = 50
K_similar = []
for product in uniq_stock:
    p_similarity = coocc_stock_df[product].sort_values(ascending = False).head(K)
    K_similar.append(p_similarity)

# cp_prediction: predict the score of a particular customer to a specific product
def cp_prediction(ProductID, K_similar, CustomerID, sparse_df,K):
    i = uniq_stock.index(ProductID)
    p_similarity = K_similar[i]
    c_history = sparse_df.loc[CustomerID]
    c_history = c_history.loc[p_similarity.index.tolist()]
    return sum((p_similarity*c_history))/sum(p_similarity)

# c_recommend: recommend 10 most related products to the customer
def c_recommend(K_similar, CustomerID, sparse_df,K):
    result = pd.Series(data = 0.0,index = uniq_stock)
    for product in uniq_stock:
        if sparse_df[product][CustomerID]!=1:
            result.set_value(product,cp_prediction(product, K_similar, CustomerID, sparse_df,K))
    p_p = result.sort_values(ascending = False).head(10)
    return p_p

# show: present the result in a clear format
def show(a):
    b = stock_df.loc[a.index.tolist()].assign(Score = a.values)
    return b

# to view the recommending product for customer '17850.0'
# show(c_recommend(K_similar, '17850.0', sparse_df,K))
