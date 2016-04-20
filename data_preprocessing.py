# set the unkonwn CustomerIDs to 'Anonymous N'
# and save the new csv file as 'transactions_full.csv'

import pandas as pd

df = pd.read_csv('../data/transactions.csv',encoding = "ISO-8859-1")
df.sort_values('InvoiceNo')
df[['StockCode','CustomerID']] = df[['StockCode','CustomerID']].astype(str)


i = 1
for index,row in df.iterrows():
    if row.CustomerID == 'nan':
        if df.loc[index-1].InvoiceNo != row.InvoiceNo:
            df.set_value(index,'CustomerID','Anonymous' + str(i))
            i += 1
        else:
            df.set_value(index,'CustomerID','Anonymous' + str(i))
print (i-1) + 'anonymous customers are added.'    

df.to_csv('../data/transactions_full.csv',index= False)
