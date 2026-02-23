import pandas as pd

df = pd.read_csv('data/olink_protein_data.txt', sep='\t')

df = df.pivot(index=['eid', 'ins_index'], columns='protein_id', values='result').reset_index()

df.to_csv('preprocess/data/ukb_table_all_ins.csv', index=False)