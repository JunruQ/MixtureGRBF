import pandas as pd
import json
INPUT_PATH = 'input/ukb/ukb_covreg2_trans1_nanf1_biom0.csv'
NUM_DEMO_COLUMNS = 7
# BIOMARKER_PATH = 'analysis/feasibility_validation/logit_param/covreg1_trans1_nanf1_biom0_ad.json'
BIOMARKER_PATH = 'preprocess/significant_biomarker/biom17.txt'
OUTPUT_PATH = 'input/ukb/ukb_covreg2_trans1_nanf1_biom17.csv'

with open(BIOMARKER_PATH, 'r') as f:
    biomarker_list = list(map(lambda x: x.strip(), f.readlines()))
# with open(f'analysis/feasibility_validation/logit_param/covreg1_trans1_nanf1_biom0_all_disease.json', 'r') as f:
#     param = json.load(f)
#     biomarker_list = param['selected_features']

input_table = pd.read_csv(INPUT_PATH)
output_table = input_table.loc[:, list(input_table.columns[:NUM_DEMO_COLUMNS]) + biomarker_list]

output_table.to_csv(OUTPUT_PATH, index=False)