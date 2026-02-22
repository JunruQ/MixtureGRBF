import json
import pandas as pd
import os
import numpy as np
import utils.utils as utils

def parse_target_field(df: pd.DataFrame) -> pd.DataFrame:
    df_result = pd.DataFrame()
    df_result['eid'] = df['eid']
    if 'Field' in df.columns:
        df_result['field'] = df['Field'].apply(lambda x: x.split(' ')[1] if not pd.isna(x) else x)
    elif 'target_cancer' in df.columns:
        df_result['field'] = df['target_cancer']
    elif 'target_death' in df.columns:
        df_result['field'] = df['target_death']
    else:
        raise ValueError('Field not found in dataframe')
    df_result['bl2t'] = df['BL2Target_yrs']
    return df_result

with open('preprocess/data/important_disease_healthspan.json', 'r') as f:
    important_disease = json.load(f)

nsubtype = 5
result_folder = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
output_dir = f'./output/result_analysis/{result_folder}/{nsubtype}_subtypes'
subtype_stage = utils.get_subtype_stage(result_folder, nsubtype)
subtype_stage = subtype_stage.rename(columns={'PTID': 'eid'})
os.makedirs(output_dir, exist_ok=True)

result_records = []

for i, (disease_name, disease_code) in enumerate(important_disease.items()):
    disease_upper_level_code = disease_code[0][0]
    
    disease_info = pd.read_csv(f'./input/disease_info/{disease_upper_level_code}0.csv')
    death_info = pd.read_csv(f'./input/disease_info/X0.csv')
    
    df_s = pd.merge(subtype_stage, parse_target_field(disease_info), on='eid', how='left')
    df_s = pd.merge(df_s, parse_target_field(death_info), on='eid', how='left', suffixes=('_disease', '_death'))

    for id in df_s['eid'].unique():
        r = df_s[df_s['eid'] == id]
        subtype = r['subtype'].values[0]
        stage_offset = r['stage'].values[0]

        disease_field = r['field_disease'].values[0]
        death_field = r['field_death'].values[0]

        bl2t_disease = r['bl2t_disease'].values[0]
        bl2t_death = r['bl2t_death'].values[0]

        stage_2_censored = not disease_field in disease_code
        stage_3_censored = pd.isna(death_field)

        end_time = stage_offset + max(bl2t_disease, bl2t_death)

        # Stage 1: 固定为 baseline
        result_records.append({
            'eid': id,
            'time': 0,
            'stage': 1,
            'event': 1,
            'subtype': subtype,
            'disease': disease_name,
        })

        # Case 1: Stage 2, 3 都删失
        if stage_2_censored & stage_3_censored:
            result_records.append({
                'eid': id,
                'time': end_time,
                'stage': 2,
                'event': 0,
                'subtype': subtype,
                'disease': disease_name,
            })
            result_records.append({
                'eid': id,
                'time': end_time,
                'stage': 3,
                'event': 0,
                'subtype': subtype,
                'disease': disease_name,
            })
        
        # Case 2: Stage 2 存在，Stage 3 删失
        elif (not stage_2_censored) & stage_3_censored:
            result_records.append({
                'eid': id,
                'time': stage_offset + bl2t_disease,
                'stage': 2,
                'event': 1,
                'subtype': subtype,
                'disease': disease_name,
            })
            result_records.append({
                'eid': id,
                'time': end_time,
                'stage': 3,
                'event': 0,
                'subtype': subtype,
                'disease': disease_name,
            })
        
        # Case 3: Stage 2 存在，Stage 3 存在
        elif (not stage_2_censored) & (not stage_3_censored):
            result_records.append({
                'eid': id,
                'time': stage_offset + bl2t_disease,
                'stage': 2,
                'event': 1,
                'subtype': subtype,
                'disease': disease_name,
            })
            result_records.append({
                'eid': id,
                'time': end_time,
                'stage': 3,
                'event': 1,
                'subtype': subtype,
                'disease': disease_name,
            })
        
        # Case 4: Stage 2 删失，Stage 3 存在
        elif stage_2_censored & (not stage_3_censored):
            result_records.append({
                'eid': id,
                'time': end_time,
                'stage': 2,
                'event': 0,
                'subtype': subtype,
                'disease': disease_name,
            })
            result_records.append({
                'eid': id,
                'time': end_time,
                'stage': 3,
                'event': 1,
                'subtype': subtype,
                'disease': disease_name,
            })

# 将结果转换为 DataFrame
result_df = pd.DataFrame(result_records)

result_df.to_csv('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/multistage_life_table.csv', index=False)