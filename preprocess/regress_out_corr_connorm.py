import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('input/ukb/ukb_covreg0_trans0_nanf0_biom0.csv')

def regress_out_covariates(df, covariate_columns, continuous_covariates, start_column=7):
    """
    对df中的每一列回归掉指定协变量的影响，并将残差替换为原列的值。

    参数:
    df (pd.DataFrame): 输入数据框。
    covariate_columns (list): 作为协变量的列名列表。
    continuous_covariates (list): 作为连续变量的协变量列名列表（不转换为哑变量）。
    start_column (int): 从第几列开始应用回归分析。

    返回:
    pd.DataFrame: 回归后的数据框。
    """
    # 选择数据子集
    subset_df = df.iloc[:, start_column:]

    # 对每一列进行回归分析
    for column in subset_df.columns:
        if column not in covariate_columns:
            # 定义因变量
            y = subset_df[column]
            
            # 将 continuous_covariates 直接作为协变量
            continuous_data = df[continuous_covariates]
            
            # 对除去连续协变量的类别协变量生成哑变量
            categorical_covariates = [col for col in covariate_columns if col not in continuous_covariates]
            categorical_data = pd.get_dummies(df[categorical_covariates], drop_first=True)
            
            # 合并连续协变量和哑变量
            X = pd.concat([continuous_data, categorical_data], axis=1)

            # 添加常数项
            X = sm.add_constant(X)

            # 去除含有 NaN 的行
            valid_index = y.notna() & X.notna().all(axis=1)
            y_valid = y[valid_index]
            X_valid = X[valid_index]

            # 拟合回归模型
            model = sm.OLS(y_valid, X_valid).fit()

            # 计算残差
            residuals = model.resid

            # 创建一个新的列，保存原始的NaN
            new_column = y.copy()
            new_column.loc[valid_index] = residuals

            # 将残差写入原数据框的对应列
            df.loc[:, column] = new_column

    return df

# 使用函数
covariate_columns = ['sex', 'education', 'centre', 'Ethnic', 'stage']
continuous_covariates = ['education', 'stage']

df.drop_duplicates(subset=['eid'], inplace=True)
df = regress_out_covariates(df, covariate_columns, continuous_covariates)

df_filled = df.copy()
df_filled.iloc[:, 7:] = df.iloc[:, 7:].apply(lambda col: col.fillna(col.mean()), axis=0)
# 归一化
df_trans1 = df_filled.copy()
df_trans1.iloc[:, 7:] = df_trans1.iloc[:, 7:].apply(lambda col: (col - col.mean()) / col.std(), axis=0)
df_trans1.to_csv('input/ukb/ukb_covreg2_trans4_nanf1_biom0.csv', index=False)

with open('input/disease_info/output/control_idx.csv', 'r') as f:
    control_id = list(map(lambda x: int(x.strip()), f.readlines()))

# 过滤出控制组的列
control_data = df.loc[df['RID'].isin(control_id), df.columns[7:]]

# 计算控制组的均值和标准差
control_mean = control_data.mean()
control_std = control_data.std()

# 对 df_filled 进行归一化
df_filled.iloc[:, 7:] = df_filled.iloc[:, 7:].apply(
    lambda col: (col - control_mean[col.name]) / control_std[col.name], axis=0)

# 保存归一化后的数据
df_filled.to_csv('input/ukb/ukb_covreg2_trans2_nanf1_biom0.csv', index=False)
# df_filled.iloc[:, 7:] = df_filled.iloc[:, 7:].apply(lambda col: (col - col.mean()) / col.std(), axis=0)

# df_filled.to_csv('input/ukb/ukb_regress1_selected.csv', index=False)
