function analyze_proteomics()
close all;

% Config

filter_method = 'pro_age_corr_spearman'; % 'pro_age_corr_pearson', 'pro_age_corr_spearman', 'logit_lasso', 'ttest'

data = readtable('input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv');

biom_name = data.Properties.VariableNames(8:end);

if strcmp(filter_method, 'logit_lasso')    
    disease_dir = './input/disease_info/';
    subtype_path = './output/ukb_MixtureGRBF_cv_nsubtype/6_subtypes/subtype_stage.csv';
    
    subtype_stage_data = readtable(subtype_path);
    files = dir(disease_dir);
    
    csvFiles = files((endsWith({files.name}, '.csv')) & (~startsWith({files.name}, 'code')));
    num_files = length(csvFiles);
    
    disease_code_file = files(startsWith({files.name}, 'code'));
    disease_code_table = readtable(fullfile(disease_dir, disease_code_file.name));
    
    if ~exist('./input/disease_info/output/control_idx.csv','file') % get control group
        control_id = get_control_id(disease_dir, subtype_stage_data, csvFiles, num_files);
    else
        control_id = readmatrix('./input/disease_info/output/control_idx.csv');
    end

    pro_indices = [];

    for i = 1:num_files
        if strcmp(csvFiles(i).name, 'A0.csv')
            continue
        end
        disease_info = readtable(fullfile(disease_dir, csvFiles(i).name));
        disease_info = join(subtype_stage_data, disease_info(:, {'eid', 'target_y', 'BL2Target_yrs'}), 'LeftKeys', 'PTID', 'RightKeys', 'eid');
        disease_info.age = disease_info.stage + disease_info.BL2Target_yrs;
        disease_info.censored = ~disease_info.target_y;
        case_id = disease_info.PTID(disease_info.censored == 0);
        control_data = data{ismember(data.RID,control_id),8:end};
        case_data = data{ismember(data.RID,case_id),8:end};
        non_zero_indices = get_protemics_lassoglm(control_data, case_data);
        pro_indices = union(pro_indices, non_zero_indices);
    end
    biom_name_filtered = biom_name(pro_indices);
    save('./input/disease_info/output/ukb_biom_filtered_logit_lasso.mat', 'biom_name_filtered');
    

%% protein age correlation
elseif strcmp(filter_method, 'pro_age_corr_pearson')
    X = data{:,8:end};
    age = data.stage;

    % 初始化存储相关系数和 p 值的数组
    numCols = size(X, 2);
    pValues = zeros(1, numCols);
    
    % 计算每一列与 age 的相关系数和 p 值
    for col = 1:numCols
        [~, P] = corrcoef(X(:, col), age);
        pValues(col) = P(1, 2);
    end
    
    biom_name_filtered = biom_name(pValues<0.05);
    save('./input/disease_info/output/ukb_biom_filtered_protein_age_correlation_pearson.mat', 'biom_name_filtered');


elseif strcmp(filter_method, 'ttest')    
    disease_dir = './input/disease_info/';
    subtype_path = './output/ukb_MixtureGRBF_cv_nsubtype/6_subtypes/subtype_stage.csv';
    
    subtype_stage_data = readtable(subtype_path);
    files = dir(disease_dir);
    
    csvFiles = files((endsWith({files.name}, '.csv')) & (~startsWith({files.name}, 'code')));
    num_files = length(csvFiles);
    
    disease_code_file = files(startsWith({files.name}, 'code'));
    disease_code_table = readtable(fullfile(disease_dir, disease_code_file.name));
    
    if ~exist('./input/disease_info/output/control_idx.csv','file') % get control group
        control_id = get_control_id(disease_dir, subtype_stage_data, csvFiles, num_files);
    else
        control_id = readmatrix('./input/disease_info/output/control_idx.csv');
    end

    pro_indices = [];

    for i = 1:num_files
        if strcmp(csvFiles(i).name, 'A0.csv')
            continue
        end
        disease_info = readtable(fullfile(disease_dir, csvFiles(i).name));
        disease_info = join(subtype_stage_data, disease_info(:, {'eid', 'target_y', 'BL2Target_yrs'}), 'LeftKeys', 'PTID', 'RightKeys', 'eid');
        disease_info.age = disease_info.stage + disease_info.BL2Target_yrs;
        disease_info.censored = ~disease_info.target_y;
        case_id = disease_info.PTID(disease_info.censored == 0);
        control_data = data{ismember(data.RID,control_id),8:end};
        case_data = data{ismember(data.RID,case_id),8:end};
        indices = p_ttest(control_data, case_data);
        pro_indices = union(pro_indices, indices);
    end
    biom_name_filtered = biom_name(pro_indices);
    save('./input/disease_info/output/ukb_biom_filtered_ttest.mat', 'biom_name_filtered');

elseif strcmp(filter_method, 'pro_age_corr_spearman')
    X = data{:,8:end};
    age = data.stage;

    % 初始化存储相关系数和 p 值的数组
    numCols = size(X, 2);
    pValues = zeros(1, numCols);
    
    % 计算每一列与 age 的相关系数和 p 值
    for col = 1:numCols
        [~, P] = corr(X(:, col), age, "type", "Spearman");
        pValues(col) = P;
    end
    
    biom_name_filtered = biom_name(multi_test_correction(pValues, 0.05, 'bonferroni'));
    fileID = fopen('preprocess/significant_biomarker/spearman.txt', 'w');
    
    for i = 1:length(biom_name_filtered)
        fprintf(fileID, '%s\n', biom_name_filtered{i});
    end
    
    fclose(fileID);
end

end

function non_zero_indices = get_protemics_lassoglm(control_data, case_data)
    % 合并对照组和病例组数据
    X = [control_data; case_data];
    y = [zeros(size(control_data, 1), 1); ones(size(case_data, 1), 1)];
    
    Options = statset(UseParallel=true);
    % 使用lassoglm进行LASSO逻辑回归
    [B, FitInfo] = lassoglm(X, y, 'binomial', 'CV', 10, 'Options', Options);

    % 获取最优lambda对应的非零系数
    idxLambda1SE = FitInfo.Index1SE;
    coef = B(:, idxLambda1SE);
    non_zero_indices = find(coef ~= 0);
end

function indices = p_ttest(control_data, case_data)

% 获取列数
num_columns = size(control_data, 2);

% 创建一个向量来存储p值
p_values = zeros(1, num_columns);

% 遍历每一列并进行t检验
for i = 1:num_columns
    % 进行t检验
    [~, p_values(i)] = ttest2(control_data(:, i), case_data(:, i));
end

% 将p值转换为显著性水平
alpha = 0.05;
indices = find(multi_test_correction(p_values, alpha, 'bonferroni') == 1);
end