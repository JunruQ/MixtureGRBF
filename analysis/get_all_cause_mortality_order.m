close all;

exp_name = 'ukb_MixtureGRBF_test_biom16';
nsubtype = 4;
disease_dir = './input/disease_info/X0.csv';
subtype_path = ['./output/',exp_name,'/',int2str(nsubtype),'_subtypes/subtype_stage.csv'];
output_dir = ['output/result_analysis/',exp_name,'/',int2str(nsubtype),'_subtypes'];

if ~exist(output_dir, "dir")
    mkdir(output_dir)
end

subtype_stage_data = readtable(subtype_path);

disease_info = readtable(disease_dir);
disease_info = join(subtype_stage_data, disease_info(:, {'eid', 'target_y', 'BL2Target_yrs'}), 'LeftKeys', 'PTID', 'RightKeys', 'eid');
disease_info.age = disease_info.stage + disease_info.BL2Target_yrs;
disease_info.censored = ~disease_info.target_y;

% 提取数据
ages = disease_info.age;
censored = disease_info.censored;
subtypes = disease_info.subtype;

unique_subtypes = unique(subtypes);
event = disease_info.censored;
time = disease_info.age;
subtypes = disease_info.subtype;
dummy_subtypes = dummyvar(subtypes);
dummy_subtypes(:,1) = [];
[b,logl,H,stats] = coxphfit(dummy_subtypes, time, 'Censoring', event);
hr = exp(b);
hr = [1; hr];
[~, sorted_indices] = sort(hr);
sorted_subtypes = unique_subtypes(sorted_indices);

writematrix(sorted_subtypes, [output_dir,'/all_cause_mortality_order.csv'])