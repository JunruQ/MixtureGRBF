function [subtype_order, sorted_subtypes] = all_cause_mortality_sort(nsubtype, subtype_path)

disease_dir = './input/disease_info/X0.csv';

subtype_stage_data = readtable(subtype_path);

disease_info = readtable(disease_dir);
disease_info = join(subtype_stage_data, disease_info(:, {'eid', 'target_y', 'BL2Target_yrs'}), 'LeftKeys', 'PTID', 'RightKeys', 'eid');
disease_info.age = disease_info.stage + disease_info.BL2Target_yrs;
disease_info.censored = ~disease_info.target_y;

% 提取数据

subtypes = disease_info.subtype;
unique_subtypes = unique(subtypes);
event = disease_info.censored;
time = disease_info.age;
dummy_subtypes = dummyvar(subtypes);
dummy_subtypes(:,1) = [];
[b,logl,H,stats] = coxphfit(dummy_subtypes, time, 'Censoring', event);
hr = exp(b);
hr = [1; hr];
[~, sorted_indices] = sort(hr);
subtype_order = unique_subtypes(sorted_indices);

map = zeros(1, max(subtype_order));
map(subtype_order) = 1:numel(subtype_order);

% 应用映射
sorted_subtypes = map(subtypes);


end