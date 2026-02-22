function get_all_cause_mortality_order_site_validation()

input_data = readtable('input/ukb/ukb_covreg1_trans1_nanf1_biom17.csv');

site_table_path = 'input/ukb/ukb_site_map.csv';
site_table = readtable(site_table_path, 'ReadVariableNames', false);

nsubtype = 5;

sites = {'North England', 'South England', 'Midlands', 'Scotland', 'Wales'};   

for site_idx = 1:numel(sites)
    site = sites{site_idx};
    
    site_result_subtype_path = ['output/ukb_MixtureGRBF_site_validation/5_subtypes/', site, '/subtype_stage.csv'];
    [site_order, ~] = all_cause_mortality_sort(nsubtype, site_result_subtype_path);
    writematrix(site_order, ['output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/site_val/', site, '_order.csv']);
end

end