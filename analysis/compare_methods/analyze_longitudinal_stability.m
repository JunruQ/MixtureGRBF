function analyze_longitudinal_stability()
close all

dataset_names = {'ukb'};
method_names = {'MixtureGRBF'};
nsubtype_list = 5;
data_splits = {'baseline'};


[confusion_matrices, percentage_consistency] = ...
        run_algorithms_for_longitudinal_stability(...
        dataset_names, method_names, nsubtype_list, data_splits);

example_cm = confusion_matrices{1,1,1,1}; %last_point, ADNI_FSX_HM, FTR_kmeans, 3 subtypes

show_confusion_matrix(example_cm);

data_inds_AD = [1];
mkr_size = [30];
method_offset = [0];
split_offset = [0];
split_marker = {'o'};

method_color = [1,0,0];

graph_hs = [];

figure;
for idx_subtype = 1:length(nsubtype_list)
    nsubtype = nsubtype_list(idx_subtype);
    
    for idx_method = 1:length(method_names)
        x1 = [];
        y1 = [];
        for idx_split = 1:length(data_splits)
            x = nsubtype + method_offset(idx_method) + split_offset(idx_split);
            x = repmat(x, length(dataset_names), 1);
            y = percentage_consistency(idx_split, data_inds_AD, idx_method, idx_subtype);
            y = squeeze(y);
            hold on;
            graph_hs(idx_split, idx_method) = scatter(x, y, mkr_size, ...
                method_color(idx_method,:), 'filled', split_marker{idx_split});
            x1 = [x1; x];
            y1 = [y1; y];
        end
    end
%         boxchart(x1, y1);    
end

% labels = {'FTR - (1:T-1)|T','FTR - 1|(2:T)','SuStaIn - (1:T-1)|T','SuStaIn - 1|(2:T)'};
labels = strrep(method_names, '_', ' ');
legend(graph_hs(1,:), labels);

ylabel('Percentage of consistent subjects', 'Interpreter', 'none');
xlabel('Number of subtypes');


end

function [confusion_matrices, percentage_consistency] = ...
        run_algorithms_for_longitudinal_stability(...
        dataset_names, method_names, nsubtype_list, data_splits)
options = [];

save_path = ['./output/longitudinal_stability'];
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

confusion_matrices = {};
percentage_consistency = [];

for split_idx = 1:length(data_splits)
    split = data_splits{split_idx};
    
    options.data_split = split;
    
    for data_idx = 1:length(dataset_names)
        data_name = dataset_names{data_idx};
        
        for method_idx = 1:length(method_names)
            method_name = method_names{method_idx};
            
            for subtype_idx = 1:length(nsubtype_list)
                nsubtype = nsubtype_list(subtype_idx);
                options.nsubtype = nsubtype;
                
                output_file_name = sprintf('longitudinal_stability/%s/%s_%s', ...
                    split, data_name, method_name);
                options.output_file_name = output_file_name;
                options.case_selector = 'single_test';
                
                save_path_confusion = [save_path, '/confusion_matrix/'];
                if ~exist(save_path_confusion, 'dir')
                    mkdir(save_path_confusion);
                end
                
                save_path_confusion = strcat(save_path_confusion, [split,'_', ...
                        data_name, '_', method_name, '_', num2str(nsubtype), '.csv']);
                
                % Model
                if ~exist(['output/',output_file_name,'/',num2str(nsubtype),'_subtypes/trajectory1.csv'], 'file')
                    disp(['--- Processing ', output_file_name, ' with ', int2str(nsubtype), ' subtypes.']);
                    
                    [subtype_stage, mdl, options, train_data, test_data] = ...
                        run_algo(data_name, method_name, options);
                    close all;

                    switch method_name
                        case 'MixtureGRBF'
                            stage_train = train_data.stage;
                            stage_test = test_data.stage;
                            [subtype_train,~] = cal_subtype(train_data.vols, train_data.RID, train_data.stage, mdl, options);
                            [subtype_test,~] = cal_subtype(test_data.vols, test_data.RID,test_data.stage, mdl, options);
                        case {'FTR_kmeans','FTR_MCEM'}
                            [stage_train,subtype_train,~] = cal_stage_subtype(train_data.vols, train_data.RID, mdl, options);
                            [stage_test,subtype_test,~] = cal_stage_subtype(test_data.vols, test_data.RID, mdl, options);
                        case 'sustain'
                            [stage_train,subtype_train,~] = cal_stage_subtype_sustain(train_data.vols, train_data.RID, mdl, options);
                            [stage_test,subtype_test,~] = cal_stage_subtype_sustain(test_data.vols, test_data.RID, mdl, options);
                    end
                    
                    train_subtype_stage = array2table([train_data.RID, train_data.years, stage_train, subtype_train], ...
                        'VariableNames', {'PTID' ,'years_from_baseline','stage','subtype'});
                    test_subtype_stage = array2table([test_data.RID, test_data.years, stage_test, subtype_test], ...
                        'VariableNames', {'PTID' ,'years_from_baseline','stage','subtype'});

                    % Initialize confusion matrix
                    num_subtypes = options.nsubtype;
                    conf_matrix = zeros(num_subtypes);

                    % Fill confusion matrix based on intersection of PTIDs between train and test subtypes
                    for i = 1:num_subtypes
                        for j = 1:num_subtypes
                            % Find PTIDs of train_data with subtype i and test_data with subtype j
                            train_PTID_subtype_i = unique(train_subtype_stage.PTID(train_subtype_stage.subtype ==  i));
                            test_PTID_subtype_j = unique(test_subtype_stage.PTID(test_subtype_stage.subtype ==  j));

                            % Count the number of common PTIDs between train and test subtypes
                            common_PTIDs = intersect(train_PTID_subtype_i, test_PTID_subtype_j);
                            conf_matrix(i, j) = numel(common_PTIDs);
                        end
                    end
 
                    writematrix(conf_matrix, save_path_confusion);
                elseif ~exist(save_path_confusion, 'file')
                    [~, ~, options, train_data, test_data] = ...
                        run_algo(data_name, method_name, options);
                    stage_train = train_data.stage;
                    stage_test = test_data.stage;
                    mdl = load_parameters_theta('output/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes');
                    [subtype_train,~] = cal_subtype(train_data.vols, train_data.RID, train_data.stage, mdl, options);
                    [subtype_test,~] = cal_subtype(test_data.vols, test_data.RID,test_data.stage, mdl, options);
                    % mdl = load_parameters_theta(['output/',output_file_name,'/',num2str(nsubtype),'_subtypes']);
                    train_subtype_stage = array2table([train_data.RID, train_data.years, stage_train, subtype_train], ...
                        'VariableNames', {'PTID' ,'years_from_baseline','stage','subtype'});
                    test_subtype_stage = array2table([test_data.RID, test_data.years, stage_test, subtype_test], ...
                        'VariableNames', {'PTID' ,'years_from_baseline','stage','subtype'});

                    % Initialize confusion matrix
                    num_subtypes = options.nsubtype;
                    conf_matrix = zeros(num_subtypes);

                    % Fill confusion matrix based on intersection of PTIDs between train and test subtypes
                    for i = 1:num_subtypes
                        for j = 1:num_subtypes
                            % Find PTIDs of train_data with subtype i and test_data with subtype j
                            train_PTID_subtype_i = unique(train_subtype_stage.PTID(train_subtype_stage.subtype ==  i));
                            test_PTID_subtype_j = unique(test_subtype_stage.PTID(test_subtype_stage.subtype ==  j));

                            % Count the number of common PTIDs between train and test subtypes
                            common_PTIDs = intersect(train_PTID_subtype_i, test_PTID_subtype_j);
                            conf_matrix(i, j) = numel(common_PTIDs);
                        end
                    end
 
                    writematrix(conf_matrix, save_path_confusion);
                else
                    conf_matrix = readmatrix(save_path_confusion);
                end
                
                confusion_matrices{split_idx, data_idx, method_idx, subtype_idx} = conf_matrix;
                percentage_consistency(split_idx, data_idx, method_idx, subtype_idx) = ...
                    sum(diag(conf_matrix)) / sum(sum(conf_matrix));
            end
        end
    end
end
end


