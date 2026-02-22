function [outputArg1,outputArg2] = analyze_model_residue(inputArg1,inputArg2)

addpath("FTR_code/");
addpath("utils/");
addpath('analysis');

%% option setting
dataset_name = 'ADNI41_fsl';
method_name = 'FTR';

%test_option = "cv_residue";
test_option = "longitudinal_stability";

options = [];
options.nsubtype = 3;
options.num_folds = 10;
options.num_int = 1001;
options.save = 0;
options.diagnosis_sel = 1;
options.input_file_name = 'ADNI/ADNI_FSL_ro_11_19.csv';
options.output_file_name = 'ADNI/ADNI_FSL_analysis';
options.max_ep = 50;
options.max_iter = 10000;
options.methods = 'kmeans';            
options.parfor = true;
options.samemax = false;
options.sigma_type = '1xK';

output_file_name = options.output_file_name;
nsubtype_list = 3;
save_path = ['./output/',output_file_name];
if ~exist(save_path, 'dir')
    mkdir(save_path)
end

% load train/test index
[train_data,test_data,all_data,biomarker_name,options] = load_dataset(dataset_name, [], [], options);


%% model residue
if test_option == "cv_residue"

    num_folds = options.num_folds;

    rss_mat = train_all_subtypes_folds(train_data.vols, train_data.RID, nsubtype_list, num_folds, save_path, options);

    writematrix(rss_mat,strcat(save_path,"/model_residual.csv"))

%% longitudinal stability
elseif test_option == "longitudinal_stability"

    options.data_split = "last_point";

    % FTR model
    [~, re_traj] =  run_algo(dataset_name, method_name, options);

    [stage_train,subtype_train,~] = cal_stage_subtype(train_data.vols,train_data.RID,re_traj);
    train_subtype_stage = array2table([train_data.RID,train_data.years,train_data.labels,stage_train,subtype_train], ...
        'VariableNames', {'PTID' ,'years_from_baseline','Diagnosis','stage','subtype'});
    [stage_test,subtype_test,~] = cal_stage_subtype(test_data.vols,test_data.RID,re_traj);
    test_subtype_stage = array2table([test_data.RID,test_data.years,test_data.labels,stage_test,subtype_test], ...
        'VariableNames', {'PTID' ,'years_from_baseline','Diagnosis','stage','subtype'});

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

    % Display confusion matrix
    disp('Confusion Matrix:');
    disp(conf_matrix);

    % Example subtypes 
    train_subtypes = {'subtype1', 'subtype2', 'subtype3'};
    test_subtypes = {'subtype1', 'subtype2', 'subtype3'};

    % Plotting the confusion matrix
    figure;
    imagesc(conf_matrix);
    colorbar;

    % Adjusting axis labels and ticks
    set(gca, 'XTick', 1:length(train_subtypes), 'XTickLabel', train_subtypes, 'YTick', 1:length(test_subtypes), 'YTickLabel', test_subtypes);
    xlabel('Test Subtypes');
    ylabel('Train Subtypes');
    title('Confusion Matrix');

    % Displaying values in the cells (optional)
    textStrings = num2str(conf_matrix(:), '%d');
    textStrings = strtrim(cellstr(textStrings));
    [x, y] = meshgrid(1:length(train_subtypes), 1:length(test_subtypes));
    hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
    midValue = mean(get(gca, 'CLim'));
    textColors = repmat(reshape(eye(num_subtypes),[],1), 1, 3);
    %repmat(conf_matrix(:) > midValue, 1, 3);
    set(hStrings, {'Color'}, num2cell(~textColors,2));

    % Adjusting the color scale (optional)
    colormap('gray'); % Change the colormap as needed

    writematrix(conf_matrix,strcat(save_path,"/confusion_matrix.csv"))

end

end

function rss = train_all_subtypes_folds(dat, PTID, nsubtype_list, num_folds, save_path, options)

nnsubtype = length(nsubtype_list);
rss = zeros(nnsubtype,num_folds);

[nsamp,nbiom] = size(dat);
uni_id = unique(PTID);
cv_inds_place = [save_path,'/cv_inds.csv'];

% YZ: added to solve the problem that a specific (nsubtype, fold) met an
% error in model selection
specify_nsubtype_fold_to_run = parse_param(options, 'specify_nsubtype_fold_to_run', []);

if isempty(specify_nsubtype_fold_to_run)
    id_inds = crossvalind('Kfold',length(uni_id),num_folds);
    inds = zeros(nsamp,1);
    for i = 1:nsamp
        inds(i) = id_inds(uni_id==PTID(i));
    end
    writematrix(inds,cv_inds_place);
else
    inds = readmatrix(join([save_path,'/cv_inds.csv']));
end


% YZ: added to select specified (nsubtype,fold) to run in case an error was
% met when running model selection
if isempty(specify_nsubtype_fold_to_run)
    [m,n] = ndgrid(nsubtype_list, 1:num_folds);
    Z = [m(:),n(:)];
else
    Z = specify_nsubtype_fold_to_run;
end

if isempty(gcp('nocreate')), parpool; end

disp('Begin cross validation for model selection:');
fprintf('%s\n', repmat('.',1,size(Z,1)));

start_t = tic;

for i = 1:size(Z,1)
    nsubtype = Z(i,1);
    fold = Z(i,2);
    % FTR algorithm
    % YZ: add the case for fold 0 (use the whole data)
    test_inds = find(inds == fold);
    train_inds = find(inds ~= fold);

    [traj,re_traj,train_subtype,train_stage,sigma,loglik_train] = ...
        FTR_model(dat(train_inds,:),PTID(train_inds,:),nsubtype,options);

    % YZ: add the case for fold 0
    if ~isempty(test_inds)
        %             caloglik = true;
        [test_stage,test_subtype,dist_samp_min] = cal_stage_subtype(...
            dat(test_inds,:),PTID(test_inds,:),re_traj);


        % YZ: changed it to data point-wise mean absolute difference
        % rss = sum(sqrt(sum(dist_samp_min,2)));
        rss(1,fold) = sum(sqrt(sum(dist_samp_min,2)));
    end
end

end

