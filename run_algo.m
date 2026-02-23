function [subtype_stage, mdl, options, train_data, test_data] = ...
    run_algo(dataset_name, method_name, options)

addpath(genpath('mixtureGRBF'));
addpath(genpath('utils'));
addpath(genpath('analysis'));

if nargin < 3
    options = [];
end

if nargin < 2
    method_name = 'MixtureGRBF';
end

if nargin < 1
    dataset_name = 'synth';
end

options = insert_param_when_absent(options,'case_selector','cv_nsubtype');

method_name1 = method_name;

switch method_name
    case 'MixtureGRBF'
        options = insert_param_when_absent(options, 'nsubtype', 1:6);

        options.num_folds = 10;

        options.max_iter = 20;

        options.max_ep = 100;

        options.num_int = 101;

        options.samemax = false;

        options.gaussian_interval = 1;

        method_name1 = method_name;

    otherwise
end

options.dataset_name = dataset_name;

switch dataset_name
    case 'ukb'
        options.input_file_name = 'ukb/ukb_covreg1_trans1_nanf1_biom17.csv';
        options = insert_param_when_absent(options, 'data_split', 'baseline'); % Junru: When case is site_validation, we split the data in training and testing.
        options = insert_param_when_absent(options, 'group_sel', false);
        options = insert_param_when_absent(options, 'stagsel', false);
        options = insert_param_when_absent(options, 'diagnosis_sel', false);
        options = insert_param_when_absent(options, 'site_sel', true);
        options = insert_param_when_absent(options, 'biomarker_column_range', [8,0]);
        options = insert_param_when_absent(options, 'site_table_file_name', 'ukb/ukb_site_map.csv');
    case 'synth'
        options.input_file_name = 'synth/synth_data_z.csv';
        options = insert_param_when_absent(options, 'data_split', 'baseline');
        options = insert_param_when_absent(options, 'group_sel', false);
        options = insert_param_when_absent(options, 'stagsel', false);
        options = insert_param_when_absent(options, 'diagnosis_sel', false);
        options = insert_param_when_absent(options, 'site_sel', false);
        options = insert_param_when_absent(options, 'biomarker_column_range', [3,0]);
    otherwise
end

case_selector = options.case_selector;

switch case_selector
    case 'cv_lambda'
        options = insert_param_when_absent(options, 'lambda', 2.^linspace(-1,4,6));
    case 'search_lambda'
        options = insert_param_when_absent(options, 'lambda', 1);
    case 'cv_nsubtype'
        % 指定subtype_lambda列表，若无则留空
        options.parfor = false;
        options = insert_param_when_absent(options, 'parfor_init', true);
        sl_path = './input/ukb/subtype_lambda.csv';
        if exist(sl_path, 'file')
            sl_table = readtable(sl_path);
            options = insert_param_when_absent(options, 'nsubtype_lamdba_map', containers.Map(sl_table{:,1}, sl_table{:,2}));
        else
            options = insert_param_when_absent(options, 'nsubtype_lamdba_map', []);
        end
    case 'site_validation'
        options = insert_param_when_absent(options, 'lambda', 1);
        options = insert_param_when_absent(options, 'site_sel', true);
        options = insert_param_when_absent(options, 'site_selected', 'North England');
    otherwise
        insert_param_when_absent(options, 'lambda', 1);

end

default_output = sprintf('%s_%s_%s', dataset_name, method_name1, case_selector);
options = insert_param_when_absent(options, 'output_file_name', default_output);


%% Read data

train_inds = parse_param(options,'train_inds',[]);
test_inds = parse_param(options,'test_inds',[]);

[train_data,test_data,all_data,biomarker_name,options] = load_dataset(dataset_name, train_inds, test_inds, options);

%% Run FTR
start_t = tic;

nsubtype = options.nsubtype;

% [~, save_dir] = get_save_result_filepath(options, nsubtype(1));
% save([save_dir, '/all_data.mat'])

% YZ: I check this part to make 'FTR' a changable method in case we need to
% compare with additional matlab algorithms.
switch method_name
    case 'MixtureGRBF'
        train_vol_sel = train_data.vols;
        train_PTID_sel = train_data.RID;
        train_stage_sel = train_data.stage;
        switch case_selector
            case 'cv_lambda'
                options.parfor = false;
                loglik = cross_validation_lambda(train_vol_sel,train_PTID_sel,train_stage_sel,nsubtype,options);
                loglik_table = array2table(loglik, "VariableNames", {'lambda', 'loglikelihood'});
                output_folder = ['output/',options.output_file_name];
                writetable(loglik_table,[output_folder,'/','loglik_table.csv']);
            case 'cv_nsubtype'
                % cross validation
                [pre_C, ~] = compute_C_and_L(all_data.stage, options);
                options.C = pre_C;
                layered_MGRBF_model_selection(train_vol_sel,train_PTID_sel,train_stage_sel,options);

                for nsubtype1 = options.nsubtype
                    save_dir_nsubtype1 = ['output/', options.output_file_name, ...
                        '/cross_validation_nsubtype', int2str(nsubtype1), '_fold0'];

                    % [sigma, re_traj, proption]
                    mdl = load_parameters_theta(save_dir_nsubtype1);

                    subtype_all = cal_subtype(all_data.vols, all_data.RID, all_data.stage, mdl, options);

                    [save_dir1, save_dir] = get_save_result_filepath(options, nsubtype1);

                    save_result(all_data, all_data.stage, subtype_all, ...
                        mdl, biomarker_name, [], save_dir1);
                end

                return;
            case 'check_stability'
                disp('Checking stability of the model');
                subtype_results = [];
                run_time = parse_param(options, 'run_time', 10);
                for run = 1:run_time
                    disp(['Run ', num2str(run), ' of ', num2str(run_time)]);
                    [pre_C, ~] = compute_C_and_L(all_data.stage, options);
                    options.C = pre_C;
                    options.stage = train_data.stage;
                    [mdl,subtype,extra] = layered_MGRBF_model( ...
                        train_vol_sel,train_PTID_sel,train_stage_sel,nsubtype,options);

                    % predict test
                    stage_all = all_data.stage;
                    [subtype_all,extra] = cal_subtype(all_data.vols, ...
                        all_data.RID, all_data.stage, mdl, options);
                    subtype_results = [subtype_results, subtype_all];
                end
                similarity_mat = zeros(run_time, run_time);
                for i = 1:run_time
                    for j = 1:run_time
                        similarity_mat(i, j) = rand_index(subtype_results(:, i), subtype_results(:, j));
                    end
                end
                disp(similarity_mat);
            case 'test'
                options = insert_param_when_absent(options, 'parfor_init', true);

                [pre_C, ~] = compute_C_and_L(all_data.stage, options);
                options.C = pre_C;
                options.stage = train_data.stage;
                [mdl,subtype,~] = layered_MGRBF_model( ...
                    train_vol_sel,train_PTID_sel,train_stage_sel,nsubtype,options);

                % predict test
                stage_all = all_data.stage;
                [subtype_all,extra] = cal_subtype(all_data.vols, ...
                    all_data.RID, all_data.stage, mdl, options);
                disp('Clustering Results:');
                tabulate(subtype_all)

            case 'search_lambda'
                [pre_C, ~] = compute_C_and_L(all_data.stage, options);
                options.C = pre_C;
                options.stage = train_data.stage;
                ini_lambda = options.lambda;
                subtype_lambda = [];
                nsubtype_list = options.nsubtype;
                for nsubtype = nsubtype_list
                    options.nsubtype = nsubtype;
                    bes_lambda = search_lambda(train_vol_sel,train_PTID_sel,train_stage_sel,nsubtype,ini_lambda,options);
                    ini_lambda = bes_lambda;
                    subtype_lambda = [subtype_lambda; [nsubtype, bes_lambda]];
                end
                writetable(array2table([subtype_lambda(:,1),subtype_lambda(:,2)],'VariableNames',{'nsubtype','lambda'}),['output/',options.output_file_name,'/subtype_lambda.csv']);

            case 'single_test'
                [pre_C, ~] = compute_C_and_L(all_data.stage, options);
                options.C = pre_C;
                [mdl,subtype,extra] = mixtureGRBF(train_vol_sel,train_PTID_sel,train_stage_sel, nsubtype, [], options);
                stage_all = all_data.stage;
                [subtype_all,extra] = cal_subtype(all_data.vols, ...
                    all_data.RID, all_data.stage, mdl, options);
            case 'site_validation'
                [pre_C, ~] = compute_C_and_L(all_data.stage, options);
                options.C = pre_C;
                ini_lambda = options.lambda;
                disp(['Searching lambda for ', options.site_selected, '...']);
                bes_lambda = search_lambda(train_vol_sel,train_PTID_sel,train_stage_sel,nsubtype,ini_lambda,options);
                options.lambda = bes_lambda;
                [mdl,subtype,~] = layered_MGRBF_model( ...
                    train_vol_sel,train_PTID_sel,train_stage_sel,nsubtype,options);

                % predict test
                stage_all = all_data.stage;
                [subtype_all,extra] = cal_subtype(all_data.vols, ...
                    all_data.RID, all_data.stage, mdl, options);
                disp('Clustering Results:');
                tabulate(subtype_all)
        end
    otherwise
end

fprintf('The method takes %.1f seconds\n', toc(start_t));

subtype_stage = array2table([all_data.RID,all_data.years,stage_all,subtype_all], ...
    'VariableNames', {'PTID' ,'years_from_baseline','stage','subtype'});


%% save output

[save_dir1, save_dir] = get_save_result_filepath(options, nsubtype);
% save([save_dir, '/all_data.mat'])

save_result(all_data, stage_all, subtype_all, mdl, ...
    biomarker_name, extra, save_dir1);

% num_pic = 11;

% YZ: An error popped up when running the depression data. I commented it
% out.
% sel_traj(re_traj,num_pic,biomarker_name,save_dir);

%% analysis

if 0
    % Display trajectory PCA
    show_trajectory_in_PCA(nsubtype, re_traj, all_data.vols, subtype_all);
end

%saveas(h,[save_dir,'\trajectory_PCA_subtype',int2str(nsubtype),'.fig']);

end

function [save_dir1, save_dir] = get_save_result_filepath(options, nsubtype)
save_dir = ['./output/',options.output_file_name];
save_dir1 = [save_dir, '/',num2str(nsubtype),'_subtypes'];
if options.site_sel
    save_dir1 = [save_dir1,'/',options.site_selected];
end
if ~exist(save_dir1, 'dir')
    mkdir(save_dir1)
end

end


