function [train_data,test_data,all_data,biomarker_name,options] = load_dataset(dataset_name, train_inds, test_inds, options)
%LOAD_DATASET Summary of this function goes here
%   Detailed explanation goes here

input_file_name = options.input_file_name;

input_data = readtable(['./input/',input_file_name],'VariableNamingRule','preserve');

if strcmp(dataset_name, 'ukb')
    input_data = input_data(input_data.stage<=70,:);
end

PTID= input_data.RID;

nsamp = size(input_data,1);

rem_inds = ones(nsamp,1);

fprintf('Original number of individuals: %d. \n', length(unique(PTID)));

if options.group_sel
    idx_case = input_data.group == 1;
    idx_control = input_data.group == 0;
    
    show_data_statistics(input_data, idx_case, idx_control);

    rem_inds1 = input_data.group;
    rem_inds1(isnan(rem_inds1)) = 0;

    rem_inds = rem_inds & rem_inds1;

    fprintf('After group selection, %d individuals left. \n', length(unique(PTID(rem_inds))));
end

if options.diagnosis_sel
    % Keep time series that have at least 1 AD time point
    PTID_abn = unique(PTID(labels==1));
    rem_inds1 = zeros(nsamp,1);
    for i = 1:length(PTID_abn)
        rem_inds1(PTID==PTID_abn(i)) = 1;
    end

    rem_inds = rem_inds & rem_inds1;

    fprintf('After diagnosis selection, %d individuals left\n', ...
        length(unique(PTID(rem_inds))));

end

if options.site_sel
    site_table_path = ['./input/', options.site_table_file_name];
    site_table = readtable(site_table_path,'ReadVariableNames', false);
    site_table_idx = strcmp(site_table.Var2, options.site_selected);
    site_ids = site_table.Var1(site_table_idx);
    rem_inds1 = ismember(input_data.centre, site_ids);
    rem_inds = rem_inds - rem_inds1;
end

bcr = parse_param(options, 'biomarker_column_range', [4,0]);
if bcr(2) <= 0
    biomarker_name = input_data.Properties.VariableNames(bcr(1):end+bcr(2));
    bioms = input_data{:,bcr(1):end+bcr(2)};
else
    biomarker_name = input_data.Properties.VariableNames(bcr(1):bcr(2));
    bioms = input_data{:,bcr(1):bcr(2)};
end

data_split = parse_param(options, 'data_split', '');

% split into training and testing
if isempty(train_inds) || isempty(test_inds)
    rem_list = find(rem_inds == 1);
    switch data_split
        case 'cross_patients'
            PTID_uniq = unique(PTID(rem_list));
            N = length(PTID_uniq);
            [trainInd,testInd] = dividerand(N,0.8,0.2);
            train_inds = search_id(PTID_uniq(trainInd),PTID);
            test_inds = search_id(PTID_uniq(testInd),PTID);
        case 'cross_points'
            N = sum(rem_inds);
            [trainInd,testInd] = dividerand(N,0.8,0.2);
            train_inds = arr2vec(rem_list(trainInd),nsamp);
            test_inds = arr2vec(rem_list(testInd),nsamp);    
        case 'cross_site'
            train_inds = rem_inds;
            test_inds = ones(nsamp,1) - train_inds;
        case 'last_point'
            PTID_uniq = unique(PTID(rem_list));
            N = length(PTID_uniq);
            test_inds = zeros(nsamp,1);
            for i = 1:N
                inds = find(PTID == PTID_uniq(i) & rem_inds == 1);
                % If a subject has only 1 point, this point is assigned
                % to the training set
                if length(inds) > 1
                    test_inds(inds(end)) = 1;
                end
            end
            train_inds = rem_inds - test_inds;
        case 'baseline'
            PTID_uniq = unique(PTID(rem_list));
            N = length(PTID_uniq);
            train_inds = zeros(nsamp,1);
            % If a subject has only 1 point, this point is assigned to the
            % training set
            for i = 1:N
                inds = find(PTID == PTID_uniq(i) & rem_inds == 1);
                train_inds(inds(1)) = 1;
            end
            test_inds = rem_inds - train_inds;
        otherwise
            train_inds = rem_inds;
            test_inds = zeros(nsamp,1);
    end
end

options.train_inds = train_inds;
options.test_inds = test_inds;

train_data = read_inds(input_data,bioms,train_inds);
test_data = read_inds(input_data,bioms,test_inds);
all_data = read_inds(input_data,bioms,ones(nsamp,1));

end

function data = read_inds(data_all,bioms,inds)
data = [];
data.RID = data_all.RID(inds==1);
data.stage = data_all.stage(inds == 1);
% data.stage = data_all.age(inds==1) + data_all.years(inds==1);
data.years = data.stage;
data.vols = bioms(inds==1,:);
end

function vec = arr2vec(arr,len)
    vec = zeros(len,1);
    for i = arr
        vec(i) = 1;
    end
end

function vec = search_id(PTID_sel,PTID)
    vec = zeros(length(PTID),1);
    for i = 1:length(PTID_sel)
        vec(PTID == PTID_sel(i)) = 1;
    end
end

function show_data_statistics(input_data, idx_AD, idx_CN)
PTID= input_data.RID;

fprintf('Case points/subjects: %d/%d; control points/subjects: %d/%d \n', ...
        length(find(idx_AD)), length(unique(PTID(idx_AD))), ...
        length(find(idx_CN)), length(unique(PTID(idx_CN))));
disp('Case (row 1)/control (row 2) age mean, age std, gender(male%)');
ages_AD = input_data.AGE(idx_AD);
genders_AD = input_data.gender(idx_AD);
ages_CN = input_data.AGE(idx_CN);
genders_CN = input_data.gender(idx_CN);
[mean(ages_AD), std(ages_AD), length(find(genders_AD == 1))/length(genders_AD); ...
    mean(ages_CN), std(ages_CN), length(find(genders_CN == 1))/length(genders_CN)]
end