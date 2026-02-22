function [outputArg1,outputArg2] = predict_atrophy_progression(inputArg1,inputArg2)
%PREDICT_ATROPHY_PROGRESSION Summary of this function goes here
%   Detailed explanation goes here
close all

data = {'ADNI_FSX_LD'};
method = {'FTR'};

nsubtype = 3;

results = load_data_result(data, method, nsubtype);
joindata = results{1,1}.joindata;
traj = results{1,1}.traj;
biomarker_names = results{1,1}.biomarker_names;


%% create data set
[predictors, target, RIDs] = create_data_for_prediction(joindata, biomarker_names);

%% perform prediction using 100 random splits
train_ratio = 0.8;

Rs = [];
test_target_all = [];
test_predict_target_all = [];

% methods = {'lasso_on_regional_volumes', 'lasso_on_subtypes_stages_from_FTR'};
methods = {'lasso_on_regional_volumes', 'trajectory_derivative_from_FTR'};
% methods = {'lasso_on_regional_volumes', 'delta_stage2change_FTR'};

for split = 1:1
    % split the subjects into 80% training and 20% test
    [train_predictors, train_target, test_predictors, test_target] = ...
        split_based_on_subjects(predictors, target, RIDs, train_ratio);
    
    for idx_method = 1:length(methods)
        eval(['test_predict_target = ',methods{idx_method}, ...
            '(train_target, train_predictors, ', ...
            'test_predictors, biomarker_names, traj);']);        

        %% evaluate
        % calculate the correlation of each data point
        R = [];
        for i = 1:size(test_target,1)
            R_i = corrcoef(test_target(i,:), test_predict_target(i,:));
            R(i) = R_i(1,2);
        end
        Rs{split,idx_method} = R(:);

        test_target_all{split,1,idx_method} = test_target;
        test_predict_target_all{split,1,idx_method} = test_predict_target;
    end
end

Rs = cell2mat(Rs);
figure, boxplot(Rs);

test_target_all = cell2mat(test_target_all);
test_predict_target_all = cell2mat(test_predict_target_all);

biomarkers_comp = ["Frontal","Parietal","Temporal","Hippocampus"];

figure;
for idx_method = 1:length(methods)
    for j = 1:length(biomarkers_comp)
        subplot(1,length(biomarkers_comp),j);
        x = test_target_all(:, biomarker_names == biomarkers_comp(j), idx_method);
        y = test_predict_target_all(:, biomarker_names == biomarkers_comp(j), idx_method);
        scatter(x,y,5,[0.2,0.2,0.2],'filled');
        R = corrcoef(x,y);
        legend(['r = ',num2str(R(1,2))]);
        xlabel({'Observed change in zscore (sigma/year)', biomarkers_comp(j)});
        ylabel('Predicted change in zscore (sigma/year)');
    end
end

end



%% methods for prediction
function test_predict_target = trajectory_derivative_from_FTR(train_target, ...
    train_predictors, test_predictors, ...
    biomarker_names, traj)
num_int = size(traj, 2);
stages = ((1:num_int)-1)/(num_int-1);
delta_stage = 1/(num_int-1);

test_predict_target = [];
for i = 1:size(test_predictors,1)
    subtype_i = test_predictors{i,'subtype'};
    stage_i = test_predictors{i,'stage'};
    stage_next = min(stage_i + 0.1, 1);
    stage_prev = max(stage_i - 0.1, 0);
    
    
    [~,ind_i] = min(abs(stages - stage_i));
    [~,ind_next] = min(abs(stages - stage_next));
    [~,ind_prev] = min(abs(stages - stage_prev));
    
    derivative = (traj(:,ind_next,subtype_i) - traj(:,ind_prev,subtype_i))...
        /(stage_next - stage_prev);
    test_predict_target(i,:) = derivative;
end

end

function test_predict_target = delta_stage2change_FTR(train_target, ...
    train_predictors, test_predictors, ...
    biomarker_names, traj)

% first predict stage change per year using age, sex, education, APOE4,
% MMSE
X = construct_predictors_from_subtypes(train_predictors, ...
    [biomarker_names,{'AGE','PTGENDER','PTEDUCAT','APOE4'}]);
train_elapsed_years = train_predictors{:,'pred_elapsed_years'};
X = X .* repmat(train_elapsed_years, 1, size(X,2));

[B,FitInfo] = lasso(X, train_predictors{:,'pred_elapsed_stage'},'CV',10);
idxLambdaMinMSE = FitInfo.IndexMinMSE;
coef = B(:,idxLambdaMinMSE);
coef0 = FitInfo.Intercept(idxLambdaMinMSE);
train_predict_stage_change = X*coef + repmat(coef0, size(train_predictors,1), 1);

X = construct_predictors_from_subtypes(test_predictors, ...
    {'AGE','PTGENDER','PTEDUCAT','APOE4','MMSE'});
test_elapsed_years = test_predictors{:,'pred_elapsed_years'};
X = X .* repmat(test_elapsed_years, 1, size(X,2));
test_predict_stage_change = X*coef + repmat(coef0, size(test_predictors,1), 1);

% then take the point at the trajectory to compute the change
test_predict_target = [];
end

function test_predict_target = lasso_on_subtypes_stages_from_FTR(train_target, ...
    train_predictors, train_elapsed_years, test_predictors, test_elapsed_years, ...
    biomarker_names, traj)
Coef = [];
Coef0 = [];
for j = 1:size(train_target,2)
%     X = train_predictors{:,[{'subtype','stage'}, {'AGE','PTGENDER','PTEDUCAT','APOE4'}]};
    X = construct_predictors_from_subtypes(train_predictors, biomarker_names);
    train_elapsed_years = train_predictors{:,'pred_elapsed_years'};
    X = X .* repmat(train_elapsed_years, 1, size(X,2));
    [B,FitInfo] = lasso(X, train_target(:,j),'CV',10);

    idxLambdaMinMSE = FitInfo.IndexMinMSE;

    coef = B(:,idxLambdaMinMSE);
    coef0 = FitInfo.Intercept(idxLambdaMinMSE);

    Coef(:,j) = coef;
    Coef0(:,j) = coef0;
end

% X = test_predictors{:,[{'subtype','stage'}, {'AGE','PTGENDER','PTEDUCAT','APOE4'}]};
X = construct_predictors_from_subtypes(test_predictors, biomarker_names);
test_elapsed_years = test_predictors{:,'pred_elapsed_years'};
X = X .* repmat(test_elapsed_years, 1, size(X,2));
test_predict_target = X*Coef + repmat(Coef0, size(test_predictors,1), 1);
end

function predictors1 = construct_predictors_from_subtypes(predictors, biomarker_names)
% construct predictors by stratifying the original predictors by the
% subtypes
subtypes = predictors{:,'subtype'};
unique_subtypes = unique(subtypes);
K = length(unique_subtypes);
D = length(biomarker_names);

predictors1 = zeros(size(predictors,1), K*D);
for k = 1:K
    predictors1(subtypes == k, (k-1)*D+1:k*D) = predictors{subtypes == k, biomarker_names};
end

% predictors1 = cat(2, predictors1, predictors{:,{'AGE','PTGENDER','PTEDUCAT','APOE4'}});
end

function test_predict_target = lasso_on_regional_volumes(train_target, ...
    train_predictors, test_predictors, ...
    biomarker_names, traj)
% use lasso on all the regional volumes and covariates to predict the
% change
Coef = [];
Coef0 = [];
for j = 1:size(train_target,2)
    X = train_predictors{:,[biomarker_names, {'AGE','PTGENDER','PTEDUCAT','APOE4'}]};
    train_elapsed_years = train_predictors{:,'pred_elapsed_years'};
    X = X .* repmat(train_elapsed_years, 1, size(X,2));
    
    [B,FitInfo] = lasso(X, train_target(:,j),'CV',10);

    idxLambdaMinMSE = FitInfo.IndexMinMSE;

    coef = B(:,idxLambdaMinMSE);
    coef0 = FitInfo.Intercept(idxLambdaMinMSE);

    Coef(:,j) = coef;
    Coef0(:,j) = coef0;
end

X = test_predictors{:,[biomarker_names, {'AGE','PTGENDER','PTEDUCAT','APOE4'}]};
test_elapsed_years = test_predictors{:,'pred_elapsed_years'};
X = X .* repmat(test_elapsed_years, 1, size(X,2));
test_predict_target = X*Coef + repmat(Coef0, size(test_predictors,1), 1);
end

%% data creation
function [train_predictors, train_target, test_predictors, test_target] = ...
    split_based_on_subjects(predictors, target, RIDs, train_ratio)
unique_RIDs = unique(RIDs); % 520 subjects with at least 2 visits with an interval >= 1 year
N = length(unique_RIDs);
idx = randperm(N);
train_ids = unique_RIDs(idx(1:round(N*train_ratio)))';
test_ids = unique_RIDs(idx(round(N*train_ratio)+1:end))';

train_predictors = table();
train_target = [];
for id = train_ids
    train_predictors = cat(1, train_predictors, predictors(RIDs == id, :));
    train_target = cat(1, train_target, target(RIDs == id, :));
end

test_predictors = table();
test_target = [];
for id = test_ids
    test_predictors = cat(1, test_predictors, predictors(RIDs == id, :));
    test_target = cat(1, test_target, target(RIDs == id, :));
end
    
end

function [predictors, target, RIDs_predict] = create_data_for_prediction(joindata, biomarker_names)
RIDs = unique(joindata.RID);

data_adjacent = table();
RID_adjacent = [];

% for each subject (time series), a visit is paired with its closest follow up
% that is apart for at least 1 year, and the status of the earlier visit is
% used to predict the annualized change of atrophy between the two visits.
for i = 1:length(RIDs)
    data_i = joindata(RIDs(i) == joindata.RID, :);
    if size(data_i, 1) > 1
        for t = 1:size(data_i, 1)-1
            % finding the visit that has an interval closest to 1 still
            % produces values -136, 90
%             [min_interval, min_ind] = min(abs(data_i{t+1:end,'years'} - data_i{t,'years'} - 1));
%             t2 = t + min_ind;
%             data_adjacent = cat(1, data_adjacent, [data_i(t, :); data_i(t2,:)]);
%             RID_adjacent = cat(1, RID_adjacent, [RIDs(i); RIDs(i)]);

            for t2 = t+1:size(data_i, 1)
                if data_i{t2,'years'} - data_i{t,'years'} >= 1 
%                     && data_i{t2,'years'} - data_i{t,'years'} <= 2
                    data_adjacent = cat(1, data_adjacent, [data_i(t, :); data_i(t2,:)]);
                    RID_adjacent = cat(1, RID_adjacent, [RIDs(i); RIDs(i)]);
                    break
                end
            end
            
            % finding the immediate next point has most stage changes around 0 
%             t2 = t + 1;
%             data_adjacent = cat(1, data_adjacent, [data_i(t, :); data_i(t2,:)]);
%             RID_adjacent = cat(1, RID_adjacent, [RIDs(i); RIDs(i)]);
        end
    end
end

% calculate the annualized change as the target (3756 points)
target = data_adjacent{2:2:end, biomarker_names} - data_adjacent{1:2:end-1, biomarker_names};

pred_elapsed_years = data_adjacent{2:2:end, 'years'} - data_adjacent{1:2:end-1, 'years'};
pred_elapsed_stage = data_adjacent{2:2:end, 'stage'} - data_adjacent{1:2:end-1, 'stage'};
% elapsed_years = repmat(elapsed_years, 1, size(target,2));
% target = target ./ elapsed_years;

% use the first point in the adjacent pair as predictors
predictors = data_adjacent(1:2:end-1, :);

predictors = addvars(predictors, pred_elapsed_years,'After','years');
predictors = addvars(predictors, pred_elapsed_stage,'After','years');

RIDs_predict = RID_adjacent(1:2:end-1);

% remove pairs that have intervals less than 6 months to avoid too large
% target
% lower_bd = median(target(:)) - 3*mad(target(:),1);
% upper_bd = median(target(:)) + 3*mad(target(:),1);
% outliers = any(target < lower_bd | target > upper_bd, 2);
% predictors(outliers, :) = [];
% target(outliers, :) = [];
end
