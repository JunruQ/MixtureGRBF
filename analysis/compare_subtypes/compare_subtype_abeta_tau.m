function compare_subtype_abeta_tau()
close all
addpath(genpath('gpml/gpml-matlab-v4.2-2018-06-11'));
% %addpath("ADNI_data/Code/github_repo/")
% 
% input = readtable('./ADNI_data/Result/ADNI_demo_MRI_PET_ro_11_18.csv',VariableNamingRule='preserve');
% subtype_stage = readtable('./ADNI_data/Result/FSX_subtype_stage.csv',VariableNamingRule='preserve');
% 
% subtype_stage = renamevars(subtype_stage, 'PTID', 'RID');
% subtype_stage = renamevars(subtype_stage, 'years_from_baseline', 'years');
% subtype_stage = renamevars(subtype_stage, 'Diagnosis', 'diagnosis');

data = {'ADNI_FSX_LS'};
method = {'FTR_MCEM'};

nsubtype = 5;

include_control = 0;
stagsel = 0;
results = load_data_result(data, method, nsubtype, include_control, stagsel);
joindata = results{1,1}.joindata;
traj = results{1,1}.traj;
biomarker_names = [results{1,1}.biomarker_names];

Abeta = 0;
Tau = 1;

if Abeta
    biomarker_names_pet = cellfun(@(x) ['Abeta-',x], biomarker_names, 'UniformOutput', 0);
elseif Tau
    biomarker_names_pet = cellfun(@(x) ['TAU-',x], biomarker_names, 'UniformOutput', 0);
end

nan_inds = any(isnan(joindata{:,biomarker_names_pet}), 2);
data = joindata;
data(nan_inds, :) = [];

fprintf('%d subjects, %d points\n', length(unique(data.RID)), size(data,1));

plot_trajectory_GP(data, biomarker_names_pet);

% if exist(save_dir, 'dir') ~= 7
%     % If it doesn't exist, create the folder
%     mkdir(save_dir);
% end

%% Identify rows where "diagnosis" is 0.5
%rowsWithDiagnosis0_5 = data.diagnosis == 0.5;

% Identify unique "RID" values where "diagnosis" is 1
%uniqueRIDs = unique(data.RID(data.diagnosis == 1));

% Identify rows with both "diagnosis" 0.5 and corresponding "RID" with "diagnosis" 1
%selectedRows = rowsWithDiagnosis0_5 & ismember(data.RID, uniqueRIDs);



%% box plots
% Create a figure with subplots for different stage ranges
figure();


stage_ranges = [0, 1/3, 2/3, 1];
cols = length(stage_ranges) - 1;

subtype_offsets = [-floor(nsubtype/2):floor(nsubtype/2)];
subtype_colors = [0.8,0.2,0;0,0.8,0.2;0.2,0,0.8; 0.5,0.5,0; 0,0.5,0.5];
if Abeta
%     biom_names = {'Abeta-summary','Abeta-Hippocampus'};
    biom_names = {'Abeta-summary','Abeta-Left-Hippocampus'};
elseif Tau
%     biom_names = {'TAU-Temporal','TAU-Parietal','TAU-Frontal','TAU-Hippocampus'};
    biom_names = {'TAU-Left-Temporal','TAU-Left-Parietal','TAU-Left-Frontal','TAU-Left-Hippocampus'};
end

% Add a title for the entire figure
% sgtitle('Box Plot for Trajectory');

hs = [];
y_all = {};

for j = 1:cols
    start_range = stage_ranges(j);
    end_range = stage_ranges(j + 1);
    
    for k = 1:nsubtype
        if end_range == 1
            ind = (data.stage >= start_range) & (data.stage <= end_range);
        else
            ind = (data.stage >= start_range) & (data.stage < end_range);
        end
        ind = ind & (data.subtype == k);
        
        ys = data{ind, biom_names};
        
        if ~isempty(ys)

        for idx_biom = 1:size(ys,2)
            subplot(size(ys,2), 1, idx_biom);
            hold on;
            
            y = ys(:,idx_biom);
            y_all{j,k,idx_biom} = y;
            
            stage_width = (nsubtype+1);
            x = (j-1)*stage_width + 1 + subtype_offsets(k);
            x = repmat(x, size(y,1), 1);
            swarmchart(x, y, 3, subtype_colors(k,:)/2, 'filled');
            hs(j,k,idx_biom) = boxchart(x, y, 'boxfacecolor', ...
                subtype_colors(k,:), 'whiskerlinecolor', subtype_colors(k,:), ...
                'markerstyle', 'none');   

%             title(sprintf('Stage %.1f - %.1f', start_range, end_range));
%             xlabel('stage - subtype');
            ylabel([biom_names{idx_biom}, ' zscore']);
            
            xticks((0:nsubtype-1)*stage_width + 1)
            xticklabels({'Early stage','Intermediate stage','Late stage'})
        end
        
        end
    end

end

subtype_names = cellfun(@(x) ['Subtype ',int2str(x)], num2cell((1:nsubtype)),'UniformOutput',false);
legend(hs(1,:,1), subtype_names);

%% t-test
disp('test stage difference abeta cortical/hippocampus (row: adjacent stage, col: subtype)')
Ps = [];
for j = 1:cols-1
    for k = 1:nsubtype
        for m = 1:size(y_all,3)
            y1 = y_all{j,k,m};
            y2 = y_all{j+1,k,m};
            if ~isempty(y1) && ~isempty(y2)
                [h,p] = ttest2(y1,y2,'Vartype','unequal');
            else
                p = NaN;
            end
            Ps(j,k,m) = p;
        end
    end
end

Ps

disp('test subtype difference abeta cortical/hippocampus (row: stage, col: 1-2 2-3 3-1)')
Ps = [];
for j = 1:cols
    for k = 1:nsubtype
        for m = 1:size(y_all,3)
            if k+1 > nsubtype
                k1 = mod(k+1,nsubtype);
            else
                k1 = k+1;
            end
            y1 = y_all{j,k,m};
            y2 = y_all{j,k1,m};
            if ~isempty(y1) && ~isempty(y2)
                [h,p] = ttest2(y1,y2,'Vartype','unequal');
            else
                p = NaN;
            end
            Ps(j,k,m) = p;
        end
    end
end

Ps

end

function plot_trajectory_GP(data, biomarker_names_pet)
%% Gaussian Process from matlab
% num_int = 100;
% nsubtype = 3;
% nbiom = length(biomarker_name);
% thres = ones(nbiom,1)*2;
% re_traj = zeros(nbiom,num_int+1,nsubtype);
% 
% for k = 1:3
%     %sigma0 = [0.2,0.2,0.2];
%     %kparams0 = [3.5, 6.2;3.5, 6.2;3.5, 6.2];
%     for j = 1:nbiom
%         %lis = data.subtype == k;
%         lis = data.subtype == k & data.diagnosis > 0;
%         %gprModel = fitrgp(data.stage(lis), data{lis,j+2},'KernelFunction','squaredexponential',...
%      %'KernelParameters',kparams0(k,:),'Sigma',sigma0(k));
%         gprModel = fitrgp(data.stage(lis), data{lis,j+2});
%         re_traj(j,:,k) = predict(gprModel, transpose((0:num_int)/num_int));
%         %re_traj(j,:,k) = interp1(data.stage(lis),data{lis,j+2},(0:num_int)/num_int);
%     end
% end

%% Multiple Gaussian Process
K = length(unique(data.subtype));

num_int = 100;
nsubtype = K;
nbiom = length(biomarker_names_pet);
thres = ones(nbiom,1)*1;
re_traj = zeros(nbiom,num_int+1,nsubtype);

meanfunc = [];
covfunc = @covSEiso; 
likfunc = @likGauss; 
hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

for k = 1:K
    lis = data.subtype == k;
    x = data.stage(lis);
    y = data{lis, biomarker_names_pet};
    xt = transpose((0:num_int)/num_int);
    hyp2 = minimize(hyp, @gp, -5, @infGaussLik, meanfunc, covfunc, likfunc,x, y);
    [yt ys] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc,x, y, xt);
    re_traj(:,:,k) = transpose(yt);
    %gprModel = fitrgp(data.stage(lis), data{lis,3:end});
    %re_traj(:,:,k) = predict(gprModel, transpose((0:num_int)/num_int));
    
    show_trajectory(re_traj(:,:,k),thres,biomarker_names_pet);
end

%ord = cal_order(re_traj,biomarker_name,save_dir);

% num_biom = size(re_traj,1);
% num_points = size(re_traj,2);
% num_subtypes = size(re_traj,3);
% 
% %Generating random stage values for each time point
% stage = (1:num_points)/num_points; % Assuming stage values range from 0 to 1
% 
% %Define stage ranges
% stage_ranges = [0, 0.3, 0.8, 1];
% 
% % Create a figure with subplots for different stage ranges
% figure('Name', 'Box Plot for PET-abeta Trajectory');
% % Add a title for the entire figure
% overall_title = 'Box Plot for PET-abeta Trajectory';
% sgtitle(overall_title); 
% 
% for j = 1:length(stage_ranges) - 1
%     start_range = stage_ranges(j);
%     end_range = stage_ranges(j + 1);
% 
%     % Find indices of time points within the current stage range
%     indices = find((stage >= start_range) & (stage < end_range));
% 
%     % Create an empty matrix to store data for each subtype
%     data_subset = zeros(length(indices), num_subtypes);
% 
%     % Extract data for the current stage range and each subtype
%     for i = 1:num_subtypes
%         subtype_data = mean(re_traj(:, :, i), 1); % Modify this according to your data
% 
%         % Store data for the current subtype
%         data_subset(:, i) = subtype_data(indices);
%     end
% 
%     % Plot boxplots for each subtype in the current stage range
%     subplot(1, length(stage_ranges) - 1, j);
%     violin(data_subset, 'Labels', {'Subtype 1', 'Subtype 2', 'Subtype 3'});
%     title(sprintf('Stage %.1f - %.1f', start_range, end_range));
%     xlabel('Subtypes');
%     ylabel('Values');
% end

end