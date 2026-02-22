function [outputArg1,outputArg2] = compare_subtype_progression(input,subtype_stage,subtype_name)

data = {'ADNI_FSX_HM'};
method = {'FTR'};

nsubtype = 3;

results = load_data_result(data, method, nsubtype);
joindata = results{1,1}.joindata;
joindata.Properties.VariableNames = strrep(joindata.Properties.VariableNames, '-', '_');
traj = results{1,1}.traj;
biomarker_names = [results{1,1}.biomarker_names];

% input = readtable("./input/ADNI/ADNI_demo_MRI_PET_ro_11_19.csv","VariableNamingRule","preserve");
% traj1 = readmatrix("./output/FTR_fsl/3_subtypes/trajectory1.csv");
% traj2 = readmatrix("./output/FTR_fsl/3_subtypes/trajectory2.csv");
% traj3 = readmatrix("./output/FTR_fsl/3_subtypes/trajectory3.csv");
% traj = cat(3, traj1, traj2, traj3);
% subtype_stage = readtable("./output/FTR_fsx/3_subtypes/subtype_stage.csv");
% 
% joindata = outerjoin(input,subtype_stage,'MergeKeys',true);
subtype_name = ["Subtype 1","Subtype 2","Subtype 3"];

% %% Fit linear regression
% % Find case group RID
% RID_diagnosis = unique(joindata.RID(joindata.diagnosis == 1));
% 
% % Select all rows with these RID
% joindata_case = joindata(ismember(joindata.RID, RID_diagnosis), :);
% 
% subtype_name = ["Hippocampal Sparing", "Typical", "Limbic Predominant"];
% variables = {'ADAS13', 'MMSE', 'ABETA', 'TAU'};
% k = length(subtype_name);
% colors = lines(k);
% markers = {'o', 's', '^'}; % Define markers for each subtype
% markerSize = 5;
% alphaValue = 0.3; % Define the opacity value
% 
% figure;
% 
% for varIdx = 1:length(variables)
%     subplot(2, 2, varIdx); % Create a subplot for each variable
%     hold on;
%     scatter_handles = [];
% 
% 
%     for i = 1:k
%         % Filter data based on subtype names
%         subtype_data = joindata_case(joindata_case.subtype == i, :);
%         y = subtype_data.years;
% 
%         scatter_handle = scatter(y, subtype_data.(variables{varIdx}), markerSize,'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', 'none');
%         scatter_handle.MarkerFaceAlpha = alphaValue; % Set opacity for the marker
%         scatter_handles = [scatter_handles, scatter_handle]; % Store scatter plot handle
% 
%         % Fit linear regression
%         mdl = fitlm(y, subtype_data.(variables{varIdx}));
% 
%         % Plot the linear regression line
%         x_range = min(y):0.1:max(y);
%         y_fit = predict(mdl, x_range');
%         plot(x_range, y_fit, 'Color', colors(i, :), 'LineWidth', 1.5);
%     end
% 
%     hold off;
% 
%     xlabel('Years');
%     ylabel(variables{varIdx});
%     title(sprintf('%s vs Years with Different Subtypes', variables{varIdx}));
%     legend(scatter_handles, subtype_name, 'Location', 'best');
% end

%% Fit linear mixed model
% Define the variables
% variables = {'ADAS13', 'MMSE'};
variables = {'MMSE','Abeta_summary','TAU_Temporal'};

reference_group = 3;

for varIdx = 1:length(variables)
    data = joindata;
    dependentVariable = variables{varIdx}; % Dependent variable (MMSE measurements)
    independentVariables = {'RID','years', 'subtype', 'AGE', 'PTGENDER', 'PTEDUCAT', 'diagnosis', 'stage'}; % Independent variables

    data.subtype = categorical(data.subtype, circshift((1:nsubtype), -reference_group+1));
    data.diagnosis = string(data.diagnosis);

    % Combine the predictors
    data.years = data.years - mean(data.years);
    data.AGE = data.AGE - mean(data.AGE);
    data.PTGENDER = data.PTGENDER - mean(data.PTGENDER);
    data.PTEDUCAT = data.PTEDUCAT - mean(data.PTEDUCAT);

    tbl = table(data.(dependentVariable), data.RID, data.years, data.subtype, data.AGE, data.PTGENDER, data.PTEDUCAT, data.diagnosis, data.stage);
    % Rename the variable names in the table
    tbl.Properties.VariableNames{1} = dependentVariable; % Rename the last variable as the dependent variable

    for i = 1:numel(independentVariables)
        tbl.Properties.VariableNames{i+1} = independentVariables{i}; % Rename other variables as independent variables
    end
    tbl2 = rmmissing(tbl);

    % Specify the formula for the linear mixed-effects model
    formula = strcat(dependentVariable,' ~ years*subtype + AGE + PTGENDER + PTEDUCAT + diagnosis + stage + (1 + years | RID)');

    % Fit the linear mixed-effects model
    mdl = fitlme(tbl2, formula,'DummyVarCoding','reference');

end
%% MMSE vs stage
% figure;
% hold on;
% scatter_handles = [];
% 
% for i = 1:k
%     % Filter data based on subtype names
%     subtype_data = joindata_case(joindata_case.subtype == i, :);
% 
%     scatter_handle = scatter(subtype_data.stage, subtype_data.MMSE, markerSize,'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', 'none');
%     scatter_handle.MarkerFaceAlpha = alphaValue; % Set opacity for the marker
%     scatter_handles = [scatter_handles, scatter_handle]; % Store scatter plot handle
% 
%     % Fit linear regression
%     mdl = fitlm(subtype_data.stage, subtype_data.MMSE);
% 
%     % Plot the linear regression line
%     x_range = min(subtype_data.stage):0.1:max(subtype_data.stage);
%     y_fit = predict(mdl, x_range');
%     plot(x_range, y_fit, 'Color', colors(i, :), 'LineWidth', 1.5);
% 
%     % Remove rows with NaN in MMSE and stage columns
%     valid_indices = ~isnan(subtype_data.MMSE) & ~isnan(subtype_data.stage);
%     cleaned_data = subtype_data(valid_indices, :);
% 
%     % Calculate correlation coefficient (r-value) and p-value
%     [r, p] = corr(cleaned_data.stage, cleaned_data.MMSE);
% 
%     % Add subtype names, r-value, and p-value to the legend
%     legend_info{i} = sprintf('%s (r = %.3f, p = %.3e)', subtype_name(i), r, p);
% end
% 
% hold off;
% 
% xlabel('Stage');
% ylabel('MMSE');
% title('MMSE vs Stages with Different Subtypes');
% legend(scatter_handles, subtype_name, 'Location', 'best');
% legend(scatter_handles, legend_info,'Location', 'best')
% 
% %end
end