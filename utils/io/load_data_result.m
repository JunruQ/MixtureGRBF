function results = load_data_result(datasets, methods, nsubtype, include_control, stagsel)
%LOAD_DATA_AND_RESULT Summary of this function goes here
%   Detailed explanation goes here
if nargin < 4
    include_control = 0;
end

if nargin < 5
    stagsel = 0;
end

results = {};

for idx_data = 1:length(datasets)
    for idx_method = 1:length(methods)
        switch datasets{idx_data}
            case {'ukb'}
                data_path = ['input/ukb/ukb_table_1.csv'];
                biomarker_columns = [8 0];
            otherwise
        end
        data = readtable(data_path,'VariableNamingRule','preserve');
        
        switch methods{idx_method}
            case {'MixtureGRBF'}
                result_path = ['output/ukb_MixtureGRBF_'];
                [subtype_stage, traj] = load_ftr_result(result_path, nsubtype);
            otherwise
        end
        
        joindata = outerjoin(subtype_stage,data,'Keys',{'RID','years'},'MergeKeys',true);

        % if include_control
        %     joindata = joindata(joindata.group == 1 | joindata.group == 0, :);
        % else
        %     % select only the case subjects
        %     joindata = joindata(joindata.group == 1, :);
        % end
        
        result = [];
        result.joindata = joindata;
        result.traj = traj;
        result.biomarker_names = data.Properties.VariableNames(biomarker_columns(1):end - biomarker_columns(2));
        
        results{idx_data, idx_method} = result;
    end
end

end

function [subtype_stage, traj, biomarker_names] = load_ftr_result(result_ftr_path, nsubtype)
result_subtype_stage_path = [result_ftr_path, '/',num2str(nsubtype),'_subtypes/subtype_stage.csv'];

for k = 1:nsubtype
    traj(:,:,k) = readmatrix([result_ftr_path,'/',num2str(nsubtype),'_subtypes/trajectory',int2str(k)])';
end

subtype_stage = readtable(result_subtype_stage_path,'VariableNamingRule','preserve');
subtype_stage = renamevars(subtype_stage, 'PTID', 'RID');
subtype_stage = renamevars(subtype_stage, 'years_from_baseline', 'years');
% subtype_stage.Diagnosis = [];

end