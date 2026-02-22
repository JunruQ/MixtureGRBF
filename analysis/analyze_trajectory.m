function [outputArg1,outputArg2] = analyze_trajectory(inputArg1,inputArg2)
%ANALYZE_TRAJECTORY Summary of this function goes here
%   Detailed explanation goes here

close all

data_names = {'ukb'};
method = {'MixtureGRBF'};

nsubtype = 6;

results = load_data_result(data_names, method, nsubtype);

biomarker_names = { ...
    'GDF15', 'CDCP1', 'CXCL17', 'EDA2R', 'NEFL', 'WFDC2', 'HAVCR1', ...
    'BCAN', 'TNFRSF10B', 'CXCL14', 'CA14', 'ADM', 'IL6', 'MEPE', ...
    'ODAM', 'REN', 'MMP12', 'ACTA2', 'EGFR', 'NTproBNP', 'TSPAN1', ...
    'LTBP2', 'ACE2', 'HSPB6', 'PLAT', 'TFF1', 'VSIG4', 'AMBP', ...
    'ENPP5', 'GFAP', 'CTSV', 'TFRC', 'CCL2', 'SKAP1' ...
};

% biomarker_names = [results{1,1}.biomarker_names];
traj = results{1,1}.traj;

% joindata.Properties.VariableNames = strrep(joindata.Properties.VariableNames, '-', '_');


% show_trajectory_in_PCA(traj, data, joindata.subtype);


% thres = 1;
stage = 1:size(traj,2);
show_trajectory(traj,stage,biomarker_names);

end

