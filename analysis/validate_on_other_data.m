function [outputArg1,outputArg2] = validate_on_other_data(inputArg1,inputArg2)
%VALIDATE_ON_EXTERNAL_DATA Summary of this function goes here
%   Detailed explanation goes here

input_file_name = 'SCZ/R5_neuroimaging_regional_vol_17ROI_for_ZY_2023.12.23.xlsx';

sheets = {'volume_T1_zscore_17ROI', 'volume_T2_zscore_17ROI', 'volume_T3_zscore_17ROI'};
start_col = 5;

nsubtypes = [2,3];

for nsubtype = nsubtypes
    discovery_traj_dir = 'SCZ17_FTR_kmeans_stagsel=1';
    % outputdir = 'SCZ122_FTR_kmeans_stagsel=1';
    
    re_traj = load_trajectories(discovery_traj_dir, nsubtype);
    
    for j = 1:length(sheets)
        % Read validation data
        inputfilename = ['./input/',input_file_name];
        opts = detectImportOptions(inputfilename);
        opts = setvartype(opts, opts.VariableNames(5:end), 'double');
        input_data = readtable(inputfilename, opts, 'Sheet',sheets{j});
        
        % calculate the subtypes and stages on the validation set
        dat = input_data{:, start_col:end};
        PTID_str = input_data{:, 1};
        PTID = convert_PTID_str2num(PTID_str);
        
        [stage,subtype,dist_samp_min] = cal_stage_subtype(dat,PTID,re_traj);
        input_data{:, start_col:start_col+1} = [subtype, stage];
        input_data.Properties.VariableNames(start_col:start_col+1) = {'subtype', 'stage'};
        input_data(:, start_col+2:end) = [];
        
        [filepath,name,ext] = fileparts(input_file_name);
        output_dir = ['./input/',filepath,'/',name,'_','nsubtype=',num2str(nsubtype),'_subtype_stage',ext];
        writetable(input_data,output_dir,'Sheet',sheets{j});
    end
end

end

function PTID = convert_PTID_str2num(PTID_str)
PTID = zeros(size(PTID_str));
for i = 1:length(PTID_str)
    PTID(i) = str2num(PTID_str{i}(2:end));
end

end

function traj = load_trajectories(outputdir, nsubtype)
save_dir = ['output/', outputdir, '/',int2str(nsubtype), '_subtypes'];
S = load(['output/', outputdir, '/all_data.mat']);
bio_name = S.biomarker_name;

for k = 1:nsubtype
    traj(:,:,k) = readmatrix([save_dir,'/trajectory',int2str(k)])';
end

end