function sel_traj(re_traj,num_pic,biomarker_name,save_dir)
%% Select traj
num_int = size(re_traj,2);
nsubtype = size(re_traj,3);

biomarker_name_left = {};
biomarker_name_right = {};
for k = 1:length(biomarker_name)
    biomarker_name_left{end+1} = strcat('Left-',biomarker_name{k});
    biomarker_name_right{end+1} = strcat('Right-',biomarker_name{k});
end

biomarker_remain = readtable('./input/Tadpole/biomarker_name_remain.csv','VariableNamingRule','preserve');
remain_name = biomarker_remain.Properties.VariableNames;
remain_name_right = {};
remain_name_left = {};
for k = 1:length(remain_name)
    remain_name_left{end+1} = strcat('Left-',remain_name{k});
    remain_name_right{end+1} = strcat('Right-',remain_name{k});    
end

for i = 1:nsubtype
    row_cell = cell(num_pic,1);
    for j = 1:num_pic
        row_cell(j,1) = cellstr(['Image',int2str(j)]);
    end
    row_table = cell2table(row_cell,"VariableNames","Image-name-unique");
    traj_sel_left = array2table(re_traj(:,1:(num_int-1)/(num_pic-1):num_int,i)', 'VariableNames', biomarker_name_left);
    traj_sel_right = array2table(re_traj(:,1:(num_int-1)/(num_pic-1):num_int,i)', 'VariableNames', biomarker_name_right);
    traj_remain_left = array2table(zeros(num_pic,length(remain_name_left)), 'VariableNames', remain_name_left);
    traj_remain_right = array2table(zeros(num_pic,length(remain_name_right)), 'VariableNames', remain_name_right);
    traj_output = [row_table,traj_sel_left,traj_sel_right,traj_remain_left,traj_remain_right];
    writetable(traj_output, [save_dir,'/sel_trajectory',int2str(i),'.csv'])
end
end