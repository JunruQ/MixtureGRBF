function plot_traj(traj, stage, label)
%This function is used to plot the trajectory of T * D data, where D
%represents the number od dimensions and T represents the number of stage
    % traj: T * D
    % stage
    % label: D * 1
    if nargin < 3
        label = [];
    end
    if nargin < 2
        stage = 39:70;
    end
    if nargin < 1
        traj = readmatrix('./output/ukb_MixtureGRBF_cv_nsubtype/cross_validation_nsubtype6_fold0/trajectory1.csv');

        data_path = ['input/ukb/ukb_table_demo1_regress1_log0_norm0_z1_nanfilled1.csv'];
        biomarker_columns = [8 0];

        data = readtable(data_path,'VariableNamingRule','preserve');
        biom_name_full = data.Properties.VariableNames(biomarker_columns(1):end - biomarker_columns(2));
        biom_name_selected = { ...
            'GDF15', 'CDCP1', 'CXCL17', 'EDA2R', 'NEFL', 'WFDC2', 'HAVCR1', ...
            'BCAN', 'TNFRSF10B', 'CXCL14', 'CA14', 'ADM', 'IL6', 'MEPE', ...
            'ODAM', 'REN', 'MMP12', 'ACTA2', 'EGFR', 'NTproBNP', 'TSPAN1', ...
            'LTBP2', 'ACE2', 'HSPB6', 'PLAT', 'TFF1', 'VSIG4', 'AMBP', ...
            'ENPP5', 'GFAP', 'CTSV', 'TFRC', 'CCL2', 'SKAP1' ...
        };
        indices = get_indices(biom_name_selected, biom_name_full);
        traj = traj(:, indices);
    end
    [T, D] = size(traj);

    cmap = parula(D);

    % [~, order] = sort(traj(1,:));
    % position_order = zeros(size(order));
    % position_order(order) = 1:length(order);

    figure;
    hold on;
    % for i = flip(order)
    %     plot(stage, traj(:,i),'Color',cmap(position_order(i),:),'LineWidth',1.5);
    % end
    for i = 1:D
        plot(stage, traj(:,i),'Color',cmap(i,:),'LineWidth',1.5);
    end

    title('traj')
    xlim([stage(1),stage(end)])
    xlabel('stage')
    ylabel('mu')
    legend(biom_name_selected) 
end


function indices = get_indices(cellArray1, cellArray2)

% 初始化一个空数组来存储索引
indices = [];

% 遍历 cellArray1 中的每个字符串
for i = 1:numel(cellArray1)
    % 使用 strcmp 函数查找 cellArray1 中的字符串在 cellArray2 中的索引
    index = find(strcmp(cellArray2, cellArray1{i}));
    
    % 如果找到了索引，将其添加到 indices 数组中
    if ~isempty(index)
        indices = [indices, index];
    end
end
end