function analyze_traj()

close all

nsubtype = 6;

biom_name_full = load('./input/disease_info/output/ukb_biom.mat').ans;
biom_name_selected = { ...
    'GDF15', 'CDCP1', 'CXCL17', 'EDA2R', 'NEFL', 'WFDC2', 'HAVCR1', ...
    'BCAN', 'TNFRSF10B', 'CXCL14', 'CA14', 'ADM', 'IL6', 'MEPE', ...
    'ODAM', 'REN', 'MMP12', 'ACTA2', 'EGFR', 'NTproBNP', 'TSPAN1', ...
    'LTBP2', 'ACE2', 'HSPB6', 'PLAT', 'TFF1', 'VSIG4', 'AMBP', ...
    'ENPP5', 'GFAP', 'CTSV', 'TFRC', 'CCL2', 'SKAP1' ...
};
indices = get_indices(biom_name_selected, biom_name_full);
traj_path = './output/ukb_MixtureGRBF_cv_nsubtype/';
traj = load_data(traj_path,nsubtype,indices);
plot_dim_num = 6;

figure('Position', [0 800 3000 1600]);
for i = 1:plot_dim_num
    for j = 1:plot_dim_num
        subplot(plot_dim_num, plot_dim_num, (i-1)*plot_dim_num + j);
        hold on
        for k = 1:nsubtype
            plot(linspace(40,71,32),traj(:,(i-1)*plot_dim_num + j,k), "LineWidth", 0.8)
        end
        xlabel('age')
        ylabel('zscore')
        
        % hold on
        % 
        % data_dim = data{data{:,1} == k,:};
        % 
        % scatter(data_dim(:,2)+randn(size(data_dim,1),1),data_dim(:,(i-1)*plot_dim_num + j +2),1)
        % hold off
        title(sprintf('%s', biom_name_selected{(i-1)*plot_dim_num + j}));
        hold off
        
        if (i-1)*plot_dim_num + j >= length(biom_name_selected)
            break
        end
        
    end
end
legend({'Subtype 1','Subtype 2','Subtype 3','Subtype 4','Subtype 5','Subtype 6',}, 'Location','best')
saveas(gcf, './output/1.png');
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

function traj_all = load_data(traj_path,nsubtype,indices)

traj_all = [];

for k = 1:nsubtype
    traj_path_k = [traj_path, 'cross_validation_nsubtype',int2str(nsubtype),'_fold0/trajectory',int2str(k),'.csv'];
    traj = readmatrix(traj_path_k);
    traj = traj(:,indices);
    traj_all(:,:,k) = traj;
end

end