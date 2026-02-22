function survival_analysis()
    close all;
    
    exp_name = 'ukb_MixtureGRBF_test_test';
    nsubtype = 4;
    disease_dir = './input/disease_info/';
    subtype_path = ['./output/',exp_name,'/',int2str(nsubtype),'_subtypes/subtype_stage.csv'];
    output_dir = ['output/result_analysis/',exp_name,'/',int2str(nsubtype),'_subtypes'];
    subtype_ord_path = [output_dir, '/all_cause_mortality_order.csv'];
    
    if ~exist(output_dir, "dir")
        mkdir(output_dir)
    end
    
    if exist(subtype_ord_path, "file")
        subtype_ord = readmatrix(subtype_ord_path);
    else
        subtype_ord = 1:nsubtype;
    end
    
    subtype_stage_data = readtable(subtype_path);
    [~, loc] = ismember(subtype_stage_data.subtype, subtype_ord);
    subtype_stage_data.subtype = loc;
    files = dir(disease_dir);
    
    csvFiles = files((endsWith({files.name}, '.csv')) & (~startsWith({files.name}, 'code')));
    num_files = length(csvFiles);
    
    % 创建保存 p 值的矩阵
    bonferroni_coef = nsubtype * (nsubtype - 1) / 2;
    
    disease_code_file = files(startsWith({files.name}, 'code'));
    disease_code_table = readtable(fullfile(disease_dir, disease_code_file.name));
    
    figure('Position',[0 800 1500 800]);
    
    p_matrix_all = NaN(nsubtype, nsubtype, num_files);
    
    % 在 3x5 子图中进行循环
    for i = 2:num_files
        % p_matrix = ones(nsubtype, nsubtype);
    
        disease_info = readtable(fullfile(disease_dir, csvFiles(i).name));
        disease_info = join(subtype_stage_data, disease_info(:, {'eid', 'target_y', 'BL2Target_yrs'}), 'LeftKeys', 'PTID', 'RightKeys', 'eid');
        disease_info.age = disease_info.stage + disease_info.BL2Target_yrs;
        disease_info.censored = ~disease_info.target_y;
    
        % 提取数据
        ages = disease_info.age;
        censored = disease_info.censored;
        subtypes = disease_info.subtype;
    
        % 获取所有唯一的类别
        unique_subtypes = unique(subtypes);
    
        % 创建颜色图以区分不同类别
        colors = lines(length(unique_subtypes));
    
        % 子图索引
        subplot_row = mod(i-2, 3) + 1; % 从1开始的行
        subplot_col = floor((i-2)/3) + 1; % 从1开始的列
    
        subplot(3, 5, (subplot_row-1)*5 + subplot_col);
        hold on;
    
        legend_entries = cell(length(unique_subtypes), 1); % 保存图例条目
    
        % 对每个类别进行生存分析并绘制生存曲线
        h_legends = [];
        all_f = cell(length(unique_subtypes), 1);
        all_x = cell(length(unique_subtypes), 1);
        for j = 1:length(unique_subtypes)
            subtype = unique_subtypes(j);
        
            % 提取该类别的数据
            idx = (subtypes == subtype);
            age_subtype = ages(idx);
            censored_subtype = censored(idx);
        
            % 使用Kaplan-Meier估计，并获取置信区间
            [f, x, flo, fup] = ecdf(age_subtype, 'censoring', censored_subtype, 'function', 'survivor');
        
            % 绘制置信区间（阴影区域）
            %  'fill' 函数需要 x 坐标和 y 坐标来定义一个多边形。
            %  我们将 x 坐标反向连接 ([x; flipud(x)])，
            %  并将下限和上限连接 ([flo; flipud(fup)]) 来创建这个多边形。
            flo(1) = 1;
            fup(1) = 1;
            all_f{j} = f;
            all_x{j} = x;
            
            if ~isempty(x)
                fill([x; flipud(x)], [flo; flipud(fup)], colors(j,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            
                % 绘制生存曲线
                h = stairs(x, f, 'Color', colors(j,:), 'LineWidth', 1.5);
                h_legends = [h_legends, h];
            
                % 保存图例条目
                legend_entries{j} = ['Subtype ', num2str(subtype)];
            end
        end
    
        
    
        % 设置图形属性
        xlabel('Age');
        ylabel('Survival Probability');
        if ~isempty(all_f{1})
            x_min = adaptive_x_min(all_f, all_x);
        else
            x_min = 0;
        end
        xlim([x_min 85]);
        disease_name = disease_code_table.Disease( strcmp( disease_code_table.Code, extractBefore(csvFiles(i).name, 3) ) );
        title(['Disease: ', disease_name]);
        hold off;
    
        % % 生存曲线显著性分析 (log-rank test)
        % for j = 1:length(unique_subtypes)-1
        %     for k = j+1:length(unique_subtypes)
        %         idx1 = (subtypes == unique_subtypes(j));
        %         idx2 = (subtypes == unique_subtypes(k));
        % 
        %         p = logrank([ages(idx1), censored(idx1)], [ages(idx2), censored(idx2)]);
        %         % p = p * 15;
        %         p_matrix(j, k) = p;
        %         p_matrix(k, j) = p; % 对称矩阵
        %     end
        % end
        % 
        % p_matrix_all(:,:,i) = p_matrix;
        % 
        % % 输出 p 值矩阵
        % disp(['P-value matrix of ',char(disease_name),':']);
        % disp(p_matrix);
    end
    % 添加图例
    legend(h_legends, legend_entries, 'Location', 'best');
    
    % 保存整体图像
    saveas(gcf, [output_dir,'/survival_curves_parent.png']);

end 
    
function x_min = adaptive_x_min(all_f, all_x)
    
    % 在绘制曲线之前添加以下代码来计算自适应xlim的最小值
    threshold = 0.002; % 设置差异阈值（可调整，例如0.1表示10%的生存概率差异）
    
    % 创建统一的x轴用于插值
    x_min_all = min(cell2mat(cellfun(@min, all_x, 'UniformOutput', false)));  % 所有x的最小值
    x_max_all = max(cell2mat(cellfun(@max, all_x, 'UniformOutput', false)));  % 所有x的最大值
    x_common = linspace(x_min_all, x_max_all, 100);  % 创建100个均匀分布的点
    
    unique_subtypes = 1:length(all_f);
    % 对每个subtype的生存函数进行插值
    interp_f = zeros(length(x_common), length(unique_subtypes));
    for j = 1:length(unique_subtypes)
        % 使用interp1进行插值
        % 对于x_common中小于最小x或大于最大x的点，使用边界值
        % 查找重复的 x 值
        [unique_x, ~, idx] = unique(all_x{j}); % 使用 unique 获取唯一的 x 值和索引
        
        % 计算每个唯一 x 值对应的 y 值的平均值
        average_y = accumarray(idx, all_f{j}, [], @mean);
        % 你也可以使用其他函数，例如 @median, @max, @min 等
        
        % 使用合并后的数据进行插值
        interp_f(:, j) = interp1(unique_x, average_y, x_common, 'pchip', 1);
    end
    
    % 找到自适应的x_min
    x_min = x_min_all;
    found_threshold = false;
    
    for i = 1:length(x_common)
        x_val = x_common(i);
        % 获取当前x值下所有生存概率
        survival_probs = interp_f(i, :);
        
        % 计算生存概率的最大差异
        max_diff = max(survival_probs) - min(survival_probs);
        
        if max_diff >= threshold && ~found_threshold
            x_min = x_val;
            found_threshold = true;
            break;  % 找到后即可退出循环
        end
    end
    
end