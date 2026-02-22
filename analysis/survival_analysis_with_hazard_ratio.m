function survival_analysis_with_hazard_ratio()
    close all;
    
    exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17';
    nsubtype = 5;
    disease_dir = './input/disease_info/';
    subtype_path = ['./output/',exp_name,'/',int2str(nsubtype),'_subtypes/subtype_stage.csv'];
    output_dir = ['output/result_analysis/',exp_name,'/',int2str(nsubtype),'_subtypes'];
    subtype_ord_path = [output_dir, '/all_cause_mortality_order.csv'];
    
    if ~exist(output_dir, "dir")
        mkdir(output_dir)
    end
    
    subtype_stage_data = readtable(subtype_path);
    files = dir(disease_dir);
    
    csvFiles = files((endsWith({files.name}, '.csv')) & (~startsWith({files.name}, 'code')));
    num_files = length(csvFiles);
    
    disease_code_file = files(startsWith({files.name}, 'code'));
    disease_code_table = readtable(fullfile(disease_dir, disease_code_file.name));
    
    % 选择参考subtype
    if exist(subtype_ord_path, 'file')
        subtype_ord = readmatrix(subtype_ord_path);
        [~, loc] = ismember(subtype_stage_data.subtype, subtype_ord);
        subtype_stage_data.subtype = loc;
        ref_subtype = 1;
    else
        fprintf('Available subtypes: 1 to %d\n', nsubtype);
        ref_subtype = input('Enter the reference subtype number (1-4): ');
    end
    figure('Units', 'normalized', 'Position', [0.1,0.1,0.35,0.5]); % 调整大小更适合森林图
    
    hold on
    % 初始化颜色方案
    subtype_colors = lines(nsubtype);
    disease_labels = cell(num_files-1, 1);
    disease_interval = 2;
    subtype_offset = 0.25;
    all_y_positions = [];
    current_y = 0; % y 轴从 0 开始
    
    % Bonferroni 校正
    bonf_cor = (num_files - 1) * (nsubtype - 1);
    alpha = 0.05 / bonf_cor; % 计算 Bonferroni 校正后的显著性阈值
    sig_levels = [0.05, 0.01, 0.001] / bonf_cor; % 计算不同显著性水平
    
    CI_high_pos = [];
    CI_low_pos = [];
    
    % 主分析循环
    for i = 2:num_files
        disease_info = readtable(fullfile(disease_dir, csvFiles(i).name));
        disease_info = join(subtype_stage_data, disease_info(:, {'eid', 'target_y', 'BL2Target_yrs'}), 'LeftKeys', 'PTID', 'RightKeys', 'eid');
        disease_info.age = disease_info.stage + disease_info.BL2Target_yrs;
        disease_info.censored = ~disease_info.target_y;
        event = disease_info.censored;
        time = disease_info.age;
        subtypes = disease_info.subtype;
        dummy_subtypes = dummyvar(subtypes);
        if ref_subtype ~= 1
            dummy_subtypes = [dummy_subtypes(:,ref_subtype) dummy_subtypes(:,1:ref_subtype-1) dummy_subtypes(:,ref_subtype+1:end)];
        end
        dummy_subtypes(:,1) = [];
        [b,logl,H,stats] = coxphfit(dummy_subtypes, time, 'Censoring', event);
        hr = exp(b);
        ci = exp(stats.se.*[-1.96 1.96] + b);
        p_values = stats.p;
    
        % 创建结果表格
        result_table = table();
        subtype_idx = 1;
        for j = 1:nsubtype
            if j ~= ref_subtype
                result_table.Subtype(subtype_idx) = j;
                result_table.HR(subtype_idx) = hr(subtype_idx);
                result_table.CI_low(subtype_idx) = ci(subtype_idx,1);
                result_table.CI_high(subtype_idx) = ci(subtype_idx,2);
                result_table.P_value(subtype_idx) = p_values(subtype_idx);
                subtype_idx = subtype_idx + 1;
            end
        end
    
        % 记录疾病名称
        disease_code = extractBefore(csvFiles(i).name, 3);
        disease_name = disease_code_table.Disease(strcmp(disease_code_table.Code, disease_code));
        disease_labels{i-1} = disease_name;
    
        % 计算 y 轴位置
        current_y = current_y + disease_interval;
        base_y = current_y;
    
        % 记录 y 轴位置
        all_y_positions = [all_y_positions; repmat(base_y, height(result_table), 1)];
    
        % 画出每个亚型的 HR 和 CI
        % 绘制参考组：HR = 1 的点
        ref_y_pos = base_y; % 参考组绘制在疾病的基准位置
        h_legends = [];
        % h = scatter(1, ref_y_pos, 50, 'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        % h_legends = [h_legends, h];
    
        for k = 1:nsubtype
            y_pos = base_y + (k - (nsubtype+1)/2) * subtype_offset; % 让不同亚型在 y 轴上稍微分开
    
            if k == ref_subtype
                h = scatter(1, y_pos, 50, 'MarkerFaceColor', subtype_colors(k,:), 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
                h_legends = [h_legends, h];
                continue
            elseif k > ref_subtype
                subtype_idx = k-1;
            else
                subtype_idx = k;
            end
    
            % 置信区间（水平线）
            line([result_table.CI_low(subtype_idx), result_table.CI_high(subtype_idx)], ...
                 [y_pos, y_pos], 'Color', subtype_colors(k,:), 'LineWidth', 2);
            
            if result_table.CI_high(subtype_idx) ~= Inf
                CI_high_pos = [CI_high_pos, result_table.CI_high(subtype_idx)];
            end
            if result_table.CI_low(subtype_idx) ~= 0
                CI_low_pos = [CI_low_pos, result_table.CI_low(subtype_idx)];
            end
    
            % HR 点
            h = scatter(result_table.HR(subtype_idx), y_pos, 50, ...
                    'MarkerFaceColor', subtype_colors(k,:), ...
                    'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
            h_legends = [h_legends, h];
    
            if result_table.P_value(subtype_idx) < sig_levels(3)
                sig_text = '***';  % p < 0.001
            elseif result_table.P_value(subtype_idx) < sig_levels(2)
                sig_text = '**';   % p < 0.01
            elseif result_table.P_value(subtype_idx) < sig_levels(1)
                sig_text = '*';    % p < 0.05
            else
                sig_text = '';     % 不显著，不显示
            end
    
            % 在 HR 点旁边显示显著性标记
            if ~isempty(sig_text)
                % text(result_table.HR(subtype_idx) + 0.1, y_pos, sig_text, ...
                %     'FontSize', 12, 'FontWeight', 'bold', 'Color', [0 0 0]);
                text(result_table.CI_high(subtype_idx) + 0.1, y_pos+0.05, sig_text, ...
                    'FontSize', 12, 'FontWeight', 'bold', 'Color', [0 0 0]);
            end
        end  
            
    end
    
    % 主分析循环结束后，在设置图形属性之前添加以下代码
    hold on;
    
    % 获取 x 轴和 y 轴范围
    x_limits = [min(CI_low_pos)-0.1, max(CI_high_pos)+0.4];
    y_min = 0.8; % ylim 的下限
    y_max = base_y + 1; % ylim 的上限
    
    % 计算每个疾病的矩形高度
    disease_y_positions = all_y_positions(1:nsubtype-1:end); % 每个疾病的基准 y 位置
    n_diseases = length(disease_y_positions);
    rect_height = disease_interval; % 每个疾病区域的高度，与 disease_interval 一致
    
    % 为奇偶疾病组绘制背景矩形
    for i = 1:n_diseases
        y_bottom = disease_y_positions(i) - rect_height / 2; % 矩形底部
        y_top = disease_y_positions(i) + rect_height / 2;    % 矩形顶部
        
        % 使用奇偶判断来设置颜色
        if mod(i, 2) == 1 % 奇数疾病组
            fill_color = [0.9 0.9 0.9]; % 浅灰色
        else % 偶数疾病组
            fill_color = [1 1 1]; % 白色
        end
        
        % 绘制矩形
        fill([x_limits(1) x_limits(2) x_limits(2) x_limits(1)], ...
             [y_bottom y_bottom y_top y_top], ...
             fill_color, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    end
    
    % 确保背景矩形在最底层
    uistack(findobj(gca, 'Type', 'patch'), 'bottom');
    
    % 统一设置图形属性
    set(gca, 'XScale', 'log'); % x 轴使用对数刻度
    set(gca, 'TickLabelInterpreter', 'latex', 'FontName', 'Times New Roman');
    
    % 设置 x 轴
    xlim([min(CI_low_pos)-0.1, max(CI_high_pos)+0.4]); % x 轴范围稍微扩大
    xlabel('Hazard Ratio');
    
    % 设置 y 轴
    set(gca, 'YDir', 'reverse', 'YGrid', 'on', 'XGrid', 'on'); % 让疾病名称从上到下排列
    xticks([0.5 1 2 4])
    yticks(all_y_positions(1:nsubtype-1:end)); % 只标记主要疾病位置
    disease_labels = cellfun(@(x) x{1}, disease_labels, 'UniformOutput', false);
    ylim([0 base_y+disease_interval])
    wrappedCell = cellfun(@(str) strjoin(textwrap({str}, 25), '\\'), disease_labels, 'UniformOutput', false);
    wrappedCell = cellfun(@(str) ['\begin{tabular}{@{}r@{}}' strrep(str, ' \', '\\ ') '\end{tabular}'], wrappedCell, 'UniformOutput', false);
    yticklabels(wrappedCell);
    % yticklabels(disease_labels);
    ylabel('Disease');
    title('Forest Plot of Hazard Ratios Across Subtypes', ...
            'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    % 添加参考线
    line([1, 1], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
    
    % 添加图例
    legend_items = arrayfun(@(x) subtype_legend(x, ref_subtype), 1:nsubtype, 'UniformOutput', false);
    legend(h_legends, legend_items, 'Location', 'eastoutside', 'Box', 'off');
    
    % 调整字体
    set(gca, 'FontSize', 10, 'TickLength', [0 0]);
    box off;
    
    output_path = [output_dir, '/hazard_ratio_significance.jpg'];
    saveas(gcf, output_path);
    % export_fig(output_path, '-r500', '-transparent');
    
end
    
function r = subtype_legend(x, ref)
    if x == ref
        r = sprintf('Subtype %d (Reference)', x);
    else
        r = sprintf('Subtype %d', x);
    end
end