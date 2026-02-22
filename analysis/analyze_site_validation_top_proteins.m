function analyze_site_validation_top_proteins()

nsubtype = 5;
top_n = 100;

all_t_stats = readtable('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/t_stats_protein_linear_reg.csv', 'VariableNamingRule', 'preserve');
all_top_prot = select_top_n_biomarkers(all_t_stats, nsubtype, top_n);

sites = {'Northern England', 'Southern England', 'Midlands', 'Scotland', 'Wales'};

set_type = {'train', 'val'};
figure('Position', [100, 100, 1000, 400], 'Visible', 'off');
t = tiledlayout(2, 5, 'TileSpacing', 'loose', 'Padding', 'loose');

for site_idx = 1:numel(sites)
    for set_idx = 1:numel(set_type)
        set_name = set_type{set_idx};
        site = sites{site_idx};
        
        site_t_stats = readtable(['output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/site_val/t_stats_',site,'_',set_name,'.csv'], 'VariableNamingRule', 'preserve');
        site_top_prot = select_top_n_biomarkers(site_t_stats, nsubtype, top_n);

        iou_matrix = zeros(nsubtype);

        % Fill confusion matrix
        for i = 1:nsubtype
            for j = 1:nsubtype
                all_subtype_proteins = all_top_prot.Biomarker(all_top_prot.Subtype == i);
                site_subtype_proteins = site_top_prot.Biomarker(site_top_prot.Subtype == j);

                % 计算交集和并集
                intersection = intersect(all_subtype_proteins, site_subtype_proteins);
                union_set = union(all_subtype_proteins, site_subtype_proteins);
                
                % 计算IoU
                intersection_count = length(intersection);
                union_count = length(union_set);
                iou = intersection_count / union_count;

                iou_matrix(i, j) = iou;
            end
        end

        % mIoU
        mIoU = mean(diag(iou_matrix));
        fprintf('mIoU for %s %s: %.3f\n', site, set_name, mIoU);
        
        nexttile(t, site_idx + (set_idx-1)*5);
        colormap summer;
        imagesc(iou_matrix);
        % colorbar;
        clim([0 1]);
        if set_idx == 1
            title(site, 'FontWeight', 'normal');
        end
        xticks(1:nsubtype);
        yticks(1:nsubtype);
        if site_idx == 1
            if strcmp(set_name, 'train')
                ylabel('Training subtype');
            else
                ylabel('Validation subtype');
            end
        end
        xlabel('All data subtype');
        
        % Add text labels
        for i = 1:nsubtype
            for j = 1:nsubtype
                text(j, i, num2str(iou_matrix(i, j), '%.2f'), ...
                    'HorizontalAlignment', 'center')
            end
        end

        set(gca, 'FontName', 'Arial');
    end

end

export_fig('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/site_validation_iou_matrices.png', '-r500', '-transparent');

end

function top_biomarkers = select_top_n_biomarkers(t_stats, nsubtype, top_n)
    % 初始化输出
    top_biomarkers = table();
    
    % 遍历每个 Subtype
    for subtype = 1:nsubtype
        % 筛选当前 Subtype 的数据
        subtype_data = t_stats(t_stats.Subtype == subtype, :);
        
        % 计算 t 的绝对值
        subtype_data.abs_t = abs(subtype_data.t);
        
        % 按 abs_t 降序排序
        [~, sort_idx] = sort(subtype_data.abs_t, 'descend');
        
        % 选择 top_n 行（如果数据不足 top_n，则取所有行）
        n = min(top_n, height(subtype_data));
        top_rows = subtype_data(sort_idx(1:n), :);
        
        % 合并到结果
        top_biomarkers = [top_biomarkers; top_rows];
    end
    
    % 移除临时列 abs_t
    top_biomarkers.abs_t = [];
end
