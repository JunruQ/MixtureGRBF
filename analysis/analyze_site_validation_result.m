function analyze_site_validation_result()

input_data = readtable('input/ukb/ukb_covreg1_trans1_nanf1_biom17.csv');

site_table_path = 'input/ukb/ukb_site_map.csv';
site_table = readtable(site_table_path, 'ReadVariableNames', false);

nsubtype = 5;
all_result_subtype_path = ['output/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/subtype_stage.csv'];
[~, all_sorted_subtype] = all_cause_mortality_sort(nsubtype, all_result_subtype_path);

sites = {'Northern England', 'Southern England', 'Midlands', 'Scotland', 'Wales'};   

% Create figure with tiledlayout for better spacing control
figure('Position', [100, 100, 1000, 400], 'Visible', 'off');
t = tiledlayout(2, 5, 'TileSpacing', 'loose', 'Padding', 'loose'); % 3 rows, 4 columns for side-by-side subplots

%% Confusion matrices

for site_idx = 1:numel(sites)
    site = sites{site_idx};
    
    site_result_subtype_path = ['output/ukb_MixtureGRBF_site_validation/5_subtypes/', site, '/subtype_stage.csv'];
    [~, site_sorted_subtype] = all_cause_mortality_sort(nsubtype, site_result_subtype_path);

    site_table_idx = strcmp(site_table.Var2, site);
    site_names = site_table.Var1(site_table_idx);
    test_inds = ismember(input_data.centre, site_names);
    nsamp = length(test_inds);
    train_inds = ones(nsamp,1) - test_inds;

    train_conf_matrix = zeros(nsubtype);
    test_conf_matrix = zeros(nsubtype);

    % Fill confusion matrix
    for i = 1:nsubtype
        for j = 1:nsubtype
            subtype_i = all_sorted_subtype == i;
            subtype_j = site_sorted_subtype == j;

            common_count = subtype_i & subtype_j;

            train_common_count = train_inds' & common_count;
            test_common_count = test_inds' & common_count;

            train_conf_matrix(i, j) = sum(train_common_count);
            test_conf_matrix(i, j) = sum(test_common_count);
        end
    end

    % percentage consistency
    train_percentage_consistency = sum(diag(train_conf_matrix)) / sum(train_conf_matrix(:));
    test_percentage_consistency = sum(diag(test_conf_matrix)) / sum(test_conf_matrix(:));
    disp({site, ':', train_percentage_consistency, ',', test_percentage_consistency});
    
    nexttile(t, site_idx);
    colormap sky;
    imagesc(train_conf_matrix);
    % colorbar;
    title(site, 'FontWeight', 'normal');
    if site_idx == 1
        ylabel('Training subtype');
    end
    yticks(1:nsubtype);
    xticks(1:nsubtype);
    xlabel('All data subtype');
    
    % Add text labels
    for i = 1:nsubtype
        for j = 1:nsubtype
            text(j, i, num2str(train_conf_matrix(i, j)), ...
                 'HorizontalAlignment', 'center')
        end
    end

    set(gca, 'FontName', 'Arial');
    
    % Right subplot (Test)
    % nexttile(sub_t, 2);
    nexttile(t, site_idx + 5);
    imagesc(test_conf_matrix);
    % colorbar;
    % title(site, 'FontWeight', 'normal');
    if site_idx == 1
        ylabel('Validation subtype');
    end
    xticks(1:nsubtype);
    yticks(1:nsubtype);
    xlabel('All data subtype');
    
    % Add text labels
    for i = 1:nsubtype
        for j = 1:nsubtype
            text(j, i, num2str(test_conf_matrix(i, j)), ...
                 'HorizontalAlignment', 'center');
        end
    end

    set(gca, 'FontName', 'Arial');
end

% Add overall title
% title(t, 'Site Validation Confusion Matrices', 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
export_fig 'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/site_validation_confusion_matrices.png' -r500 -transparent

close;

end