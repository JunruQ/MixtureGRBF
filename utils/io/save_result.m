function save_result(all_data, stage_all, subtype_all, mdl, ...
    biomarker_name, extra, save_dir1)

num_subtype = size(mdl.f, 3);

writetable(array2table([all_data.RID,stage_all,subtype_all], ...
    'VariableNames', {'PTID','stage','subtype'}), ...
    [save_dir1,'/subtype_stage.csv'])

save_parameters_theta(save_dir1, mdl, biomarker_name);

if ~isempty(extra)    
    if isfield(extra, 'subtype_prob')
        for k = 1:num_subtype
            subtype_names{k} = ['subtype', int2str(k)];
        end

        subtype_prob = array2table([all_data.RID, all_data.years, extra.subtype_prob], ...
            'VariableNames', [{'PTID' ,'years_from_baseline'},subtype_names]);
        writetable(subtype_prob, [save_dir1,'/subtype_prob.csv']);
    end

    % if isfield(extra, 'loglik_all')
    %     writematrix(extra.loglik_all, [save_dir1,'/loglik_all.csv']);
    % end
end


end

