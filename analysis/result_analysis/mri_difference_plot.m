function mri_difference_plot()

addpath(genpath('ENIGMA/matlab'))

nsubtype = 5;
t = readtable('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/mri_difference.csv');

cortical_region = {
    'bankssts'
    'caudalanteriorcingulate'
    'caudalmiddlefrontal'
    'cuneus'
    'entorhinal'
    'fusiform'
    'inferiorparietal'
    'inferiortemporal'
    'isthmuscingulate'
    'lateraloccipital'
    'lateralorbitofrontal'
    'lingual'
    'medialorbitofrontal'
    'middletemporal'
    'parahippocampal'
    'paracentral'
    'parsopercularis'
    'parsorbitalis'
    'parstriangularis'
    'pericalcarine'
    'postcentral'
    'posteriorcingulate'
    'precentral'
    'precuneus'
    'rostralanteriorcingulate'
    'rostralmiddlefrontal'
    'superiorfrontal'
    'superiorparietal'
    'superiortemporal'
    'supramarginal'
    'frontalpole'
    'temporalpole'
    'transversetemporal'
    'insula'
};

subcortical_region = {
    'accumbens'
    'amygdala'
    'caudate'
    'hippocampus'
    'pallidum'
    'putamen'
    'thalamus'
    'ventricles'
};

diff_plot(t, cortical_region, 5, 'cortical')
diff_plot(t, subcortical_region, 5, 'subcortical')

end

function diff_plot(t, regions, nsubtype, name)

vmax = max(abs(table2array(t(:,'signed_log10FDR'))));

hemisphere_label = {'Left', 'Right'};

region_num = length(regions);

for subtype = 1:nsubtype
    arr = zeros(region_num * 2, 1);

    for hemisphere_idx = 1:length(hemisphere_label)
        for region_idx = 1:region_num
            arr_idx = (hemisphere_idx - 1) * region_num + region_idx;
            region_name = [hemisphere_label(hemisphere_idx), ' ', regions(region_idx)];
            region_name = strjoin(region_name, '');
            value = t(strcmp(t.region, region_name) & t.subtype == subtype, 'signed_log10FDR');
            if isempty(value)
                value = 0;
                % disp([region_name, ' has no value']);
            else
                value = value.signed_log10FDR;   % 取表里的值
            end
            arr(arr_idx) = value;
        end
    end
    f = figure;
    if strcmp(name, 'cortical')
        arr_fsa5 = parcel_to_surface(arr, 'aparc_fsa5');
        plot_cortical(arr_fsa5, 'surface_name', 'fsa5', 'color_range', ...
            [-vmax, vmax], 'cmap', 'RdBu_r')
    else % subcortical
        plot_subcortical(arr, 'color_range', ...
            [-vmax, vmax], 'cmap', 'RdBu_r')
    end
    export_fig(['output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/mri_difference_',name,'_subtype',num2str(subtype),'.png'], '-r500')
    close(f)
end

end