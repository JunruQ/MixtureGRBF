function [rs, ris] = analyze_cv()
%ANALYZE_CV 
% YZ: This whole file is added.
close all

%output_file_name = 'TADPOLE_kmeans_remove_CN_MCI_CV_812';
% output_file_name = 'Tadpole_FTR_kmeans_stagsel=0';
% output_file_name = 'ADNI41_fsl_FTR_kmeans_stagsel=0';
% output_file_name = 'ADNI41_FTR_kmeans_stagsel=0';
% output_file_name = 'ADNI41_fsx_oldcsf_FTR_kmeans_stagsel=0';

output_file_name = 'ukb_MixtureGRBF_cv_nsubtype';

% output_file_name = 'SCZ17_FTR_MCEM_stagsel=1';
% output_file_name = 'SCZ122_FTR_kmeans_stagsel=1';

% output_file_name = 'OASIS3_FTR_kmeans_stagsel=0';

% output_file_name = 'Depression_FTR_kmeans_stagsel=1';

cv_loglikelihood = readmatrix(['output/', output_file_name, '/loglikelihood_cross_validation.csv']);
figure, plot(sum(cv_loglikelihood,1));
xlabel('number of subtypes');
ylabel('loglikelihood from cross validation');

max_nsubtype = size(cv_loglikelihood, 2);

figure;
check_inds = 2:max_nsubtype;
for j = check_inds
    [cef_mat, rand_mat] = cal_stability(output_file_name, j, 0);

    subplot(2,length(check_inds),j-1); boxplot(cef_mat(:)); 
    xlabel([num2str(j),' subtypes']); ylabel('Trajectory correlation between folds');

    subplot(2,length(check_inds),j-1+length(check_inds)); boxplot(rand_mat(:));
    xlabel([num2str(j),' subtypes']); ylabel('Adjusted Rand Index between folds');

    rs(j) = mean(cef_mat(:),'omitnan');
    ris(j) = mean(rand_mat(:),'omitnan');
end
rs(1) = nan;
ris(1) = nan;

figure, plot(rs), title('mean of trajectory correlation between folds');
figure, plot(ris), title('mean of adjusted Rand index between folds')

end

