function [rs, ris] = analyze_cv()
close all

output_file_name = 'synth_MixtureGRBF_cv_nsubtype';

cv_loglikelihood = readmatrix(['output/', output_file_name, '/loglikelihood_cross_validation.csv']);
figure, plot(sum(cv_loglikelihood,1));
xlabel('number of subtypes');
ylabel('loglikelihood from cross validation');

max_nsubtype = size(cv_loglikelihood, 2);

% figure;
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

export_fig(['output/result_analysis/', output_file_name, '/trajectory_correlation.jpg'], '-r500', '-transparent');

% figure, plot(ris), title('mean of adjusted Rand index between folds')

end

