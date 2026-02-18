function save_parameters_theta(save_dir, mdl, biomarker_name)
%SAVE_PARAMETERS_THETA Summary of this function goes here
%   Detailed explanation goes here
if nargin < 3
    biomarker_name = [];
end

sigma = mdl.sigma2;
re_traj = mdl.f;
proption = mdl.pi;
weight = mdl.W;
    
nsubtype = length(proption);

writematrix(sigma, strcat(save_dir,'/pred_sigma.csv'));
for i = 1:nsubtype
    if isempty(biomarker_name)
        writematrix(re_traj(:,:,i), [save_dir, '/trajectory',int2str(i),'.csv'])
    else
        writetable(array2table(re_traj(:,:,i), 'VariableNames', biomarker_name), ...
            [save_dir, '/trajectory',int2str(i),'.csv']);
    end
end
writematrix(proption, strcat(save_dir,'/proportion.csv'));

for i = 1:nsubtype
    if isempty(biomarker_name)
        writematrix(weight(:,:,i), [save_dir, '/weight',int2str(i),'.csv'])
    else
        writetable(array2table(weight(:,:,i), 'VariableNames', biomarker_name), ...
            [save_dir, '/weight',int2str(i),'.csv']);
    end
end

end

