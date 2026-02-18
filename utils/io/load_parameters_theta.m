function mdl = load_parameters_theta(save_dir)
%LOAD_PARAMETERS_THETA Summary of this function goes here
%   Detailed explanation goes here
proption = readmatrix(strcat(save_dir,'/proportion.csv'));
nsubtype = length(proption);

sigma = readmatrix(strcat(save_dir,'/pred_sigma.csv'));
for p = 1:nsubtype
    re_traj(:,:,p) = readmatrix([save_dir,'/trajectory',int2str(p),'.csv']);
    weight(:,:,p) = readmatrix([save_dir,'/weight',int2str(p),'.csv']);
end

mdl.sigma2 = sigma;
mdl.f = re_traj;
mdl.pi = proption;
mdl.W = weight;

end

