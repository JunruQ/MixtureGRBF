function show_trajectory_in_PCA(re_traj, data, subtype)
%SHOW_TRAJECTORY_IN_PCA Summary of this function goes here
%   Detailed explanation goes here

nsubtype = size(re_traj, 3);

h = figure;
c = distinguishable_colors(nsubtype);
trajec = reshape(re_traj,size(re_traj,1),[],1);

% [coeffs,data_pca,latent,tsquared,explained,mu] = pca(trajec', 'NumComponents', 3);
[coeffs,data_pca,latent,tsquared,explained,mu] = pca(data, 'NumComponents', 3);

hs = [];

for k = 1:nsubtype
    traj_pca = (re_traj(:,:,k)' - repmat(mu, [size(re_traj(:,:,k),2), 1])) * coeffs;
    hs(k) = plot3(traj_pca(:,1), traj_pca(:,2), traj_pca(:,3),'-','Color',c(k,:),'LineWidth',5);
    hold on;
    data_k = data(subtype==k,:);
    data_pca = (data_k - repmat(mu, [size(data_k,1), 1])) * coeffs;
    scatter3(data_pca(:,1), data_pca(:,2), data_pca(:,3),2,c(k,:));
    hold on;
end

subtype_names = cellfun(@(x) ['Subtype ',int2str(x)], num2cell((1:nsubtype)),'UniformOutput',false);
legend(hs, subtype_names);

end

