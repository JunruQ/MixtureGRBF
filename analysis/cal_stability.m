function [cef_mat, rand_mat] = cal_stability(output_file_name, nsubtype, show_fig)
addpath(genpath('mixtureGRBF/'))
addpath('utils/')

% YZ: added
if nargin < 1

    %output_file_name = 'TADPOLE_kmeans_remove_CN_MCI_CV_812';
    %output_file_name = 'TADPOLE_cross_validation_815';
    % output_file_name = 'SCZdata_17features_kmeans_8_18';
    output_file_name = 'Depression_data_kmeans_stage_selection';
    
    nsubtype = 3;

    show_fig = 1;
end


num_fold = 5;

% YZ: added
[num_int, nbiom, nsamp] = get_num_points_biomarkers_and_samples(output_file_name, nsubtype);

%nbiom = 40;
%nbiom = 122;
%nbiom = 19;

% num_int = 1001;
traj_all = zeros(nsubtype,nbiom,num_int,num_fold);
traj_mid = zeros(nsubtype,nbiom,num_fold);
per_mat = zeros(nsubtype,num_fold);

%% Mean Atrophy
for i = 1:num_fold
    for j = 1:nsubtype
        save_dir = ['./output/',output_file_name,'/cross_validation_nsubtype',int2str(nsubtype),'_fold',int2str(i),'/trajectory',int2str(j)];
        traj = readmatrix(save_dir);
        traj_all(j,:,:,i) = traj';
        traj_mid(j,:,i) = mean(traj,1);
    end
    if i > 1
        A = traj_mid(:,:,i);
        [P,B] = permute_endmembers(traj_mid(:,:,1),A);
        traj_mid(:,:,i) = B;
        traj_i = shiftdim(traj_all(:,:,:,i), 1);
        traj_i_perm = P * reshape(traj_i, [num_int*nbiom, nsubtype])';
        traj_all(:,:,:,i) = shiftdim(reshape(traj_i_perm', [nbiom,num_int,nsubtype]), 2);
        %traj_all(:,:,:,i) = pagemtimes(P,traj_all(:,:,:,i));
    end
end

% YZ: added. Check if there are flat trajectories
if show_fig
    figure;
    for j = 1:nsubtype
        subplot(1,nsubtype,j);
        plot(squeeze(traj_all(j,:,:,1))');
    end
end

cef_mat = zeros(num_fold,num_fold,nsubtype);
dist_mat = zeros(num_fold,num_fold);
for i = 1:num_fold
    for j = 1:num_fold
        % YZ: Changed it to MAD
        dist_all = abs(traj_all(:,:,:,i) - traj_all(:,:,:,j));
        dist_mat(i,j) = mean(dist_all(:));
            
        for k = 1:nsubtype
            traj_ki = traj_all(k,:,:,i);
            traj_kj = traj_all(k,:,:,j);
            cef = corrcoef(traj_ki(:),traj_kj(:));
%             cef = corrcoef(traj_mid(k,:,i),traj_mid(k,:,j));
            cef_mat(i,j,k) = cef(1,2);
        end
    end
end

for i = 1:num_fold
   cef_mat(i,i,:) = nan;
end

% YZ: changed
if show_fig
    figure, boxplot(cef_mat(:)), title('Trajectory correlation between folds')
end

%% Rand Index
%nsamp = 1023;
%nsamp = 1096;
%nsamp = 871;
subtype = zeros(nsamp,num_fold);
for i = 1:num_fold
        save_dir1 = ['./output/',output_file_name,'/cross_validation_nsubtype',int2str(nsubtype),'_fold',int2str(i),'/train_subtype_stage'];
        train_subtype_stage = readmatrix(save_dir1);
        save_dir2 = ['./output/',output_file_name,'/cross_validation_nsubtype',int2str(nsubtype),'_fold',int2str(i),'/test_subtype_stage'];
        test_subtype_stage = readmatrix(save_dir2);
        % TODO: Change the code to remove 1:4 later
        subtype_stage = [train_subtype_stage;test_subtype_stage(:,1:4)];
        [~,I] = sort(subtype_stage(:,1));
        subtype(:,i) = subtype_stage(I,3);
end

rand_mat = zeros(num_fold,num_fold);
for i = 1:num_fold
    for j = 1:num_fold
        % YZ: changed it to adjusted Rand index to account for the effect
        % of number of clusters. The two ways to compute the adjusted Rand
        % index yield the same number
        [rand_mat(i,j),~,~,~] = rand_index(subtype(:,i),subtype(:,j));
%         rand_mat1(i,j) = rand_index1(subtype(:,i),subtype(:,j),'adjusted');
    end
end

for i = 1:num_fold
    rand_mat(i,i,:) = nan;
end

% YZ: moved it after setting nan to the diagonal
a = mean(rand_mat,"all",'omitnan');

% YZ: added
if show_fig
    figure, boxplot(rand_mat(:)), title('Adjusted Rand index between folds')
end

%cm = confusionchart(subtype(:,1),subtype(:,10));

% nsamp1 = size(output1,1);
% nsamp2 = size(output2,1);
% 
% stage1 = output1.stage1;
% subtype1 = output1.subtype1;
% 
% stage2 = zeros(nsamp1,1);
% subtype2 = zeros(nsamp1,1);
% 
% for i = 1:nsamp1
%     j = find(output2{:,1} == output1{i,1} & output2{:,2} == output1{i,2});
%     stage2(i) = output2.stage2(j);
%     subtype2(i) = output2.subtype2(j);
% end

end

% YZ: added
function [num_int, nbiom, nsamp] = get_num_points_biomarkers_and_samples(output_file_name, nsubtype)
save_dir = ['./output/',output_file_name,'/cross_validation_nsubtype', ...
    int2str(nsubtype),'_fold',int2str(1),'/trajectory',int2str(1)];
traj = readmatrix(save_dir);
[num_int, nbiom] = size(traj);

save_dir1 = ['./output/',output_file_name,'/cross_validation_nsubtype', ...
    int2str(nsubtype),'_fold',int2str(1),'/train_subtype_stage'];
train_subtype_stage = readmatrix(save_dir1);
save_dir2 = ['./output/',output_file_name,'/cross_validation_nsubtype', ...
    int2str(nsubtype),'_fold',int2str(1),'/test_subtype_stage'];
test_subtype_stage = readmatrix(save_dir2);
nsamp = size(train_subtype_stage, 1) + size(test_subtype_stage, 1);
end
