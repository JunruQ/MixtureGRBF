function mat_z_score = reg_norm_tadpole(input_data)

input = input_data{:,:};
diagnosis = input(:,3);
mat = input(:,11:end);
mat_icv_ro = zeros(size(mat));
mat_all_ro = zeros(size(mat));
mat_z_score = zeros(size(mat));
nbiom = size(mat,2);

ICV = input(:,9);
cov = input(:,5:8);
abeta = input(:,10);

idx_CN = diagnosis==0 & abeta>880;
cov_CN = input(idx_CN,5:8);
icv_CN = input(idx_CN,9);
mat_CN = input(idx_CN,10:end);
one = ones(sum(idx_CN),1);

mean_icv_CN = mean(icv_CN);
mean_cov_CN = mean(cov_CN,1);

% regress out ICV
b_icv = zeros(nbiom,2);
p_icv = zeros(nbiom,1);
for i = 1:nbiom
    [b_icv(i,:),~,~,~,stats] = regress(mat_CN(:,i),[icv_CN,one]);
    p_icv(i) = stats(3);
    mat_icv_ro(:,i) = mat(:,i) - b_icv(i,1) * (ICV - mean_icv_CN);
end

% covariant p-value
mat_icv_ro_CN = mat_icv_ro(idx_CN,:);
b_cov = zeros(nbiom,4);
p_cov = zeros(nbiom,4);
for i = 1:nbiom
    for j = 1:4
        [b,~,~,~,stats] = regress(mat_icv_ro_CN(:,i),[cov_CN(:,j),one]);
        b_cov(i,j) = b(1);
        p_cov(i,j) = stats(3);
    end
end

% regress out age/sex/edu/APOE
ro_tf = p_cov < 0.05 * ones(size(p_cov));
p_all = zeros(nbiom,1);
for i = 1:nbiom
    [b,~,~,~,stats] = regress(mat_icv_ro_CN(:,i),[cov_CN(:,ro_tf(i,:)),one]);
    p_all(i) = stats(3);
    if sum(ro_tf(i,:)) == 0
        mat_all_ro(:,i) = mat_icv_ro(:,i);
    else
        mat_all_ro(:,i) = mat_icv_ro(:,i) - (cov(:,ro_tf(i,:))-mean_cov_CN(:,ro_tf(i,:))) * b(1:end-1);
    end
end

% normalize z-score
for i = 1:nbiom
    pd = fitdist(mat_all_ro(idx_CN,i),'Normal');
    mat_z_score(:,i) = (mat_all_ro(:,i)-pd.mu)/pd.sigma * -1;
end

output_data = input_data;
output_data{:,11:end} = mat_z_score;
writetable(output_data,'./input/Tadpole/TADPOLE_input_pass_z.csv')

end