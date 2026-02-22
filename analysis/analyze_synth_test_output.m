function analyze_synth_test_output(nsubtype,source_folder,output_folder,plot_dim_num,source_range,output_range)
close all
if nargin < 1
    nsubtype = 3;
end


if nargin < 2
    source_folder = './output/synth_MixtureGRBF_cv_nsubtype/';
end

if nargin < 3
    output_folder = './output/synth_MixtureGRBF_test/';
end

if nargin < 4
    plot_dim_num = 10;
end

if nargin < 5
    source_range = linspace(30,90,1001);
end

if nargin < 6
    output_range = 30:90;
end

output_traj_dir = [output_folder,int2str(nsubtype),'_subtypes/'];

for k = 1:nsubtype
    figure;
    source_traj_k = readmatrix([source_folder,'synth_traj_',int2str(k),'.csv']);
    source_traj_k = zscore(source_traj_k);
    output_nsubtype_order = [3,2,1];
    output_traj_k = readtable([output_traj_dir,'trajectory',int2str(output_nsubtype_order(k)),'.csv']);
    for i = 1:plot_dim_num
        for j = 1:plot_dim_num
            subplot(plot_dim_num, plot_dim_num, (i-1)*plot_dim_num + j);
            
            plot(source_range,source_traj_k(:,(i-1)*plot_dim_num + j));
            hold on;
            plot(output_range,output_traj_k{:,(i-1)*plot_dim_num + j});
            hold off;
            title(sprintf('dim %d', (i-1)*plot_dim_num + j));
        end
    end
end
end