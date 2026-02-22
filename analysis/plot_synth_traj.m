function plot_synth_traj()
close all

data = readtable('./output/synth_data/synth_data.csv');
k = 1;
traj = readtable(['./output/synth_data/synth_traj_',int2str(k),'.csv']);
plot_dim_num = 5;
figure;
for i = 1:plot_dim_num
    for j = 1:plot_dim_num
        subplot(plot_dim_num, plot_dim_num, (i-1)*plot_dim_num + j);
        plot(linspace(30,90,1001),traj{:,(i-1)*plot_dim_num + j})

        hold on

        data_dim = data{data{:,1} == k,:};

        scatter(data_dim(:,2)+randn(size(data_dim,1),1),data_dim(:,(i-1)*plot_dim_num + j +2),1)
        hold off

        title(sprintf('dim %d', (i-1)*plot_dim_num + j));
    end
end


