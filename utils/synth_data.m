function synth_data(output_dir,options)

if nargin < 2
    options = [];
end

if nargin < 1
    output_dir = './output/synth_data';
end

if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

% 生成轨迹
ndim = parse_param(options,'ndim',300);
traj_min_x = parse_param(options,'min_x',30);
traj_max_x = parse_param(options,'max_x',90);
traj_y = [];
nsubtype = parse_param(options,'nsubtype',3);
num_pts_control = parse_param(options,'num_pts_control',4);

for k = 1:nsubtype
    traj_k = [];
    for i = 1:ndim
        [traj_i,traj_x] = create_traj_data_randomly(traj_min_x,traj_max_x,num_pts_control,1);
        traj_k = horzcat(traj_k,traj_i');
    end
    traj_y(:,:,k) = traj_k;
    writematrix(traj_k, [output_dir, '/synth_traj_', int2str(k),'.csv'])
end

% 生成数据
npoints = parse_param(options,'npoints',3000);
data = [];
for i = 1:npoints
    subtype_npoint = randi([1, nsubtype]);
    stage_npoint = randi([traj_min_x,traj_max_x]);
    data_j = [];
    for j = 1:ndim
        data_npoint_j = interp1(traj_x, traj_y(:,j,subtype_npoint), stage_npoint);
        data_j = [data_j, data_npoint_j];
    end
    data_j = [subtype_npoint, stage_npoint, data_j];
    data(i,:) = data_j;
end

% 添加Gaussian noise
sigma = 0.01 + 0.001 * randn(1,ndim);
for i = 1:ndim
    gaussian_noise_dim = sigma(i) * randn(1, npoints);
    data(:,i+2) = data(:,i+2) + gaussian_noise_dim';
end

writematrix(data, [output_dir, '/synth_data.csv'])

end

function [traj_y_interp, traj_x_interp]= create_traj_data_randomly(start_pt, end_pt, num_pts, maximum_traj)

traj_x = linspace(start_pt, end_pt, num_pts);
traj_y = rand(1, num_pts) * maximum_traj;

traj_sigma = maximum_traj .^ 2 * 200;
traj_x_interp = linspace(min(traj_x), max(traj_x), 1001);
traj_y_interp = interp1(traj_x, traj_y, traj_x_interp);
traj_y_interp = imgaussfilt(traj_y_interp, traj_sigma);
end
