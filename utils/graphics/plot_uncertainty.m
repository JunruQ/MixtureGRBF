function h = plot_uncertainty(X, Y, options)
% Input:
%   X - 1 x N vector
%   Y - M x N matrix or 1 x N cell array

%% Default options
if nargin < 3
    options = [];
end

handle = parse_param(options, 'figure', gcf);
%color_area = parse_param(options, 'color_area', [128 193 219]./255);    % Blue theme
%color_line = parse_param(options, 'color_line', [ 52 148 186]./255);
%options.color_area = [243 169 114]./255;    % Orange theme
%options.color_line = [236 112  22]./255;
color_area = parse_param(options, 'color_area', [169,169,169]./255);    % Black theme
color_line = parse_param(options, 'color_line', [0 0 0]./255);
alpha = parse_param(options, 'alpha', 0.5);
line_width = parse_param(options, 'line_width', 2);

center_type = parse_param(options, 'center_type', 'mean');
error_type = parse_param(options, 'error_type', 'std');

plot_type = parse_param(options, 'plot_type', 'plot');

%%
if size(X,2) == 1
    Y = Y';
    X = X';
end

if ~iscell(Y) % Y is M x N
    Y1 = {};
    for i = 1:size(Y,2)
        Y1{i} = Y(:,i);
    end
    Y = Y1;
end


%% Computing the mean and standard deviation of the data matrix
data_center = [];

if strcmp(center_type, 'mean')    
    for i = 1:length(Y)
        data_center(:,i) = mean(Y{i},1);
    end
elseif strcmp(center_type, 'median')
    for i = 1:length(Y)
        data_center(:,i) = median(Y{i},1);
    end
end

for i = 1:length(Y)
    data_std(:,i)  = std(Y{i},0,1);
end

% Type of error plot
switch(error_type)
    case 'std'
        error = data_std;
        upper_bd = data_center + error;
        bottom_bd = data_center - error;
    case 'q1-q3' 
        for i = 1:length(Y)
            upper_bd(:,i) = quantile(Y{i}, 0.75);
            bottom_bd(:,i) = quantile(Y{i}, 0.25);
        end
    otherwise
end

% Plotting the result
h = figure(handle);

if strcmp(plot_type, 'plot')
    %plot(X, data_center, 'color', color_line, 'LineWidth', line_width);
elseif strcmp(plot_type, 'semilogx')
    semilogx(X, data_center, 'color', color_line, 'linewidth', line_width);
    grid on
elseif strcmp(plot_type, 'loglog')
    loglog(X, data_center, 'color', color_line, 'linewidth', line_width);
    grid on
end
hold on;

x_vector = [X, fliplr(X)];
patch = fill(x_vector, [upper_bd,fliplr(bottom_bd)], color_area);
set(patch, 'edgecolor', 'none');
set(patch, 'FaceAlpha', alpha);
hold off;

% send patch object to background
set(gca,'children',flipud(get(gca,'children')))

end