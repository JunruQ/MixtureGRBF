function dat_sel = sel_component(dat, num_component, options)

if nargin < 3
    options = [];
end

sel_strategy = parse_param(options, 'select_component', 'pca');
% use one strategy
if strcmp(sel_strategy, 'random')
    dat_sel = sel_component_random(dat, num_component);
elseif strcmp(sel_strategy, 'pca')
    dat_sel = sel_component_pca(dat, num_component);
end

end

function dat_sel = sel_component_random(dat, num_component)
random_cols = randi(size(dat,2), 1, num_component);
dat_sel = dat(:,random_cols);
end

function dat_sel = sel_component_pca(dat, num_component)
[~,score,~] = pca(dat);
dat_sel = score(:,1:num_component);
end