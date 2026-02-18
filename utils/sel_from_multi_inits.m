function bes_ep = sel_from_multi_inits(subtype, loglik_all, options)
%SEL_FROM_MULTI_INITS Summary of this function goes here
%   Detailed explanation goes here

% the strategy that selects the largest log-likelihood has a median RI of
% 0.52 while the strategy that selects the most common partition has a
% median RI of 0.97 over 5 runs, each with 20 random initializations

% compare_strategy_stability(subtype, loglik_all);

sel_strategy = parse_param(options, 'select_from_multi_inits', 'partition');
% use one strategy
if strcmp(sel_strategy, 'loglikelihood')
    bes_ep = sel_based_on_loglik(loglik_all);
elseif strcmp(sel_strategy, 'partition')
    bes_ep = sel_based_on_partition(subtype);
end


end

function compare_strategy_stability(subtype, loglik_all)
max_ep = size(subtype,2);
inds = crossvalind('Kfold', max_ep, 5);

% compare two strategies on selecting from multiple initializations
for k = 1:5
    subtype_k = subtype(:, inds == k);
    loglik_k = loglik_all(:, inds == k);
    
    bes_ep = sel_based_on_loglik(loglik_k);
    bes_subtype(:,k) = subtype_k(:,bes_ep);
    
    bes_ep1 = sel_based_on_partition(subtype_k);
    bes_subtype1(:,k) = subtype_k(:,bes_ep1);
end

% evaluate the 2 strategies using partition similarity
for ep1 = 1:5
    for ep2 = 1:5
        similarity_mat(ep1, ep2) = rand_index(bes_subtype(:,ep1), bes_subtype(:,ep2));
        similarity_mat1(ep1, ep2) = rand_index(bes_subtype1(:,ep1), bes_subtype1(:,ep2));
    end
end

end

function bes_ep = sel_based_on_loglik(loglik_all)
loglik_mat = loglik_all(end,:);
[~,bes_ep] = max(loglik_mat);
end

function bes_ep1 = sel_based_on_partition(subtype)
max_ep = size(subtype, 2);
similarity_mat = zeros(max_ep, max_ep);
for ep1 = 1:max_ep
    for ep2 = 1:max_ep
        if any(isnan(subtype(:,ep1))) || any(isnan(subtype(:,ep2)))
            similarity_mat(ep1, ep2) = NaN;
        else
            similarity_mat(ep1, ep2) = rand_index(subtype(:,ep1), subtype(:,ep2));
        end
    end
end
sim_vec = mean(similarity_mat, 2, 'omitnan');
[~, bes_ep1] = max(sim_vec);

% figure, histogram(similarity_mat(:));
% figure,plot(loglik_all);

end