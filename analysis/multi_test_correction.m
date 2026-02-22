function ind_reject = multi_test_correction(ps, level, mode)
%MULTI_TEST_CORRECTION Summary of this function goes here
%   Detailed explanation goes here
m = length(ps);
if strcmp(mode, 'bonferroni')
    ind_reject = ps < level / m;
elseif strcmp(mode, 'BH') % Benjaminin-Hochberg correction
    [ps_sorted, inds] = sort(ps);
    compare = ps_sorted(:) <= (1:m)'/m*level;
    k = find(~compare, 1);
    if isempty(k)
        ind_reject = true(size(ps));
    elseif k == 1
        ind_reject = false(size(ps));
    else
        k = k-1;
        ind_reject = false(size(ps));
        ind_reject(inds(1:k)) = true;
    end
end

end

