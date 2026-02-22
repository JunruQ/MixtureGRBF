function [outputArg1,outputArg2] = analyze_hyperparameters_on_depression(inputArg1,inputArg2)
%ANALYZE_HYPERPARAMETERS_ON_DEPRESSION Summary of this function goes here
%   Detailed explanation goes here

corrs = [];
rand_indices = [];

for i = 1:5
    options = [];
    options.nsubtype = [1,3];
    options.lambda = 20;
    options.lambda2 = 0.02;
    options.scale_lambdas = 0;
    run_algo('Depression', 'FTR', options);
    [rs, ris] = analyze_cv();
    corrs(1,i) = rs(3);
    rand_indices(1,i) = ris(3);

    options.lambda = 10;
    options.lambda2 = 0.01;
    options.scale_lambdas = 1;
    [m,n] = ndgrid([1,3], 0:10);
    Z = [m(:),n(:)];
    options.specify_nsubtype_fold_to_run = Z;
    run_algo('Depression', 'FTR', options);
    [rs, ris] = analyze_cv();
    corrs(2,i) = rs(3);
    rand_indices(2,i) = ris(3);

    corrs
    rand_indices
end

end

