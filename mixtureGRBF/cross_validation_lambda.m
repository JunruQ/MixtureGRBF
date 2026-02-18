function loglik = cross_validation_lambda(X,PTID,stage,nsubtype,options)
    % K-fold

    % initialize parameters
    numFolds = 10;
    lambdas = parse_param(options,'lambda',2.^linspace(-4,-4,7));
    npoints = size(X,1);
    L = max(stage) - min(stage) + 1;
    C = linspace(min(stage),max(stage),L);
    options.C = C;
    [subjects,~,ic] = unique(PTID, 'stable');
    N = size(subjects,1);
    indices = crossvalind('Kfold', N, numFolds);
    loglik = [];
    
    
    for lambda = lambdas
        fprintf('Begin cross validation on lambda = %f.\n', lambda);
        options.lambda = lambda;
        loglik_lambda = 0;
        for fold = 1:numFolds

            testSubjects = (indices == fold);
            trainSubjects = ~testSubjects;

            testIndices = ismember(PTID, subjects(testSubjects));
            trainIndices = ~testIndices;
            
            [mdl,~,~] = MGRBF_model(X(trainIndices,:),PTID(trainIndices,:),stage(trainIndices,:),nsubtype,options);
            loglik_test = cal_mgrbf_loglik(X(testIndices,:), PTID(testIndices,:),stage(testIndices,:),mdl,options);
            loglik_lambda = loglik_lambda + loglik_test;
            fprintf('Fold %d: Train on %d samples, Test on %d samples\n', fold, sum(trainIndices), sum(testIndices));
        end
        loglik = [loglik; [lambda, loglik_lambda]];
        
    end

end    
