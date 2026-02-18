function bes_lambda = search_lambda(X,PTID,stage,nsubtype,ini_lambda,options)
    fprintf('Search best lambda on nsubtype = %d.\n', nsubtype);
    % initialize parameters
    max_search_count = 10;
    numFolds = 3;
    if ~ini_lambda
        ini_lambda = 512;
    end
    npoints = size(X,1);
    
    gaussian_interval = parse_param(options,'gaussian_interval',1);

    if ~isempty(parse_param('options','C',[]))
        C = options.C;
        L = (max(C) - min(C))/gaussian_interval + 1;
    else
        L = (max(stage) - min(stage))/gaussian_interval + 1;
        C = linspace(min(stage),max(stage),L);
        options.C = C;
    end

    [subjects,~,ic] = unique(PTID, 'stable');
    N = size(subjects,1);
    indices = crossvalind('Kfold', N, numFolds);
    loglik = [];

    % 初次搜索
    search_count = 0;
    for lambda = [ini_lambda, ini_lambda*2]
        loglik_record = cal_loglik_cv_lambda(lambda, indices, X, PTID, stage, nsubtype, numFolds, subjects, options);
        loglik = [loglik; loglik_record];
        search_count = search_count + 1;
    end
    
    % 确定搜索方向
    if loglik(1,2) >= loglik(2,2)
        search_direction = 'left';
        lambda = ini_lambda;
        loglik = flip(loglik, 1);
    else
        search_direction = 'right';
        lambda = ini_lambda * 2;
    end

    % 继续搜索直至触发条件
    search_termination_condition = 0;
    while ~search_termination_condition
        search_count = search_count + 1;
        switch search_direction
            case 'left'
                lambda = lambda / 2;
            case 'right'
                lambda = lambda * 2;
        end
        loglik_record = cal_loglik_cv_lambda(lambda, indices, X, PTID, stage, nsubtype, numFolds, subjects, options);
        loglik = [loglik; loglik_record];
        if loglik(search_count,2) < loglik(search_count - 1,2)
            search_termination_condition = 1;
        end
        if search_count == max_search_count
            search_termination_condition = 1;
        end
    end

    % 输出最大lambda并保存lambda的结果
    [~,bes_ind] = max(loglik(:,2));
    bes_lambda = loglik(bes_ind,1);
    lambda_loglik = array2table([loglik(:,1),loglik(:,2)],'VariableNames',{'lambda','loglik'});
    output_dir = ['./output/', options.output_file_name];
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    writetable(lambda_loglik,['output/',options.output_file_name,'/lambda_loglik_',int2str(nsubtype),'subtypes.csv']);
end

function loglik_record = cal_loglik_cv_lambda(lambda, indices, X, PTID, stage, nsubtype, numFolds, subjects, options)
    fprintf('\tBegin cross validation on lambda = %f.\n', lambda);
    options.lambda = lambda;
    loglik_lambda = 0;
    for fold = 1:numFolds
        testSubjects = (indices == fold);
        trainSubjects = ~testSubjects;
        testIndices = ismember(PTID, subjects(testSubjects));
        trainIndices = ~testIndices;
        fprintf('\t\tFold %d: Train on %d samples, Test on %d samples\n', fold, sum(trainIndices), sum(testIndices));
        [mdl,~,~] = layered_MGRBF_model(X(trainIndices,:),PTID(trainIndices,:),stage(trainIndices,:),nsubtype,options);
        loglik_test = cal_mgrbf_loglik(X(testIndices,:), PTID(testIndices,:),stage(testIndices,:),mdl,options);
        loglik_lambda = loglik_lambda + loglik_test;
        fprintf('\t\tLoglikelihood = %f.\n', loglik_test)
    end
    loglik_record = [lambda, loglik_lambda];
end
