function loglkh = cal_mgrbf_loglik(X, PTID, stage, model, options)
    [subjects,~,ic] = unique(PTID, 'stable');
    N = size(subjects,1);
    f = model.f;
    K = size(f,3);
    sigma_squares = model.sigma2;
    C = options.C;
    [~, ind] = ismember(stage, C);
    pis = model.pi;

    loglkh_subjects = zeros(N,K);
    for k = 1:K
        f_k = f(:,:,k);
        f_k = f_k(ind,:);
        loglkh_k_points = logmvn(X,f_k,diag(sigma_squares(:,k)));
        loglkh_k_subjects = accumarray(ic,loglkh_k_points);
        loglkh_subjects(:,k) = loglkh_k_subjects;
    end
    loglkh_mixture = calc_log_mixture(loglkh_subjects,pis);
    loglkh = sum(loglkh_mixture);
end