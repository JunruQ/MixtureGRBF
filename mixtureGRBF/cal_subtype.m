function [subtype,extra] = cal_subtype(dat, PTID, stage, mdl, options)
    % use the component to reconstruct the trajectory
    % but not the gaussian center

    [subjects,~,ic] = unique(PTID,'stable');
    X = dat;
    W = mdl.W;
    [npoints, D] = size(X);
    N = size(subjects,1);
    K = size(W,3);
    pis = mdl.pi;
    sigma_squares = mdl.sigma2;

    C = parse_param(options,'C',[]);
    if isempty(C)
        [C, L] = compute_C_and_L(stage);
    else
        gaussian_interval = parse_param(options,'gaussian_interval',1);
        L = (max(C) - min(C))/gaussian_interval + 1;
    end

    % reconstruct trajactories
    sigma_gaussian = parse_param(options,'sigma_gaussian',1);
    diff = repmat(stage, [1, L]) - repmat(C, [npoints, 1]);
    Phi = exp(-diff.^2 / (2*sigma_gaussian^2)) / (sqrt(2*pi)*sigma_gaussian);
    f = zeros(npoints,D,K);
    for k = 1:K
        f(:,:,k) = Phi * W(:,:,k);
    end

    loglkh_subjects = zeros(N,K);
    for k = 1:K
        loglkh_k_points = logmvn(X,f(:,:,k),diag(sigma_squares(:,k)));
        loglkh_k_subjects = accumarray(ic,loglkh_k_points);
        loglkh_subjects(:,k) = loglkh_k_subjects;
    end
    loglkh_subjects = bsxfun(@plus, loglkh_subjects, -max(loglkh_subjects, [], 2));
    gamma = exp(loglkh_subjects) .* repmat(pis, [N,1]);
    gamma = bsxfun(@rdivide, gamma, sum(gamma, 2));
    Ts = accumarray(ic, ones(npoints,1));
    [~,subtype] = max(gamma,[],2);
    subtype = repelem(subtype, Ts, 1);
    gamma = repelem(gamma, Ts, 1);
    extra.subtype_prob = gamma;
end


