% function [model,subtype,extra] = mixtureGRBF(X, PTID, stage, nsubtypes, pre_subtype, pre_sgm, pre_A, pre_B, options)
function [model,subtype,extra] = mixtureGRBF(X, PTID, stage, nsubtypes, pre_subtype, options)

% X - npoints by D data, notice that X should not contain any NaN.
% PTID - npoints by 1 data
% stage - npoints by 1 data
% nsubtypes - integer
% options - structure for other options
    
    %% initialize

    % output
    model = [];
    extra = [];

    % C_extension_size

    C_extension_size = parse_param(options,'C_extension_size',2);
    
    % parse param from pre_model
    pre_sgm = parse_param(options, 'sigma2', []);
    pre_A = parse_param(options, 'A', []);
    pre_B = parse_param(options, 'B', []);
    pre_C = parse_param(options, 'C', []);

    % max_iter
    max_iter = parse_param(options,'max_iter',50);
    
    % number of subtypes
    K = nsubtypes;

    % number of points & dimension
    [npoints, D] = size(X);
    
    % subjects 
    [subjects,~,ic] = unique(PTID, 'stable');
    N = size(subjects,1);
    Ts = accumarray(ic, ones(npoints,1));
    
    % random subtype
    if ~isempty(pre_subtype)
        subtype = pre_subtype;
    else    
        subtype = randi([1,K],N,1);
    end    

    % gamma_ik
    gamma = dummyvar(categorical(subtype, 1:K));
    
    % pi_k
    pis = sum(gamma,1) / N;
    
    % specify mixture gaussian
    % set the interval to be 1 and the variance to be 1 as default
    gaussian_interval = parse_param(options, 'gaussian_interval', 1);
    sigma_gaussian = parse_param(options,'sigma_gaussian',1);
    if ~isempty(pre_C)
        C = pre_C;
        L = (max(C) - min(C))/gaussian_interval + 1;
    else
        L = (max(stage) - min(stage))/gaussian_interval + 1;
        C = linspace(min(stage),max(stage),L);
    end
    
    left_extention = ones(1,C_extension_size) * min(C) - (C_extension_size:-1:1) * gaussian_interval;
    right_extention = ones(1,C_extension_size) * max(C) + (1:C_extension_size) * gaussian_interval;
    C = [left_extention, C, right_extention];
    L = L + C_extension_size * 2;

    % lambda & Laplace
    lambda = parse_param(options,'lambda',1);
    lambda = lambda * npoints;
    Lap = diag(ones(L,1)) + diag(-2 * ones(L-1,1),-1) + diag(ones(L-2,1),-2);
    Lap = Lap(3:end,:);

    % compute Phi
    diff = repmat(stage, [1, L]) - repmat(C, [npoints, 1]);
    Phi = exp(-diff.^2 / (2*sigma_gaussian^2)) / (sqrt(2*pi)*sigma_gaussian);
    % Phi(Phi<1e-15) = 0;
    
    % compute A, B
    if isempty(pre_A) && isempty(pre_B)
        A_npoints = pagemtimes(reshape(Phi',L,1,npoints),reshape(Phi',1,L,npoints));
        B_npoints = pagemtimes(reshape(Phi',L,1,npoints),reshape(X',1,D,npoints));
        A = zeros(L,L,N);
        B = zeros(L,D,N);
        for i = 1:N
            A(:,:,i) = sum(A_npoints(:,:,ic == i), 3);
            B(:,:,i) = sum(B_npoints(:,:,ic == i), 3);
        end
    else
        A = pre_A;
        B = pre_B;
    end
    
    
    %% iteration
    loglkh = [];
    for ite = 1:max_iter
        % compute w/omega
        omega = zeros(L,D,K);
        for k = 1:K
            % omega_B_part = zeros(L,D);
            % omega_A_part = zeros(L,L);
            % for i = 1:N
            %     omega_A_part = omega_A_part + gamma(i,k) * A(:,:,i);
            %     omega_B_part = omega_B_part + gamma(i,k) * B(:,:,i);
            % end
            omega_A_part = reshape(reshape(A, '', size(A, 3)) * gamma(:, k), size(A, 1:2));
            omega_B_part = reshape(reshape(B, '', size(B, 3)) * gamma(:, k), size(B, 1:2));
            % omega_A_part = sum(A .* shiftdim(gamma(:,k), -2), 3);
            % omega_B_part = sum(B .* shiftdim(gamma(:,k), -2), 3);
            % gamma_reshaped = reshape(gamma(:,k),1,1,N);
            % omega_A_part = sum(A .* gamma_reshaped, 3);
            % omega_A_part = sum(bsxfun(@times, A, reshape(gamma(:,k),1,1,N)),3);
            omega_A_part = lambda * Lap' * Lap + omega_A_part;
            % omega_B_part = sum(bsxfun(@times, B, reshape(gamma(:,k),1,1,N)),3);
            % omega{k} = omega_A_part \ omega_B_part;
            omega(:,:,k) = inv(omega_A_part) * omega_B_part;
        end

        % compute f
        f = zeros(npoints,D,K);
        for k = 1:K
            f(:,:,k) = Phi * omega(:,:,k);
        end
        
        % compute sigma
        sigma_squares = zeros(D,K);
        for k = 1:K
            [yy,xx] = ndgrid(ic,1:D);
            diff = (X-f(:,:,k)).^2;
            tmp = accumarray([yy(:),xx(:)], diff(:));
            RHS = tmp' * gamma(:,k);
            prior_item = zeros(D,1);
            omega_k = omega(:,:,k);
            for j = 1:D
                prior_item(j,1) = norm(Lap * omega_k(:,j),2)^2;
            end
            RHS = RHS + lambda * prior_item;
            LHS = gamma(:,k)' * accumarray(ic, ones(npoints,1));
            sigma_square = RHS / LHS;
            sigma_squares(:,k) = sigma_square;
        end
        
        % compute gamma
        loglkh_subjects = zeros(N,K);
        for k = 1:K
            loglkh_k_points = logmvn(X,f(:,:,k),diag(sigma_squares(:,k)));
            loglkh_k_subjects = accumarray(ic,loglkh_k_points);
            loglkh_subjects(:,k) = loglkh_k_subjects;
        end
        loglkh_subjects = bsxfun(@plus, loglkh_subjects, -max(loglkh_subjects, [], 2));
        gamma = exp(loglkh_subjects) .* repmat(pis, [N,1]);
        % gamma = gamma ./ repmat(sum(gamma,2), 1, K);
        gamma = bsxfun(@rdivide, gamma, sum(gamma, 2));
        
        % compute pi
        pis = sum(gamma,1) / N;

        % compute loglikelihood
        loglkh_subjects = zeros(N,K);
        for k = 1:K
            loglkh_k_points = logmvn(X,f(:,:,k),diag(sigma_squares(:,k)));
            loglkh_k_subjects = accumarray(ic,loglkh_k_points);
            loglkh_subjects(:,k) = loglkh_k_subjects;
        end
        loglkh_mixture = calc_log_mixture(loglkh_subjects,pis);
        loglkh = [loglkh, sum(loglkh_mixture)];
    end

    diff = repmat(C', [1, L]) - repmat(C, [L, 1]);
    Phi = exp(-diff.^2 / (2*sigma_gaussian^2)) / (sqrt(2*pi)*sigma_gaussian);

    f = zeros(L,D,K);
    for k = 1:K
        f(:,:,k) = Phi * omega(:,:,k);
    end


    % model output
    model.pi = pis;
    model.W = omega(1+C_extension_size:end-C_extension_size,:,:);
    model.sigma2 = sigma_squares;
    model.f = f(1+C_extension_size:end-C_extension_size,:,:);
    model.C = C(1,1+C_extension_size:end-C_extension_size);
    model.loglik = loglkh(end);

    [~,subtype] = max(gamma,[],2);
    
    subtype = repelem(subtype, Ts, 1);
    gamma = repelem(gamma, Ts, 1);

    extra.loglik_list = loglkh;
    extra.gamma = gamma;
    
end
