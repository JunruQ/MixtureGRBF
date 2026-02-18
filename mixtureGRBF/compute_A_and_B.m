function [A,B] = compute_A_and_B(X, PTID, stage, options)
    [npoints, D] = size(X);
    [subjects,~,ic] = unique(PTID,'stable');
    N = size(subjects,1);
    sigma_gaussian = parse_param(options,'sigma_gaussian',1);
    gaussian_interval = parse_param(options, 'gaussian_interval', 1);
    C_extension_size = parse_param(options,'C_extension_size',2);
    C = options.C;
    
    if isempty(C)
        L = (max(stage) - min(stage)) / gaussian_interval + 1;
        C = linspace(min(stage),max(stage),L);
    else
        L = (max(C) - min(C)) / gaussian_interval + 1;
    end

    left_extention = ones(1,C_extension_size) * min(C) - (C_extension_size:-1:1) * gaussian_interval;
    right_extention = ones(1,C_extension_size) * max(C) + (1:C_extension_size) * gaussian_interval;
    C = [left_extention, C, right_extention];
    L = L + C_extension_size * 2;
    
    diff = repmat(stage, [1, L]) - repmat(C, [npoints, 1]);
    Phi = exp(-diff.^2 / (2*sigma_gaussian^2)) / (sqrt(2*pi)*sigma_gaussian);
    A_npoints = pagemtimes(reshape(Phi',L,1,npoints),reshape(Phi',1,L,npoints));
    B_npoints = pagemtimes(reshape(Phi',L,1,npoints),reshape(X',1,D,npoints));
    A = zeros(L,L,N);
    B = zeros(L,D,N);
    for i = 1:N
        A(:,:,i) = sum(A_npoints(:,:,ic == i), 3);
        B(:,:,i) = sum(B_npoints(:,:,ic == i), 3);
    end

end