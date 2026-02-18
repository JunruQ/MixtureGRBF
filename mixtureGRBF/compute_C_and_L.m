function [C, L] = compute_C_and_L(stage,options)
    
    is_mgrbf = parse_param(options,'is_mgrbf',true);

    if is_mgrbf
        gaussian_interval = parse_param(options,'gaussian_interval',1);
        L = (ceil(max(stage)) - floor(min(stage)))/gaussian_interval + 1;
        C = linspace(floor(min(stage)),ceil(max(stage)),L);
    else
        gaussian_interval = parse_param(options,'gaussian_interval',1);
        L = (max(stage) - min(stage))/gaussian_interval + 1;
        C = linspace(min(stage),max(stage),L);
    end

end