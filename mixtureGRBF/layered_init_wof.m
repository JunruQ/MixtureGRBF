function pre_subtype = layered_init_wof(dat,PTID,stage,nsubtype,options)

% 不输出pre_sigma，因为要节省计算时间，第三层可不在这层逻辑中计算
% 固定的初始化次数，因为是对三层100次的数据进行初始化，泛化性较弱
% 选择策略
options = insert_param_when_absent(options, 'parfor_init', true);
options = insert_param_when_absent(options, 'num_layer', '2_layer');
options = insert_param_when_absent(options, 'select_from_multi_inits', 'loglikelihood');
options = insert_param_when_absent(options, 'select_component', 'pca');

max_ep = 100;

num_component_layer_3 = size(dat,2);
num_component_layer_1 = ceil(num_component_layer_3/100);
num_component_layer_2 = ceil(num_component_layer_3/10);

dat_layer_1 = sel_component(dat,num_component_layer_1,options);
dat_layer_2 = sel_component(dat,num_component_layer_2,options);

subtype = zeros(size(dat,1), max_ep);
loglik_all = zeros(options.max_iter, max_ep);

if nsubtype == 1
    [~,pre_subtype,~]= ...
        mixtureGRBF(dat_layer_1,PTID,stage,nsubtype,[],options);

else
    switch options.num_layer
        case '2_layer'
            dat_1 = dat_layer_2;
        case '3_layer'
            dat_1 = dat_layer_1;
    end
    % 第一层
    [pre_A,pre_B] = compute_A_and_B(dat_1, PTID, stage, options);
    options.A = pre_A;
    options.B = pre_B;
    if options.parfor_init
        parfor ep = 1:max_ep
            [model,subtype_layer_1(:,ep),~]= ...
                mixtureGRBF(dat_1,PTID,stage,nsubtype,[],options);
            loglik_mat_layer_1(ep) = model.loglik;
        end
    else
        for ep = 1:max_ep
            [model,subtype_layer_1(:,ep),~]= ...
                mixtureGRBF(dat_1,PTID,stage,nsubtype,[],options);
            loglik_mat_layer_1(ep) = model.loglik;
        end
    end
    % 分成10份后取最大值
    bes_ep_layer_1 = [];
    for l = 1:10
        bes_ep = sel_from_multi_inits(subtype_layer_1(:,10*l-9:10*l), loglik_mat_layer_1(10*l-9:10*l), options);
        bes_ep_layer_1 = [bes_ep_layer_1, bes_ep+(l-1)*10];
    end

    % 第二层
    switch options.num_layer
        case '2_layer'
            dat_2 = dat;
        case '3_layer'
            dat_2 = dat_layer_2;
    end
    presubtype_layer_2 = subtype_layer_1(:,bes_ep_layer_1);
    [pre_A,pre_B] = compute_A_and_B(dat_2, PTID, stage, options);
    options.A = pre_A;
    options.B = pre_B;
    if options.parfor_init
        parfor ep = 1:10
            [model,subtype_layer_2(:,ep),~]= ...
                mixtureGRBF(dat_2,PTID,stage,nsubtype,presubtype_layer_2(:,ep),options);
            loglik_mat_layer_2(ep) = model.loglik;
        end
    else
        for ep = 1:10
            [model,subtype_layer_2(:,ep),~]= ...
                mixtureGRBF(dat_2,PTID,stage,nsubtype,presubtype_layer_2(:,ep),options);
            loglik_mat_layer_2(ep) = model.loglik;
        end
    end

    bes_ep = sel_from_multi_inits(subtype_layer_2, loglik_mat_layer_2, options);

    % 第三层
    pre_subtype = subtype_layer_2(:,bes_ep);
end

end

