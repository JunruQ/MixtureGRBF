function [model,subtype,extra] = MGRBF_model(dat,PTID,stage,nsubtype,options)

verify_PTID_order = 1;
for i = 2:length(PTID)
    if ~(PTID(i) == PTID(i-1) || PTID(i) > PTID(i-1))
        verify_PTID_order = 0;
    end
end
assert(verify_PTID_order, ['PTID is not monotonically increasing in FTR_model. ',...
    'Possible error may occur in cal_loglik.m and MCEM_subtype.m']);

[pre_A,pre_B] = compute_A_and_B(dat, PTID, stage, options);
options.A = pre_A;
options.B = pre_B;

[pre_subtype,pre_sigma] = init_wof(dat,PTID,stage,nsubtype,options);
options.max_iter = 1;

% methods
[model,subtype,extra]= mixtureGRBF(dat,PTID,stage,nsubtype,pre_subtype,options);


end    

