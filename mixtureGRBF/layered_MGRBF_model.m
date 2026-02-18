function [model,subtype,extra] = layered_MGRBF_model(dat,PTID,stage,nsubtype,options)

pre_subtype = layered_init_wof(dat,PTID,stage,nsubtype,options);

% methods
[model,subtype,extra]= mixtureGRBF(dat,PTID,stage,nsubtype,pre_subtype,options);
