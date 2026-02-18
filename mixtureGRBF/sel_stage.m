function [dat_sel,PTID_sel,stage_sel] = sel_stage(dat,PTID,options)

nsubtype = 1;

[mdl,subtype,stage,extra] = FTR_model(dat,PTID,nsubtype,options);

dat_sel = dat(stage>0,:);
PTID_sel = PTID(stage>0,:);
stage_sel = stage(stage>0);

end
