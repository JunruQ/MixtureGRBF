function [outputArg1,outputArg2] = analyze_adni_fsx_hippocampus_deconv(inputArg1,inputArg2)
%ANALYZE_ADNI_FSX_HIPPOCAMPUS_DECONV Summary of this function goes here
%   Detailed explanation goes here
load('ADNI41_fsx_hippocampus_deconv_demo/all_data.mat');
[s,f] = ftr_base(train_data.vols(pre_subtype==1,11), 1001, sigma(11,1));
figure, plot(s,f);
end

