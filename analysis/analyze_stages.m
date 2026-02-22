function [outputArg1,outputArg2] = analyze_stages(inputArg1,inputArg2)
%ANALYZE_STAGES Summary of this function goes here
%   Detailed explanation goes here

data_names = {'ADNI_FSX_LM'};
method_names = {'FTR'};

nsubtype = 3;

results = load_data_result(data_names, method_names, nsubtype, 1);
joindata = results{1,1}.joindata;
joindata.Properties.VariableNames = strrep(joindata.Properties.VariableNames, '-', '_');
traj = results{1,1}.traj;
biomarker_names = [results{1,1}.biomarker_names];

stage = joindata.stage;
data = joindata;

% Plot stage
nsamp = length(data.diagnosis);
stag_CN = stage(data.diagnosis == 0);
stag_MCI = stage(data.diagnosis == 0.5);
stag_AD = stage(data.diagnosis == 1);
binrng = 0:0.05:1;
counts(1,:) = histcounts(stag_CN, binrng);
counts(2,:) = histcounts(stag_MCI, binrng);
counts(3,:) = histcounts(stag_AD, binrng);

figure();
h = bar(binrng(1:end-1),counts);
set(h(1),'FaceColor',[0,0,1]);
set(h(2),'FaceColor',[0,1,0]);
set(h(3),'FaceColor',[1,0,0]);
legend('CN','MCI','AD')
xlabel({'Stage'});
ylabel({'Count'});

% calculate AUC
pred_stage_sel = stage(data.diagnosis~=0.5);
true_label_sel = data.diagnosis(data.diagnosis~=0.5);
[X,Y,T,AUC] = perfcurve(true_label_sel,pred_stage_sel,1);

AUC

plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC curve')

figure;
data1 = data(~isnan(data.MMSE),:);
scatter(data1.stage, data1.MMSE);
xlabel('Stage');
ylabel('MMSE');
[r,p] = corrcoef(data1.stage, data1.MMSE);
r(1,2)

end

