function [outputArg1,outputArg2] = analyze_scz_subtypes_stages(inputArg1,inputArg2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

close all

demographics_filename = 'input/scz data/subject_info.xlsx';
demo = readtable(demographics_filename,'VariableNamingRule','preserve');

scz_result_filename = 'output/SCZ17_FTR_kmeans_stagsel=1/2_subtypes/subtype_stage.csv';
scz_result = readtable(scz_result_filename,'VariableNamingRule','preserve');

result_demo = join(scz_result, demo, 'LeftKeys', 'PTID', 'RightKeys', 'RID');
subtype = result_demo.subtype;
stage = result_demo.stage;

N = length(subtype);
K = length(unique(subtype));

group_inds = false(N, K+1);
for k = 0:K
    if k == 0
        group_inds(:,k+1) = (stage == 0);
    else
        group_inds(:,k+1) = (subtype == k & stage > 0);
    end
end

if K == 2
    group_names = {'Stage 0', 'Subtype 1', 'Subtype 2'};
elseif K == 3
    group_names = {'Stage 0', 'Subtype 1', 'Subtype 2', 'Subtype 3'};
end

compare_vars = {'age','PANSS_P','PANSS_N','PANSS_G','PANSS_T'};


figure;
for j = 1:length(compare_vars)
    data_j = result_demo.(compare_vars{j});
    
    ys = {};
    xs = {};
    
    subplot(2,3,j); 
    for k = 1:size(group_inds,2)
        data_jk = data_j(group_inds(:,k));
        x = k * ones(length(data_jk), 1);
        
        ys{k} = data_jk;
        xs{k} = x;
    end
        
    ylabel_name = compare_vars{j};
    boxplot_w_dots_p_value(xs, ys, group_names, ylabel_name);
end

duration = result_demo.('Duration(y)');
figure, scatter(stage, duration, 5);
xlabel('Stage');
ylabel('Duration (year)');
X = [stage, duration];
X(isnan(X(:,2)),:) = [];
[rho,pval] = corrcoef(X);
legend({sprintf('r = %.2f\nP = %.4f', rho(1,2), pval(1,2))}, 'Interpreter', 'none');

end

function boxplot_w_dots_p_value(xs, ys, xlabel_names, ylabel_name)
hold on

K1 = length(ys);

C = nchoosek((1:K1), 2);
for i = 1:size(C,1)
    y1 = ys{C(i,1)};
    y2 = ys{C(i,2)};
    [h,p] = ttest2(y1,y2,'Vartype','unequal');
    P(i) = p;
end

xlabel_names = [{''}, xlabel_names, {''}];
for k = 1:K1
%         scatter(x, data_jk, 5, 'filled', 'MarkerFaceAlpha',0.6,'jitter', ...
%             'on', 'jitterAmount', 0.15);
    swarmchart(xs{k}, ys{k}, 5, [0.5,0.5,0.5], 'filled');
    boxchart(xs{k}, ys{k});    
end

ylabel(ylabel_name, 'Interpreter', 'none');
set(gca,'XTick',(0:K1+1),'XTickLabel',xlabel_names);

% display the p values as * and **
thresh = 0.05;
yt = get(gca,'YTick');
axis([xlim, 0, ceil(max(yt)*1.3)]);
for i = 1:size(C,1)
    if P(i) < thresh
        line_x = C(i,:);
        line_x(1) = line_x(1) + 0.02;
        line_x(2) = line_x(2) - 0.02;
        
        line_y = [1,1]*max(yt)*(1 + abs(C(i,1)-C(i,2))*0.1);
        
        plot(line_x, line_y, '-k');
        plot([line_x(1),line_x(1)],[line_y(1),line_y(1)-0.02*abs(yt(1)-yt(end))], '-k');
        plot([line_x(2),line_x(2)],[line_y(2),line_y(2)-0.02*abs(yt(1)-yt(end))], '-k');
         
        text(mean(C(i,:)), line_y(1) + max(yt)*0.05, sprintf('P = %.3f', P(i)), ...
            'horizontalAlignment', 'center'); 
    end
end

end
