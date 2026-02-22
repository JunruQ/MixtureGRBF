function [outputArg1,outputArg2] = show_confusion_matrix(conf_matrix)
%SHOW_CONFUSION_MAT Summary of this function goes here
%   Detailed explanation goes here
 % Display confusion matrix
disp('Confusion Matrix:');
disp(conf_matrix);

perc_consistency = sum(diag(conf_matrix)) / sum(sum(conf_matrix));

% Example subtypes 
num_subtypes = size(conf_matrix, 1);
train_subtypes = {'subtype1', 'subtype2', 'subtype3'};
test_subtypes = {'subtype1', 'subtype2', 'subtype3'};

% Plotting the confusion matrix
figure;
imagesc(conf_matrix);
colorbar;

% Adjusting axis labels and ticks
set(gca, 'XTick', 1:length(train_subtypes), 'XTickLabel', train_subtypes, 'YTick', 1:length(test_subtypes), 'YTickLabel', test_subtypes);
xlabel('Test Subtypes');
ylabel('Train Subtypes');
title(['Confusion Matrix: ',num2str(perc_consistency*100,3),'% consistency']);

% Displaying values in the cells (optional)
textStrings = num2str(conf_matrix(:), '%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:length(train_subtypes), 1:length(test_subtypes));
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
midValue = mean(get(gca, 'CLim'));
textColors = repmat(reshape(eye(num_subtypes),[],1), 1, 3);
%repmat(conf_matrix(:) > midValue, 1, 3);
set(hStrings, {'Color'}, num2cell(~textColors,2));

% Adjusting the color scale (optional)
% colormap('gray'); % Change the colormap as needed



end

