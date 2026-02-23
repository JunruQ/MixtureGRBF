raw_synth_path = 'output/synth_data/synth_data.csv';

output_synth_dir = 'input/synth';

if ~exist(output_synth_dir, "dir")
    mkdir(output_synth_dir);
end

t = readmatrix(raw_synth_path);

% 创建列名
nCols = size(t, 2);  % 302
varNames = [{'RID', 'stage'}, arrayfun(@(i) sprintf('biom_%d', i-2), 3:nCols, 'UniformOutput', false)];

% 第3列到最后做zscore
t(:, 3:end) = zscore(t(:, 3:end));

% 转为table并加上列名
t = array2table(t, 'VariableNames', varNames);
t.RID = (1:size(t, 1))';

writetable(t, [output_synth_dir, '/synth_data_z.csv']);