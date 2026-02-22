function show_trajectory(traj,stage,bio_name)

[nbiom,num_int,nsubtype] = size(traj);

if nargin < 2
    bio_name = cell(1, nbiom);
    for j = 1:nbiom
        bio_name{j} = ['biomarker ',int2str(j)];
    end
end

% ord = zeros(nsubtype,nbiom);
s = stage;

% YZ: I changed the presentation.
[colors, linestyles] = get_color_linestyle(bio_name);
for k = 1:nsubtype
    % comp = sum(traj(:,:,k) < thres,2);
    % [~,ord(k,:)] = sort(comp);

    h = figure;

    legendText = get_legend_texts(bio_name);

    for j = 1:nbiom
        plot(s,traj(j,:,k),linestyles{j},'Color',colors(j,:),'LineWidth',1);
        hold on;
    end
    xlabel("Stage")
    ylabel("z-score")
    
    % plot([0,1],[thres,thres],':','Color','k','LineWidth',1);
    %lgd = legend(legendText,'Location', 'northwest');
    lgd = legend(legendText,'Location', 'eastoutside', 'Interpreter', 'none');
    hold off;
end

end

function [colors, linestyles] = get_color_linestyle(bio_name)
linestyles = repmat({'-','--',':','-.'}, 1, ceil(length(bio_name) / 4));
linestyles = linestyles(1:length(bio_name));
colors = turbo(length(bio_name));
colors = flipud(colors);
end

function legendText = get_legend_texts(bio_name)
for j = 1:length(bio_name)
    if isempty(bio_name)
        legendText{j} = sprintf(int2str(j-1));
    else
        % YZ: The second argument j in sprintf may not be needed
        legendText{j} = sprintf(char(bio_name(j)));
    end
end

end