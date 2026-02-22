function plot_traj_and_data(X,f,C,stage,subtype)
    for k = 1:max(subtype)
        f_k = f(:,:,k);
        X_k = X(subtype == k, :);
        stage_k = stage(subtype == k, :);
        figure;
        for i = 1:9     
            subplot(3,3,i);
            scatter(stage_k + rand(size(stage_k,1),size(stage_k,2))-0.5, X_k(:,4 * i - 3),'.');
            hold on;
            plot(C,f_k(:,4 * i - 3),'LineWidth',2);
            xlim([min(C) max(C)])
            hold off;
        end
    end
end