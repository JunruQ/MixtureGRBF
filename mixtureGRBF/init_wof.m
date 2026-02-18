function [pre_subtype,pre_sigma] = init_wof(dat,PTID,stage,nsubtype,options)

% options.filter = false;
max_ep = options.max_ep;

% if set a single trajectory, no need to randomly initialize multiple times
if nsubtype == 1
    max_ep = 1;
end

subtype = zeros(size(dat,1), max_ep);
loglik_all = zeros(options.max_iter, max_ep);

if ~options.parfor
    parfor ep = 1:max_ep
        try 

            
            [model,subtype(:,ep),extra]= ...
                mixtureGRBF(dat,PTID,stage,nsubtype,[],options);
            sgm(:,:,ep) = model.sigma2; 
            loglik_mat(ep) = model.loglik;

        catch me
            fprintf('WARNING: exception in init_wof at run %d/%d with %d subtypes.\n', ...
                ep, max_ep, nsubtype);
            loglik_mat(ep) = -Inf;

            msgText = getReport(me);
            disp(msgText);
        end
    end
else
    for ep = 1:max_ep
        try 

            
            [model,subtype(:,ep),extra]= ...
                mixtureGRBF(dat,PTID,stage,nsubtype,[],options);
            sgm(:,:,ep) = model.sigma2; 
            loglik_mat(ep) = model.loglik;

        catch me
            fprintf('WARNING: exception in init_wof at run %d/%d with %d subtypes.\n', ...
                ep, max_ep, nsubtype);
            loglik_mat(ep) = -Inf;

            msgText = getReport(me);
            disp(msgText);
        end
    end
end

[~,bes_ep] = max(loglik_mat);
pre_subtype = subtype(:,bes_ep);
pre_sigma = sgm(:,:,bes_ep);

end

