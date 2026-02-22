function [h, p] = my_logrank(x1, x2, c1, c2)
% logrank performs the log-rank test for comparing two survival curves
%
% x1, x2: survival times
% c1, c2: censoring indicators (1 if the event happened, 0 if censored)
%
% h: hypothesis test result (0 = do not reject null hypothesis, 1 = reject null hypothesis)
% p: p-value of the test

% Combine the data
x = [x1; x2];
c = [c1; c2];

% Sort the combined data
[x, idx] = sort(x);
c = c(idx);

% Determine the number of events and censored
n = length(x);
n1 = length(x1);
n2 = length(x2);

% Initialize counts
O1 = 0; % Observed events in group 1
E1 = 0; % Expected events in group 1

% Calculate observed and expected events
for i = 1:n
    if c(i) == 1
        % Event occurred
        if i <= n1
            O1 = O1 + 1;
        end

        % Risk set
        risk_set = sum(c == 1);
        E1 = E1 + (n1 / risk_set);
    end
end

% Calculate variance
var_E1 = (n1 * (n2 / (n1 + n2)) * ((n1 + n2 - 1) / (n1 + n2 - 2))) / (n1 + n2);

% Compute the log-rank statistic
logrank_stat = (O1 - E1) / sqrt(var_E1);

% Compute p-value using the normal distribution
p = 2 * (1 - normcdf(abs(logrank_stat), 0, 1));

% Determine the hypothesis test result
alpha = 0.05; % Significance level
h = p < alpha;
end

