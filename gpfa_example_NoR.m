load('mat_sample/sample_dat');
Y = dat(1);
Y = Y.spikes;
T = size(Y,2);
q = size(Y,1);

% latent dimension
p = 10;
% noise power
np = 0.05; 
% bin spike coumt
binwidth = 20;
Ybar = [];
for i = 1:T/binwidth
    z = sum(Y(:,(i-1) * binwidth + 1:i * binwidth ),2);
    Ybar = [Ybar,z];
end
T = T/binwidth;
Y = Ybar;
bary = reshape(Y,[q*T,1]);

% optimize hyperparameters
optTime = 200; % number of EM iteration
C = randn(q,p); scale = abs(randn(p,1)) + 10^(-1); R = diag(0.01 * rand(q,1)); d = randn(q,1); % parameter initialization
barR = kron(eye(T),R);

for t = 1:optTime   
    [Mean,Cov,Cov2] = EMexp(C,scale,R,d,Y);
    [C,scale,d] = EMmaxNoR(Mean,Cov,Cov2,Y);
    
    % calculate loglikelihood
    bard = kron(ones(T,1),d);
    barC = kron(eye(T),C);
    barK = [];
    for i = 1:T
        tempK = [];
        for j = 1:T
            v = [];
            if i == j
                for k = 1:p
                    v = [v;(1-10^(-3))*exp(-(i-j)^2/2/scale(k)^2)+10^(-3)];
                end
                tempK = [tempK,diag(v)];
            else
                for k = 1:p
                    v = [v;(1-10^(-3))*exp(-(i-j)^2/2/scale(k)^2)];
                end
                tempK = [tempK,diag(v)];
            end
        end
        barK = [barK;tempK];
    end
    %covK = barC * barK * barC' + barR;
    covK = barC * barK * barC' + barR + np^2 * eye(q*T);
    loglikelihood = -1/2 * (bary - bard)' * covK^(-1) * (bary - bard) - 1/2 * (q * T * log(2*pi) + log(det(covK)));
    disp(['Iteration ' num2str(t) ': logLikelihood = ' num2str(loglikelihood) ';']);
end

% fcn = @(x) -loglih(reshape(x(1:q*p),[q,p]), p, reshape(x(p*q + 1:p*q + p),[p,1]), reshape(x(p*q + p + 1:p*q + p + q),[q,1]), reshape(x(p*q + p + q + 1:p*q + p + 2*q),[q,1]), Y, reshape(x(p*q + p + 2*q + 1:p*q + p + 2*q + p),[p,1]), reshape(x(p*q + p + 2*q + p + 1:p*q + p + 2*q + 2*p),[p,1]));
% init = [randn(1,q*p),randn(1,p),randn(1,q),randn(1,q),randn(1,p),randn(1,p)];
% [parOpt,vOpt] = patternsearch(fcn,init,[],[],[],[]);

