load('mat_sample/sample_dat');
Y = dat(1);
Y = Y.spikes;
T = size(Y,2);
q = size(Y,1);

% latent dimension
p = 10;
% bin spike coumt
binwidth = 20;
Ybar = [];
for i = 1:T/binwidth
    z = sum(Y(:,(i-1) * binwidth + 1:i * binwidth ),2);
    Ybar = [Ybar,z];
end
Y = Ybar;
% lb = [-Inf(1,q*p),zeros(1,p) + 10^(-3),zeros(1,q) + 10^(-3),-Inf(1,q),zeros(1,p) + 10^(-3),zeros(1,p) + 10^(-3)];
% fcn = @(x) -loglih(reshape(x(1:q*p),[q,p]), p, reshape(x(p*q + 1:p*q + p),[p,1]), reshape(x(p*q + p + 1:p*q + p + q),[q,1]), reshape(x(p*q + p + q + 1:p*q + p + 2*q),[q,1]), Y, reshape(x(p*q + p + 2*q + 1:p*q + p + 2*q + p),[p,1]), reshape(x(p*q + p + 2*q + p + 1:p*q + p + 2*q + 2*p),[p,1]));
% init = [randn(1,q*p),abs(randn(1,p)) + 10^(-3),abs(randn(1,q)) + 10^(-3),randn(1,q),abs(randn(1,p)) + 10^(-3),abs(randn(1,p)) + 10^(-3)];
% [parOpt,vOpt] = patternsearch(fcn,init,[],[],[],[],lb,[]);

fcn = @(x) -loglih(reshape(x(1:q*p),[q,p]), p, reshape(x(p*q + 1:p*q + p),[p,1]), reshape(x(p*q + p + 1:p*q + p + q),[q,1]), reshape(x(p*q + p + q + 1:p*q + p + 2*q),[q,1]), Y, reshape(x(p*q + p + 2*q + 1:p*q + p + 2*q + p),[p,1]), reshape(x(p*q + p + 2*q + p + 1:p*q + p + 2*q + 2*p),[p,1]));
init = [randn(1,q*p),randn(1,p),randn(1,q),randn(1,q),randn(1,p),randn(1,p)];
[parOpt,vOpt] = patternsearch(fcn,init,[],[],[],[]);