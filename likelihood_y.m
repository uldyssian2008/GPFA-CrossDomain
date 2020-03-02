function y = loglih(C, p, timescale, noise, offset, Y, sigma_f, sigma_n)
%
% Parameter list:
% timescale: R^(p), scale of kernels
% p: dimension of latent space
% C: R^(q,q), basis vectors
% noise: R^(q), neuron noise
% offset: R^(q), affine offset
% Y: R^(q,T), spike observation
% sigma_f, sigma_n: R^(p) 


T = size(Y);
q = T(1);
T = T(2);
BigC = kron(eye(T),C);
K = {};
for i = 1:T
    for j = 1:T
        v = [];
        if i == j
            for k = 1:p
                v = [v,sigma_f(k)*exp(-(i-j)^2/2/timescale(k)) + sigma_n(k)];
                K{i,j} = diag(v);
            end
        else
            for k = 1:p
                v = [v,sigma_f(k)*exp(-(i-j)^2/2/timescale(k))];
                K{i,j} = diag(v);
            end
        end
    end
end
BigK = [];
for i = 1:T
    Z = [];
    for j = 1:T
        Z = [Z,K{i,j}];
    end
    BigK = [BigK;Z];
end
Bigd = kron((zeros(T,1) + 1),offset);
BigR = kron(eye(T),diag(noise));
Sigma = BigC*BigK*BigC' + BigR;
ybar = reshape(Y,[q*T,1]);
y = (ybar - Bigd)'*Sigma^(-1)*(ybar - Bigd) - log(det(Sigma));







