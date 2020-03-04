function [traj,preError] = testGP(Ytest,Bj,Bjc,bard,Sigma,barR)

% parameter setup
q = size(Ytest,1);
T = size(Ytest,2);
bary = reshape(Ytest,[q*T,1]);

traj = [];
for i = 1:q
    %traj = [traj , Bj{i} * bard + (Bj{i} * Sigma * Bjc{i}') * (Bjc{i} * Sigma * Bjc{i}')^(-1) * (Bjc{i} * bary - Bjc{i} * bard)];
    traj = [traj , Bj{i} * bard + (Bj{i} * (Sigma - barR) * Bjc{i}') * (Bjc{i} * Sigma * Bjc{i}')^(-1) * (Bjc{i} * bary - Bjc{i} * bard)];
end
    
preError = 0;
for j = 1:q
    preError = preError + norm(traj(:,j) - Bj{j} * bary)^2;
end

