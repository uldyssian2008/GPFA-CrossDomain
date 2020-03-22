function y = predictGP(neuron,Y,Bj,Bjc,Sigma,bard,q,T)

% parameter setup
bary = reshape(Y,[q*T,1]);    
% SigmaP = zeros(p,p*T);
% 
% for i = 1:p
%     for j = 1:p*T
%         if mod(i,p) == mod(j,p)
%             SigmaP(i,j) = kernelEva(time,TimeTable(ceil(j/p)),scale(i));
%         end
%     end
% end
    
y = Bj{neuron} * bard + Bj{neuron} * Sigma * Bjc{neuron}' * (Bjc{neuron} * Sigma * Bjc{neuron}')^(-1) * (Bjc{neuron} * bary - Bjc{neuron} * bard);


