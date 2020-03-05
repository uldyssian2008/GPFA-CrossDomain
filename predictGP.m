function y = predictGP(time,neuron,Y,barC,Bjc,scale,Sigma,bard,p,q,T,d)

% parameter setup
bary = reshape(Y,[q*T,1]);    
SigmaP = zeros(p,p*T);

for i = 1:p
    for j = 1:p*T
        if mod(i,p) == mod(j,p)
            SigmaP(i,j) = kernelEva(time,ceil(j/p),scale(i));
        end
    end
end
    
y = d(neuron) + barC(neuron,1:p) * SigmaP * barC' * Bjc{neuron}' * (Bjc{neuron} * Sigma * Bjc{neuron}')^(-1) * (Bjc{neuron} * bary - Bjc{neuron} * bard);
a = 1;

