function y = predictGP(time,neuron,Y,barC,Bjc,scale,Sigma,bard,p,q,T,d)

% parameter setup
bary = reshape(Y,[q*T,1]);    
SigmaP = zeros(p,p*T);

for i = 1:p
    for j = 1:p*T
        if mod(i,p) == mod(j,p)
            if time == ceil(j/p)
                SigmaP(i,j) = (1 - 10^(-3)) * exp(-(time - ceil(j/p))^2/2/scale(i)^2) + 10^(-3);
            else
                SigmaP(i,j) = (1 - 10^(-3)) * exp(-(time - ceil(j/p))^2/2/scale(i)^2);
            end
        end
    end
end
    
y = d(neuron) + barC(neuron,1:p) * SigmaP * barC' * Bjc{neuron}' * (Bjc{neuron} * Sigma * Bjc{neuron}')^(-1) * (Bjc{neuron} * bary - Bjc{neuron} * bard);
a = 1;

