function [barC,barR,bard,barK,Bj,Bjc,Sigma] = ImpStat(C,scale,R,d)

barC = kron(eye(T),C);
barR = kron(eye(T),R);
bard = kron(ones(T,1),d);
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


Bj = {};
Ide = eye(T*q);
for i = 1:q
    Btemp = [];
    for j = 1:T
        Btemp = [Btemp;Ide(:,i + (j-1) * q)'];
    end
    Bj{i} = Btemp;
end
Bjc = {};
for i = 1:q
    BcTemp = [];
    for j = 1:q
        if i ~= j
            BcTemp = [BcTemp;Bj{j}];
        end
    end
    Bjc{i} = BcTemp;
end

Sigma = barC * barK * barC' + barR;
