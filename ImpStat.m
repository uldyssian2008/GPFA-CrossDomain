function [barC,barR,bard,barK,Bj,Bjc,Sigma] = ImpStat(C,scale,R,d,T)

p = size(C,2);
q = size(C,1);
barC = kron(eye(T),C);
barR = kron(eye(T),R);
bard = kron(ones(T,1),d);
barK = [];
for i = 1:T
    tempK = [];
    for j = 1:T
        v = [];
        for k = 1:p
            v = [v;kernelEva(i,j,scale(k))];
        end
        tempK = [tempK,diag(v)];
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
