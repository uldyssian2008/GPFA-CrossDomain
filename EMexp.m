function [Mean,Cov,Cov2] = EMexp(C,scale,R,d,Y)
q = size(Y,1);
T = size(Y,2);
p = size(C,2);
barC = kron(eye(T),C);
bary = reshape(Y,[q*T,1]);
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
Mean = barK*barC'*(barC*barK*barC' + barR)^(-1)*(bary - bard);
CovT = barK - barK*barC'*(barC*barK*barC' + barR)^(-1)*barC*barK + Mean*Mean';
Mean = reshape(Mean,[p,T]);
for i = 1:T
    Cov{i} = CovT((i-1)*p + 1:i*p,(i-1)*p + 1:i*p);
end
for i = 1:p
   ZZ = [];
   for l1 = 1:T
       for l2 = 1:T
            ZZ(l1,l2) = CovT((l1-1) * p + i,(l2-1) * p + i);
       end
   end
   Cov2{i} = ZZ;
end
a = 2;