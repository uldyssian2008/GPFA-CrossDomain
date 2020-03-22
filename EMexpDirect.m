function [Mean,Cov,Cov2,Cov3] = EMexpDirect(C,scale,R,d,baryTrain,p,T,TrainNum,TimeTable)

barC = kron(eye(T),C);
barR = kron(eye(T),R);
bard = kron(ones(T,1),d);
barK = [];
for i = 1:T
    tempK = [];
    for j = 1:T
        v = [];
        for k = 1:p
            v = [v;kernelEva(TimeTable(i),TimeTable(j),scale(k))];
        end
        tempK = [tempK,diag(v)];
    end
    barK = [barK;tempK];
end
Mean = {};
for i = 1:TrainNum
    Mean{i} = barK*barC'*(barC*barK*barC' + barR)^(-1)*(baryTrain(:,i) - bard);
end
Cov3 = {};
for i = 1:TrainNum
    Cov3{i} = barK - barK*barC'*(barC*barK*barC' + barR)^(-1)*barC*barK + Mean{i}*Mean{i}';
end
for i = 1:TrainNum
    Mean{i} = reshape(Mean{i},[p,T]);
end
Cov = {};
for j = 1:TrainNum
   for i = 1:T
        Cov{i,j} = Cov3{j}((i-1)*p + 1:i*p,(i-1)*p + 1:i*p);
   end
end
Cov2 = {};
for j = 1:TrainNum
    for i = 1:p
       ZZ = [];
       for l1 = 1:T
           for l2 = 1:T
                ZZ(l1,l2) = Cov3{j}((l1-1) * p + i,(l2-1) * p + i);
           end
       end
       Cov2{i,j} = ZZ;
    end
end

