function [C,d,R] = EMmaxDirectNOB(Mean,Cov,Cov3,YTrain,T)

TrainNum = size(YTrain,2);
Z1 = 0;
for l = 1:TrainNum
    for i = 1:T
        Z1 = Z1 + YTrain{l}(:,i)*[Mean{l}(:,i)',1];
    end
end
Z2 = 0;
for l = 1:TrainNum
    for i = 1:T
        Z2 = Z2 + [Cov{i,l},Mean{l}(:,i);Mean{l}(:,i)',1];
    end
end
ZZ = Z1/Z2;
C = ZZ(:,1:end - 1);
d = ZZ(:,end);

Z3 = 0;
for l = 1:TrainNum
    for i = 1:T
        Z3 = Z3 + (YTrain{l}(:,i) - d) * (YTrain{l}(:,i) - d)' - (YTrain{l}(:,i) - d) * Mean{l}(:,i)'*C';
    end
end
R = 1 / T / TrainNum * diag(diag(Z3));



% optimization of kernel width (DIRECT algorithm)

Covar = 0;
for i = 1:TrainNum
    Covar = Covar + Cov3{i};
end
