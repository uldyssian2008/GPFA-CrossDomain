function [C,scale,d,R] = EMmaxDirect(Mean,Cov,Cov2,Cov3,YTrain,p,T)

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
        Z3 = Z3 + (YTrain{l}(:,i) - d) * (YTrain{l}(:,i) - d)' - C * Mean{l}(:,i) * (YTrain{l}(:,i) - d)' - (YTrain{l}(:,i) - d) * Mean{l}(:,i)'*C' + C*Cov{i,l}*C';
        %Z3 = Z3 + (Y(:,i) - d) * (Y(:,i) - d)' - (Y(:,i) - d) * Mean(:,i)'*C';
    end
end
R = 1 / T / TrainNum * diag(diag(Z3));



% optimization of kernel width

Covar = 0;
for i = 1:TrainNum
    Covar = Covar + Cov3{i};
end

scale = abs(randn(p,1)) + 10^(-1);
%fun = @(x) trace(kcomp(x,p,T) \ Covar) + TrainNum * log(det(kcomp(x,p,T)));
fun = @(x) log(exp(trace(kcomp(x,p,T) \ Covar) * det(kcomp(x,p,T))^TrainNum ));
scale = patternsearch(fun,scale,[],[],[],[],zeros(1,p),[]);


disp([' kernel width = ' num2str(scale') ';']);