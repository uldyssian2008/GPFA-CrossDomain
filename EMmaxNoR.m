function [C,scale,d] = EMmaxNoR(Mean,Cov,Cov2,YTrain,p,T)

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


% gradient ascent on scale
scaleExp = zeros(p,1);
OptIter = 100;
lr = 10^(-4);
for t = 1:OptIter
    for i = 1:p
        Ki = [];
        Der = 0;
        for l1 = 1:T
            for l2 = 1:T
                Ki(l1,l2) = kernelEva(l1,l2,exp(scaleExp(i)));
            end
        end
        Ki2 = [];
        for l1 = 1:T
            for l2 = 1:T
                Ki2(l1,l2) = kernelDer(l1,l2,exp(scaleExp(i))) * exp(scaleExp(i)); % change of variable: scale = exp(scaleExp)
            end
        end 
        for trial = 1:TrainNum
            Ki1 = 1/2 * (-Ki^(-1) + -Ki^(-1) * Cov2{i,trial} * Ki^(-1));
            Der = Der + trace(Ki1' * Ki2);
        end
        scaleExp(i) = scaleExp(i) + lr * Der;
    end
end
scale = exp(scaleExp);
disp([' kernel width = ' num2str(scale') ';']);


