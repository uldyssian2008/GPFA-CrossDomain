function [C,scale,d] = EMmaxNoR(Mean,Cov,Cov2,Y)

q = size(Y,1);
T = size(Y,2);
p = size(Mean,1);

Z1 = 0;
for i = 1:T
    Z1 = Z1 + Y(:,i)*[Mean(:,i)',1];
end
Z2 = 0;
for i = 1:T
    Z2 = Z2 + [Cov{i},Mean(:,i);Mean(:,i)',1];
end
Z2 = Z2^(-1);
ZZ = Z1 * Z2;
C = ZZ(:,1:end - 1);
d = ZZ(:,end);


% gradient ascent on scale
scaleExp = zeros(p,1);
OptIter = 100;
lr = 10^(-3);
for t = 1:OptIter
    for i = 1:p
        Ki = [];
        for l1 = 1:T
            for l2 = 1:T
                if l1 == l2
                    Ki(l1,l2) = (1 - 10^(-3))*exp(-(l1-l2)^2/2/exp(2*scaleExp(i))) + 10^(-3);
                else
                    Ki(l1,l2) = (1 - 10^(-3))*exp(-(l1-l2)^2/2/exp(2*scaleExp(i)));
                end
            end
        end
        Ki1 = 1/2 * (-Ki^(-1) + -Ki^(-1) * Cov2{i} * Ki^(-1));
        Ki2 = [];
        for l1 = 1:T
            for l2 = 1:T
                Ki2(l1,l2) = (1 - 10^(-3)) * (l1 - l2)^2 / exp(3 * scaleExp(i)) * exp(-(l1 - l2)^2 / 2 / exp(2 * scaleExp(i))) * exp(scaleExp(i));
            end
        end
        Der = trace(Ki1' * Ki2);
        scaleExp(i) = scaleExp(i) + lr * Der;
    end
end
scale = exp(scaleExp);
disp([' kernel width = ' num2str(scale') ';']);


