function y = kernelDer(t1,t2,scale)

y = (1 - 10^(-4)) * (t1 - t2)^2 / scale^3 * exp(-(t1 - t2)^2 / 2 / scale^2);

%y = (t1 - t2)^2 / scale^3 * exp(-(t1 - t2)^2 / 2 / scale^2);