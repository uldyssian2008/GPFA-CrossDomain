function y = kernelEva(t1,t2,scale)

if t1 == t2
    y = (1 - 10^(-3)) * exp(-(t1 - t2)^2/2/scale^2) + 10^(-3);
else
    y = (1 - 10^(-3)) * exp(-(t1 - t2)^2/2/scale^2);
end

%y = exp(-(t1 - t2)^2/2/scale^2);