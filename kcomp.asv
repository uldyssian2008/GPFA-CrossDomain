function barK = kcomp(scale,p,T)

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