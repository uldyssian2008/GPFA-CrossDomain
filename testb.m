b = 0.95;

for tt = 1:10
    zz = [];
    b = 0.9 + 0.01 * (tt - 1);
    for i = 1:100
        zz(i) = powerB(b,i);
    end
    plot(zz)
    hold on
end
zz = [];
    b = 0.5;
    for i = 1:100
        zz(i) = powerB(b,i);
    end
    plot(zz)
