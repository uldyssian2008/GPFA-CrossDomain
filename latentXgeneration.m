p = 10;


t1 = cputime;
np = 0.1; 
SigmaT = {};
SamScale = ones(p,1);
for tt = 1:p
    for i = 1:samNum
        for j = 1:samNum
            SigmaT{tt}(i,j) = kernelEva(0.1*i + 0.9,0.1*j + 0.9,SamScale(tt));
        end
    end
end

X = {};
for i = 1:p
    X{i} = mvnrnd(zeros(samNum,1)',SigmaT{i} + 0.01^2 * eye(samNum),TrialNum);
end