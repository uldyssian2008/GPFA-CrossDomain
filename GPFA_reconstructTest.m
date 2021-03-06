%%%%%%%%%%%% GPFA Implementation %%%%%%%%%%%%%%%

rng(0,'twister'); % For reproducibility

%%% Caution: change kernelEva & kernelDer simultaneously %%%

%% sample from GP/Test (for Testing Purpose only)

t1 = cputime;
totalNum = 40; % number of GP approximation points
TrialNum = 10;
sampleNum = 40; % number of sample
gap = totalNum/sampleNum;
TimeTable = zeros(sampleNum,1);
for i = 1:sampleNum
    TimeTable(i) = gap * (i - 1) + 1;
end


q = 10;
p = 2;
np = 0.1;

SigmaT = {};
GPScale = 4 * ones(p,1);
for tt = 1:p
    for i = 1:totalNum
        for j = 1:totalNum
            SigmaT{tt}(i,j) = kernelEva(i,j,GPScale(tt));
        end
    end
end
trueC = 10 * randn(q,p);

X = {}; % X{i}(j,k):  i,j,k = index of trial, latent, time % whole latent whole process
for i = 1:TrialNum
    X{i} = [];
    for j = 1:p
        X{i} = [X{i};mvnrnd(zeros(totalNum,1)',SigmaT{j} + 0.01^2 * eye(totalNum),1);];
    end
end

XTrain = {}; % latent process at sample timesteps
for i = 1:TrialNum
    XTrain{i} = [];
    for j = 1:sampleNum
        XTrain{i} = [XTrain{i},X{i}(:,TimeTable(j))];
    end
end

YTrain = {}; %YTrain{i}(j,k): i,j,k = index of trial, latent, sample
for l = 1:TrialNum
    YTrain{l} = trueC * XTrain{l};
end

Ytrue = {};
for l = 1:TrialNum
    Ytrue{l} = trueC * X{l};
end

baryTrain = [];
for l = 1:TrialNum
    baryTrain = [baryTrain,reshape(YTrain{l},[q*sampleNum,1])];
end

%% Test Data Generation

TestNum = 4;
XTestW = {};
for i = 1:TestNum
    XTestW{i} = [];
    for j = 1:p
        XTestW{i} = [XTestW{i};mvnrnd(zeros(totalNum,1)',SigmaT{j} + 0.01^2 * eye(totalNum),1);];
    end
end

XTest = {}; % latent process at sample timesteps
for i = 1:TestNum
    XTest{i} = [];
    for j = 1:sampleNum
        XTest{i} = [XTest{i},X{i}(:,TimeTable(j))];
    end
end

YTest = {}; %YTrain{i}(j,k): i,j,k = index of trial, latent, sample
for l = 1:TestNum
    YTest{l} = trueC * XTest{l};
end

YtrueTest = {};
for l = 1:TestNum
    YtrueTest{l} = trueC * XTestW{l};
end

%% Train

TrainNum = TrialNum;
% optimize hyperparameters 
% very sensitive to initialization!!!
OptNum = 10; % number of reinitialization
EMnum = 5; % number of EM iteration
bestEv = -1000000000; % highest loglikelihood found
scaleOPt = 0; Copt = 0; Ropt = 0; dopt = 0; % best scale found so far
for tt = 1:OptNum

        C = 10 * rand * randn(q,p); scale = 5 * rand * abs(randn(p,1)) + 1; R = diag(np^2 * ones(q,1)); d = 10 * rand * randn(q,1); % parameter initialization

       % scale = 3 * rand * abs(randn(p,1)) + scaleOPt; C = 2 * rand * randn(q,p) + Copt; R = diag(np^2 * ones(q,1)); d = 2 * rand * randn(q,1) + dopt;

    barR = kron(eye(sampleNum),R);
    for t = 1:EMnum   
        [Mean,Cov,Cov2,Cov3] = EMexpDirect(C,scale,R,d,baryTrain,p,sampleNum,TrainNum,TimeTable);
        [C,scale,d,R] = EMmaxDirect(Mean,Cov,Cov3,YTrain,p,sampleNum,scale,TimeTable);
        % calculate loglikelihood
        bard = kron(ones(sampleNum,1),d);
        barC = kron(eye(sampleNum),C);
        barK = [];
        for i = 1:sampleNum
            tempK = [];
            for j = 1:sampleNum
                v = [];
                for k = 1:p
                    v = [v;kernelEva(TimeTable(i),TimeTable(j),scale(k))];
                end
                tempK = [tempK,diag(v)];
            end
            barK = [barK;tempK];
        end
        covK = barC * barK * barC' + barR;
        %covK = barC * barK * barC' + np^2 * eye(q*T);
        loglikelihood = 0;
        covK = covK + 0.001 * eye(q*sampleNum);
        for i = 1:TrainNum
            loglikelihood = loglikelihood -1/2 * (baryTrain(:,i) - bard)' * covK^(-1) * (baryTrain(:,i) - bard) - 1/2 * (q * sampleNum * log(2*pi) + 2 * sum(log(diag(chol(covK)))));  % cholesky to compute logdet          
        end
        if loglikelihood > bestEv 
            bestEv = loglikelihood;
            scaleOPt = scale;
            Copt = C;
            Ropt = R;
            dopt = d;
        end
        disp(['Inner EM Iteration ' num2str(t) ': logLikelihood = ' num2str(loglikelihood) ', kernel width = ' num2str(scale') ';']);
    end
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
    disp(['Outer EM Iteration ' num2str(tt) ': best logLikelihood = ' num2str(bestEv) ', best kernel width = ' num2str(scaleOPt') ';']);
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
end

%%  Model extraction 
C = Copt;
R = Ropt;
d = dopt;
scale = scaleOPt;
% get important modeling statistics to save repeated computations
[barC,barR,bard,barK,Bj,Bjc,Sigma] = ImpStat(C,scale,R,d,sampleNum,TimeTable);


%% Latent recovery error


latentX = [];
for i = 1:TrialNum
    latentX = [latentX,latentEst(barK,barC,barR,baryTrain(:,i),bard)];
end
Xvec = [];
for i = 1:TrialNum
    Xvec = [Xvec,reshape(XTrain{i},[p*sampleNum,1])];
end
error = norm(latentX - Xvec)/norm(Xvec); % normalized error, sum over trials, all latent, timestep
t2 = cputime - t1;
disp([' normalized recover error = ' num2str(error) ' takes time = ' num2str(t2)]);


la = reshape(latentX(:,1),[p,sampleNum]);
la1 = la(1,:);
la2 = la(2,:);

% latent recovery plot
figure
subplot(2,4,1)
plot(la1);
hold on
plot(XTrain{1}(1,:))
title('latent recovery')
legend(['x^r_1 : ',num2str(scale(1))],['x_1 : ', num2str(GPScale(1))])
subplot(2,4,2)
plot(la2);
hold on
plot(XTrain{1}(2,:))
title('latent recovery')
legend('x^r_2','x_2')
legend(['x^r_2 : ',num2str(scale(2))],['x_2 : ', num2str(GPScale(2))])
% C recovery

subplot(2,4,3)
imagesc(Copt);
title('estimated C')
subplot(2,4,4)
imagesc(trueC);
title('true C, random Gaussian')

% Leave-one-out test

subplot(2,4,5)
plot(1:sampleNum,YTest{1}(1,:),'-x','DisplayName','True trajectory');
esty = predictGP(1,YTest{1},Bj,Bjc,Sigma,bard,q,sampleNum); % Test 1 data
title('leave 1 out')
hold on
plot(1:sampleNum,esty,'DisplayName','GP-prediction');
legend

subplot(2,4,6)
plot(1:sampleNum,YTest{1}(2,:),'-x','DisplayName','True trajectory');
esty = predictGP(2,YTest{1},Bj,Bjc,Sigma,bard,q,sampleNum); % Test 1 data
title('leave 2 out')
hold on
plot(1:sampleNum,esty,'DisplayName','GP-prediction');
legend

subplot(2,4,7)
plot(1:sampleNum,YTest{1}(3,:),'-x','DisplayName','True trajectory');
esty = predictGP(3,YTest{1},Bj,Bjc,Sigma,bard,q,sampleNum); % Test 1 data
title('leave 3 out')
hold on
plot(1:sampleNum,esty,'DisplayName','GP-prediction');
legend

subplot(2,4,8)
plot(1:sampleNum,YTest{1}(4,:),'-x','DisplayName','True trajectory');
esty = predictGP(4,YTest{1},Bj,Bjc,Sigma,bard,q,sampleNum); % Test 1 data
hold on
title('leave 4 out')
plot(1:sampleNum,esty,'DisplayName','GP-prediction');
legend

sgtitle(['parameters (p,q,n,w_1,w_2) = (', num2str([p,q,sampleNum,GPScale(1),GPScale(2)]),')'])