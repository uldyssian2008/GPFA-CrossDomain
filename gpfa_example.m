%%%%%%%%%%%% GPFA Implementation %%%%%%%%%%%%%%%

rng(0,'twister'); % For reproducibility

%%% Caution: change kernelEva & kernelDer simultaneously %%%

%% Set up

load('mat_sample/sample_dat');
dataNum = size(dat,2);
TrainNum = 30;
TestNum = dataNum - TrainNum;
Y = dat(1:TrainNum);
Ytes = dat(TrainNum + 1:end);
T = size(Y(1).spikes,2);
q = size(Y(1).spikes,1);

% latent dimension
p = 10;
% noise power
np = 0.1; 
% bin spike coumt
% Train set/Test set
YTrain = {};
YTest = {};
binwidth = 10;
for d = 1:TrainNum
    Ybar = [];
    for i = 1:T/binwidth
        z = sum(Y(d).spikes(:,(i-1) * binwidth + 1:i * binwidth ),2);
        Ybar = [Ybar,z];
    end
    YTrain{d} = Ybar;
end
for d = 1:TestNum
    Ybar = [];
    for i = 1:T/binwidth
        z = sum(Ytes(d).spikes(:,(i-1) * binwidth + 1:i * binwidth ),2);
        Ybar = [Ybar,z];
    end
    YTest{d} = Ybar;
end
for d = 1:TrainNum
    YTrain{d} = sqrt(YTrain{d});
end
for d = 1:TestNum
    YTest{d} = sqrt(YTest{d});
end
T = T/binwidth;
baryTrain = [];
for l = 1:TrainNum
    baryTrain = [baryTrain,reshape(YTrain{l},[q*T,1])];
end
baryTest = [];
for l = 1:TestNum
    baryTest = [baryTest,reshape(YTest{l},[q*T,1])];
end

%% sample from GP/Test (for Testing Purpose only)

samNum = 391;
TrialNum = 10;
T = 40;
q = 4;
p = 2;
SigmaT1 = zeros(samNum,samNum);
SigmaT2 = zeros(samNum,samNum);
SamScale1 = 3;
SamScale2 = 3;
C = randn(q,p);
for i = 1:samNum
    for j = 1:samNum
        SigmaT1(i,j) = kernelEva(0.1*i + 0.9,0.1*j + 0.9,SamScale1);
    end
end
for i = 1:samNum
    for j = 1:samNum
        SigmaT2(i,j) = kernelEva(0.1*i + 0.9,0.1*j + 0.9,SamScale2);
    end
end  
X1 = mvnrnd(zeros(samNum,1)',SigmaT1 + 0.01^2 * eye(samNum),TrialNum);
X2 = mvnrnd(zeros(samNum,1)',SigmaT2 + 0.01^2 * eye(samNum),TrialNum);
YGPtest = C*[X1(1,:);X2(1,:)];
figure
plot(1:0.1:40,YGPtest(1,:));
hold on
plot(1:0.1:40,YGPtest(2,:));
hold on
plot(1:0.1:40,YGPtest(3,:));
hold on
plot(1:0.1:40,YGPtest(4,:));
YTrain = {};
for l = 1:TrialNum
    Z = [];
    for i = 1:40
        Z = [Z,C*[X1(l,10 * i -9);X2(l,10 * i -9)]];
    end
    YTrain{l} = Z;
end
TrainNum = TrialNum;
baryTrain = [];
for l = 1:TrainNum
    baryTrain = [baryTrain,reshape(YTrain{l},[q*T,1])];
end    

%% Train

% optimize hyperparameters 
% very sensitive to initialization!!!
OptNum = 20; % number of reinitialization
EMnum = 5; % number of EM iteration
bestEv = -1000000000; % highest loglikelihood found
scaleOPt = 0; Copt = 0; Ropt = 0; dopt = 0; % best scale found so far
for tt = 1:OptNum
    C = 10 * rand * randn(q,p); scale =5 * rand * abs(randn(p,1)) + 10^(-1); R = diag(np^2 * ones(q,1)); d = 10 * rand * randn(q,1); % parameter initialization
    barR = kron(eye(T),R);
    for t = 1:EMnum   
        [Mean,Cov,Cov2,Cov3] = EMexpDirect(C,scale,R,d,baryTrain,p,q,T,TrainNum);
        [C,scale,d,R] = EMmaxDirect(Mean,Cov,Cov2,Cov3,YTrain,p,T,scale);

        % calculate loglikelihood
        bard = kron(ones(T,1),d);
        barC = kron(eye(T),C);
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
        covK = barC * barK * barC' + barR;
        %covK = barC * barK * barC' + np^2 * eye(q*T);
        loglikelihood = 0;
        for i = 1:TrainNum
            loglikelihood = loglikelihood -1/2 * (baryTrain(:,i) - bard)' * covK^(-1) * (baryTrain(:,i) - bard) - 1/2 * (q * T * log(2*pi) + log(det(covK)));
        end
        if loglikelihood > bestEv 
            bestEv = loglikelihood;
            scaleOpt = scale;
            Copt = C;
            Ropt = R;
            dopt = d;
        end
        disp(['EM Iteration ' num2str(t) ': logLikelihood = ' num2str(loglikelihood) ',best kernel width = ' num2str(scaleOpt') ';']);
    end
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
    disp(['Outer Iteration ' num2str(tt) ': best logLikelihood = ' num2str(bestEv) ';']);
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
end
%% Test
C = Copt;
R = Ropt;
d = dopt;
scale = scaleOpt;
% get important modeling statistics to save repeated computations
[barC,barR,bard,barK,Bj,Bjc,Sigma] = ImpStat(C,scale,R,d,T);

preError = 0;
for ii = 1:TestNum
    Ytest = dat(TrainNum + ii);
    Ytest = Ytest.spikes;
    T1 = size(Ytest,2);
    Ybar = [];
    for i = 1:T1/binwidth
        z = sum(Ytest(:,(i-1) * binwidth + 1:i * binwidth ),2);
        Ybar = [Ybar,z];
    end
    Ytest = Ybar;
    [~,val] = testGP(Ytest,Bj,Bjc,bard,Sigma,barR);
    preError = preError + val;
end
preError = preError / TestNum;
disp([ ' Prediction Error = ' num2str(preError) ';']);
% visualization

figure
subplot(1,6,1);
hold on 
scatter(1:T,Ytest(1,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
Num = 100;
esty = [];
for it = 1:Num + 1
    esty(it) = predictGP((it-1)*T/Num,1,Ytest,barC,Bjc,scale,Sigma,bard,p,q,T,d);
end
hold on
plot(0:T/Num:T,esty,'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')

subplot(1,6,2);
hold on 
scatter(1:T,Ytest(2,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
Num = 100;
esty = [];
for it = 1:Num + 1
    esty(it) = predictGP((it-1)*T/Num,2,Ytest,barC,Bjc,scale,Sigma,bard,p,q,T,d);
end
hold on
plot(0:T/Num:T,esty,'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')

subplot(1,6,3);
hold on 
scatter(1:T,Ytest(3,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
Num = 100;
esty = [];
for it = 1:Num + 1
    esty(it) = predictGP((it-1)*T/Num,3,Ytest,barC,Bjc,scale,Sigma,bard,p,q,T,d);
end
hold on
plot(0:T/Num:T,esty,'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')

subplot(1,6,4);
hold on 
scatter(1:T,Ytest(4,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
Num = 100;
esty = [];
for it = 1:Num + 1
    esty(it) = predictGP((it-1)*T/Num,4,Ytest,barC,Bjc,scale,Sigma,bard,p,q,T,d);
end
hold on
plot(0:T/Num:T,esty,'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')

subplot(1,6,5);
hold on 
scatter(1:T,Ytest(5,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
Num = 100;
esty = [];
for it = 1:Num + 1
    esty(it) = predictGP((it-1)*T/Num,5,Ytest,barC,Bjc,scale,Sigma,bard,p,q,T,d);
end
hold on
plot(0:T/Num:T,esty,'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')

subplot(1,6,6);
hold on 
scatter(1:T,Ytest(6,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
Num = 100;
esty = [];
for it = 1:Num + 1
    esty(it) = predictGP((it-1)*T/Num,6,Ytest,barC,Bjc,scale,Sigma,bard,p,q,T,d);
end
hold on
plot(0:T/Num:T,esty,'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')





