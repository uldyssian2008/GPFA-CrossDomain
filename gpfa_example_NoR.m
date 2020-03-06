%%%%%%%%%%%% GPFA Implementation %%%%%%%%%%%%%%%

%%% Caution: change kernelEva & kernelDer simultaneously %%%
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

rng(0,'twister'); % For reproducibility

% optimize hyperparameters
optTime = 50; % number of EM iteration
C = randn(q,p); scale = abs(randn(p,1)) + 10^(-1); R = diag(np^2 * ones(q,1)); d = randn(q,1); % parameter initialization
barR = kron(eye(T),R);

for t = 1:optTime   
    [Mean,Cov,Cov2] = EMexp(C,scale,R,d,baryTrain,p,q,T,TrainNum);
    [C,scale,d] = EMmaxNoR(Mean,Cov,Cov2,YTrain,p,T);
    
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
    %covK = barC * barK * barC' + barR;
    %covK = barC * barK * barC' + barR + np^2 * eye(q*T);
    %loglikelihood = -1/2 * (bary - bard)' * covK^(-1) * (bary - bard) - 1/2 * (q * T * log(2*pi) + log(det(covK)));
    %disp(['Iteration ' num2str(t) ': logLikelihood = ' num2str(loglikelihood) ';']);
end

% get important modeling statistics to save repeated computations
[barC,barR,bard,barK,Bj,Bjc,Sigma] = ImpStat(C,scale,R,d,T);

Ytest = dat(55);
Ytest = Ytest.spikes;
T1 = size(Ytest,2);
Ybar = [];
for i = 1:T1/binwidth
    z = sum(Ytest(:,(i-1) * binwidth + 1:i * binwidth ),2);
    Ybar = [Ybar,z];
end
Ytest = Ybar;

[traj,preError] = testGP(Ytest,Bj,Bjc,bard,Sigma,barR);


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





