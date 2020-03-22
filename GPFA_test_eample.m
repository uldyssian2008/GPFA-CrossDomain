%%%%%%%%%%%% GPFA Implementation %%%%%%%%%%%%%%%

rng(0,'twister'); % For reproducibility

%%% Caution: change kernelEva & kernelDer simultaneously %%%

%% sample from GP/Test (for Testing Purpose only)

samNum = 391;
TrialNum = 1;
T = 40;
np = 0.1;
load('latent_process_10');

%% 
p = 3;
q = 3;

% X{a}(b,c) : a - latent index, b - trial index, c - time index
trueC = randn(q,p); % transform matrix

YTrain = {};
for l = 1:TrialNum
    Z = [];
    for i = 1:40
        ZP = [];
        for kk = 1:p
            ZP = [ZP;X{kk}(l,10 * i -9)];
        end
        Z = [Z,trueC * ZP];
    end
    YTrain{l} = Z;
end
TrainNum = TrialNum;
baryTrain = [];
for l = 1:TrainNum
    baryTrain = [baryTrain,reshape(YTrain{l},[q*T,1])];
end
% YTest = {};
% XTest = [];
% testNum = 4;
% X = {};
% for i = 1:p
%     X{i} = mvnrnd(zeros(samNum,1)',SigmaT{i} + 0.01^2 * eye(samNum),testNum);
% end
% for l = 1:testNum
%     Z = [];
%     for i = 1:40
%         ZP = [];
%         for kk = 1:p
%             ZP = [ZP;X{kk}(l,10 * i -9)];
%         end
%         Z = [Z,trueC * ZP];
%     end
%     YTest{l} = Z;
% end


% for l = 1:testNum
%     Z = [];
%     for i = 1:40
%         ZP = [];
%         for kk = 1:p
%             ZP = [ZP;X{kk}(l,10 * i -9)];
%         end
%         Z = [Z,ZP];
%     end
%     XTest(:,l) = reshape(Z,[p*T,1]);
% end
testNum = 1;
Ytrue = {};
for l = 1:testNum
    Z = [];
    for i = 1:samNum
        ZP = [];
        for kk = 1:p
            ZP = [ZP;X{kk}(l,i)];
        end
        Z = [Z,trueC * ZP];
    end
    Ytrue{l} = Z;
end

%% Train

% optimize hyperparameters 
% very sensitive to initialization!!!
OptNum = 10; % number of reinitialization
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
        disp(['EM Iteration ' num2str(t) ': logLikelihood = ' num2str(loglikelihood) ', kernel width = ' num2str(scale') ';']);
    end
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
    disp(['Outer Iteration ' num2str(tt) ': best logLikelihood = ' num2str(bestEv) ', best kernel width = ' num2str(scaleOpt') ';']);
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
end

%% Latent recovery error

baryTrain = [];
for l = 1:testNum
    baryTrain = [baryTrain,reshape(YTrain{l},[q*T,1])];
end
latentX = [];
for i = 1:testNum
    latentX = [latentX,latentEst(barK,barC,barR,baryTrain(:,i),bard)];
end
la = reshape(latentX(:,1),[p,T]);

Yest = {};
for l = 1:testNum
    Yest{l} = Copt * la + kron(ones(1,T),dopt);
end




figure
plot(1:40,la(1,:));
hold on
plot(1:40,X{1}(1,1:10:391));
legend('x^r_1','x_1')

figure
plot(1:40,la(2,:));
hold on
plot(1:40,X{2}(1,1:10:391));
legend('x^r_2','x_2')

figure
plot(1:40,la(3,:));
hold on
plot(1:40,X{3}(1,1:10:391));
legend('x^r_3','x_3')

figure
plot(1:40,Ytrue{1}(1,1:10:391));
hold on
plot(1:40,Yest{1}(1,:));
legend('y^r_1','y_1')



