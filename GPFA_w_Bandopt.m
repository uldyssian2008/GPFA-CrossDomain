%%%%%%%%%%%% GPFA Implementation %%%%%%%%%%%%%%%

rng(0,'twister'); % For reproducibility

%%% Caution: change kernelEva & kernelDer simultaneously %%%

%% sample from GP/Test (for Testing Purpose only)

t1 = cputime;
totalNum = 50; % number of GP approximation points
TrialNum = 10;
sampleNum = 50; % number of sample
gap = totalNum/sampleNum;
TimeTable = zeros(sampleNum,1);
for i = 1:sampleNum
    TimeTable(i) = gap * (i - 1) + 1;
end


q = 20;
p = 2;
np = 0.1;

SigmaT = {};
GPScale = 4 * ones(p,1);
% GPScale(1) = 10;
% GPScale(2) = 2;
for tt = 1:p
    for i = 1:totalNum
        for j = 1:totalNum
            SigmaT{tt}(i,j) = kernelEva(i,j,GPScale(tt));
        end
    end
end
trueC = 5 * randn(q,p);
% sparse matrix
for i = 1:q
    for j = 1:p
        if rand(1) > 1
            trueC(i,j) = 0;
        end
    end
end
%trueC = 5 * sprandn(q,p,0.2);
trued = 5 * randn(q,1);


X = {}; % X{i}(j,k):  i,j,k = index of trial, latent, time % whole latent process
for i = 1:TestNum
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
    YTrain{l} = trueC * XTrain{l} + trued;
end

Ytrue = {};
for l = 1:TrialNum
    Ytrue{l} = trueC * X{l} + trued;
end

baryTrain = [];
for l = 1:TrialNum
    baryTrain = [baryTrain,reshape(YTrain{l},[q*sampleNum,1])];
end

%% Test Data Generation

TestNum = 10;


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
        XTest{i} = [XTest{i},XTestW{i}(:,TimeTable(j))];
    end
end

YTest = {}; %YTrain{i}(j,k): i,j,k = index of trial, latent, sample
for l = 1:TestNum
    YTest{l} = trueC * XTest{l} + trued;
end

YtrueTest = {};
for l = 1:TestNum
    YtrueTest{l} = trueC * XTestW{l} + trued;
end

%% Train

TrainNum = TrialNum;
% optimize hyperparameters 
% very sensitive to initialization!!!
OptNum = 10; % number of EM reinitialization
EMnum = 15; % number of inner EM iteration
bestEv = -1000000000; % highest loglikelihood found
scaleOPt = 0; Copt = 0; Ropt = 0; dopt = 0; % best scale found so far
for tt = 1:OptNum
    C = 10 * rand * randn(q,p); scale = 5 * rand * abs(randn(p,1)) + 1; R = diag(np^2 * ones(q,1)); d = 10 * rand * randn(q,1); % parameter initialization
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
latentx = {};
CT = {};
for i = 1:TrialNum
    latentx{i} = reshape(latentX(:,i),[p,sampleNum]);
    CT{i} = [];
    for j = 1:p
        norm1 = norm(latentx{i}(j,:));
        latentx{i}(j,:) = latentx{i}(j,:) / norm1;
        CT{i}(:,j) = Copt(:,j) * norm1;
    end
end

% la = reshape(latentX(:,1),[p,sampleNum]);
% aa = norm(la(1,:));
% bb = norm(la(2,:));
% la1 = la(1,:)/aa;
% la2 = la(2,:)/bb;
% Copt(:,1) = Copt(:,1) * aa;
% Copt(:,2) = Copt(:,2) * bb;
TrueC = {};
for i = 1:TrialNum
    for j = 1:p
        norm1 = norm(XTrain{i}(j,:)); 
        TrueC{i}(:,j) = trueC(:,j) * norm1;
        XTrain{i}(j,:) = XTrain{i}(j,:) / norm1;
    end
end
% permutation check/ x_1 and x_2 may be swapped if identical distributed


for i = 1:TrialNum
    min1 = min(norm(XTrain{i}(1,:) - latentx{i}(1,:)) + norm(XTrain{i}(2,:) - latentx{i}(2,:)),norm(XTrain{i}(1,:) - latentx{i}(1,:)) + norm(XTrain{i}(2,:) + latentx{i}(2,:)));
    min1 = min(min1,norm(XTrain{i}(1,:) + latentx{i}(1,:)) + norm(XTrain{i}(2,:) + latentx{i}(2,:)));
    min1 = min(min1,norm(XTrain{i}(1,:) + latentx{i}(1,:)) + norm(XTrain{i}(2,:) - latentx{i}(2,:)));
    min1 = min(min1,norm(XTrain{i}(1,:) - latentx{i}(2,:)) + norm(XTrain{i}(2,:) - latentx{i}(1,:)));
    min1 = min(min1,norm(XTrain{i}(1,:) + latentx{i}(2,:)) + norm(XTrain{i}(2,:) - latentx{i}(1,:)));
    min1 = min(min1,norm(XTrain{i}(1,:) - latentx{i}(2,:)) + norm(XTrain{i}(2,:) + latentx{i}(1,:)));
    min1 = min(min1,norm(XTrain{i}(1,:) + latentx{i}(2,:)) + norm(XTrain{i}(2,:) + latentx{i}(1,:)));

    if norm(XTrain{i}(1,:) - latentx{i}(1,:)) + norm(XTrain{i}(2,:) + latentx{i}(2,:)) == min1
                
        latentx{i}(2,:) = -latentx{i}(2,:);
        CT{i}(:,2) = -CT{i}(:,2);
       
    elseif norm(XTrain{i}(1,:) + latentx{i}(1,:)) + norm(XTrain{i}(2,:) - latentx{i}(2,:)) == min1
              
        latentx{i}(1,:) = -latentx{i}(1,:);
        CT{i}(:,1) = -CT{i}(:,1);
       
    elseif norm(XTrain{i}(1,:) + latentx{i}(1,:)) + norm(XTrain{i}(2,:) + latentx{i}(2,:)) == min1
      
        latentx{i}(1,:) = -latentx{i}(1,:);
        latentx{i}(2,:) = -latentx{i}(2,:);
        CT{i}(:,1) = -CT{i}(:,1);
        CT{i}(:,2) = -CT{i}(:,2);
        
      
    elseif norm(XTrain{i}(1,:) - latentx{i}(2,:)) + norm(XTrain{i}(2,:) - latentx{i}(1,:)) == min1

        z = latentx{i}(1,:);
        latentx{i}(1,:) = latentx{i}(2,:);
        latentx{i}(2,:) = z;
        Ct = CT{i};
        CT{i}(:,1) = Ct(:,2);
        CT{i}(:,2) = Ct(:,1);
        
    elseif norm(XTrain{i}(1,:) + latentx{i}(2,:)) + norm(XTrain{i}(2,:) - latentx{i}(1,:)) == min1
     
        z = latentx{i}(1,:);
        latentx{i}(1,:) = -latentx{i}(2,:);
        latentx{i}(2,:) = z;
        Ct = CT{i};
        CT{i}(:,1) = -Ct(:,2);
        CT{i}(:,2) = Ct(:,1);
      
  
    elseif norm(XTrain{i}(1,:) - latentx{i}(2,:)) + norm(XTrain{i}(2,:) + latentx{i}(1,:)) == min1
        
        z = latentx{i}(1,:);
        latentx{i}(1,:) = latentx{i}(2,:);
        latentx{i}(2,:) = -z;
        Ct = CT{i};
        CT{i}(:,1) = Ct(:,2);
        CT{i}(:,2) = -Ct(:,1); 
       
      
    elseif norm(XTrain{i}(1,:) + latentx{i}(2,:)) + norm(XTrain{i}(2,:) + latentx{i}(1,:)) == min1
        
        z = latentx{i}(1,:);
        latentx{i}(1,:) = -latentx{i}(2,:);
        latentx{i}(2,:) = -z;
        Ct = CT{i};
        CT{i}(:,1) = -Ct(:,2);
        CT{i}(:,2) = -Ct(:,1); 
    
      
    end
end

Xvec = [];
for i = 1:TrialNum
    Xvec = [Xvec,reshape(XTrain{i},[p*sampleNum,1])];
end


% Rotation align x
for i = 1:TrialNum
    OptNum = 10;
    optV = 1000000000;
    OptTheta = 0;
    fun = @(theta) norm([cos(theta),-sin(theta);sin(theta),cos(theta)] * latentx{i} - XTrain{i});
    for j = 1:OptNum
        theta0 = 2 * pi * rand(1);
        [thetaTemp,fval] = patternsearch(fun,theta0,[],[],[],[],0,2 * pi);
        if fval < optV
            optV = fval;
            OptTheta = thetaTemp;
        end
    end
    disp(['x angle = ',num2str(OptTheta)])
    latentx{i} = [cos(OptTheta),-sin(OptTheta);sin(OptTheta),cos(OptTheta)] * latentx{i};
    CT{i} = CT{i}/[cos(OptTheta),-sin(OptTheta);sin(OptTheta),cos(OptTheta)];
end




% principle latent processes
Platentx = {};
TPlatentx = {};
S = [];
S2 = [];
for i = 1:TrialNum
    [U,S,V] = svd(CT{i});
    Platentx{i} = V' * latentx{i};
    [U2,S2,V2] = svd(TrueC{i});
    TPlatentx{i} = V2' * XTrain{i};
end
S = diag(S);
S = [S;max(S) / min(S)];
S2 = diag(S2);
S2 = [S2;max(S2) / min(S2)];


% Rotation align principle x
for i = 1:TrialNum
    OptNum = 10;
    optV = 1000000000;
    OptTheta = 0;
    fun = @(theta) norm([cos(theta),-sin(theta);sin(theta),cos(theta)] * Platentx{i} - TPlatentx{i});
    for j = 1:OptNum
        theta0 = 2 * pi * rand(1);
        [thetaTemp,fval] = patternsearch(fun,theta0,[],[],[],[],0,2 * pi);
        if fval < optV
            optV = fval;
            OptTheta = thetaTemp;
        end
    end
    disp(['x principle angle = ',num2str(OptTheta)])
    Platentx{i} = [cos(OptTheta),-sin(OptTheta);sin(OptTheta),cos(OptTheta)] * Platentx{i};
    CT{i} = CT{i}/[cos(OptTheta),-sin(OptTheta);sin(OptTheta),cos(OptTheta)];
end

Xerror = zeros(p,TrialNum); 
for i = 1:p
    for j = 1:TrialNum
        Xerror(i,j) = norm(latentx{j}(i,:) - XTrain{j}(i,:)) / sqrt(sampleNum);
    end
end

PXerror = zeros(p,TrialNum); 
for i = 1:p
    for j = 1:TrialNum
        PXerror(i,j) = norm(Platentx{j}(i,:) - TPlatentx{j}(i,:)) / sqrt(sampleNum);
    end
end


Xerror = Xerror';
stderror = std(Xerror,1);
meanerror = mean(Xerror,1);
t2 = cputime - t1;
disp([' normalized X recover error = ' mat2str(meanerror) ' takes time = ' num2str(t2)]);

PXerror = PXerror';
Pstderror = std(PXerror);
Pmeanerror = mean(PXerror);
t2 = cputime - t1;
disp([' normalized principle X recover error = ' mat2str(Pmeanerror) ' takes time = ' num2str(t2)]);

Cerror = zeros(TrialNum,1);
for i = 1:TrialNum
    Cerror(i) = norm(CT{i} - TrueC{i}) / norm(TrueC{i});
end
stdC = std(Cerror);
meanC = mean(Cerror);

derror = 0;
for i = 1:TrialNum
    derror = norm(dopt - trued) / norm(trued);
end


disp([' normalized C recover error = ' mat2str(meanC) ' takes time = ' num2str(t2)]);
disp([' normalized d recover error = ' mat2str(derror) ' takes time = ' num2str(t2)]);

% Leave-one-out test

Loo = zeros(q,TestNum);
for i = 1:q
    for j= 1:TestNum
        Loo(i,j) = norm(predictGP(i,YTest{j},Bj,Bjc,Sigma,bard,q,sampleNum) - YTest{j}(i,:)') / norm(YTest{j}(i,:));%sqrt(sampleNum);
    end
end
Loo = Loo';
mLoo = mean(Loo);
stdLoo = std(Loo);




% latent recovery plot




figure
subplot(4,5,1)
hold on
bar(1:q,mLoo)
errorbar(1:q,mLoo,stdLoo,'.')
title('Leave-one-out error')

subplot(4,5,2)
hold on
bar(1:p + 1,S2)
%legend('\lambda_1','\lambda_2','\lambda_2/\lambda_1')
title('True eigenvalue + condition number')

subplot(4,5,3)
hold on
bar(1:p + 1,S)
%legend('\lambda_1','\lambda_2','\lambda_2/\lambda_1')
title('Estimated eigenvalue + condition number')

% C recovery

subplot(4,5,4)
imagesc(CT{1});
title('estimated C')
colorbar
subplot(4,5,5)
imagesc(TrueC{1});
title('true C, random Gaussian')
colorbar


subplot(4,5,6)
hold on
bar(1:p,meanerror)
errorbar(1:p,meanerror,stderror,'.')
title('x recovery error')


subplot(4,5,7)
scatter(XTrain{1}(1,:),XTrain{1}(2,:),[],1:sampleNum);
colorbar
title('True latent trajectory')

subplot(4,5,8)
scatter(latentx{1}(1,:),latentx{1}(2,:),[],1:sampleNum);
colorbar
title('Estimated latent trajectory')

subplot(4,5,9)
plot(latentx{1}(1,:));
hold on
plot(XTrain{1}(1,:))
title('latent recovery')
legend(['x^r_1 : ',num2str(scale(1))],['x_1 : ', num2str(GPScale(1))])
subplot(4,5,10)
plot(latentx{1}(2,:));
hold on
plot(XTrain{1}(2,:))
title('latent recovery')
legend('x^r_2','x_2')
legend(['x^r_2 : ',num2str(scale(2))],['x_2 : ', num2str(GPScale(2))])



subplot(4,5,11)
hold on
bar(1:p,Pmeanerror)
errorbar(1:p,Pmeanerror,Pstderror,'.')
title('Principle x recovery error')
%title([' (x^r_1,x_1) : (',num2str(scale(1)), ',' num2str(GPScale(1)),'), (x^r_2,x_2) : (',num2str(scale(2)), ',' ,num2str(GPScale(2)),')'])


subplot(4,5,12)
scatter(TPlatentx{1}(1,:),TPlatentx{1}(2,:),[],1:sampleNum);
colorbar
title('True principle latent trajectory')

subplot(4,5,13)
scatter(Platentx{1}(1,:),Platentx{1}(2,:),[],1:sampleNum);
colorbar
title('Estimated principle latent trajectory')

subplot(4,5,14)
plot(Platentx{1}(1,:));
hold on
plot(TPlatentx{1}(1,:))
title('1st principle latent recovery')
subplot(4,5,15)
plot(Platentx{1}(2,:));
hold on
plot(TPlatentx{1}(2,:))
title('2nd principle latent recovery')

subplot(4,5,16)
scatter3(YTrain{1}(1,:),YTrain{1}(2,:),YTrain{1}(3,:),[],1:sampleNum);
colorbar
title('True neuron trajectory')

subplot(4,5,17)
ZZZZ = CT{1} * latentx{1};
scatter3(ZZZZ(1,:),ZZZZ(2,:),ZZZZ(3,:),[],1:sampleNum);
colorbar
title('Estimated neuron trajectory')


% subplot(2,4,7)
% scatter3(XTrain{1}(1,:),XTrain{1}(2,:),XTrain{1}(3,:),[],1:sampleNum);
% colorbar
% title('True latent trajectory')
% 
% subplot(2,4,8)
% scatter3(latentx{1}(1,:),latentx{1}(2,:),latentx{1}(3,:),[],1:sampleNum);
% colorbar
% title('Estimated latent trajectory')

% subplot(2,4,7)
% plot(1:sampleNum,YTest{1}(3,:),'-x','DisplayName','True trajectory');
% esty = predictGP(3,YTest{1},Bj,Bjc,Sigma,bard,q,sampleNum); % Test 1 data
% title('leave 3 out')
% hold on
% plot(1:sampleNum,esty,'DisplayName','GP-prediction');
% legend
% 
% subplot(2,4,8)
% plot(1:sampleNum,YTest{1}(4,:),'-x','DisplayName','True trajectory');
% esty = predictGP(4,YTest{1},Bj,Bjc,Sigma,bard,q,sampleNum); % Test 1 data
% hold on
% title('leave 4 out')
% plot(1:sampleNum,esty,'DisplayName','GP-prediction');
% legend
sgtitle(['# Latent = ',num2str(p),', # Neuron = ',num2str(q), ', # Sample = ',num2str(sampleNum), ', # Trial = ',num2str(TrialNum),', Bandwidth = ',mat2str([GPScale(1),GPScale(2)]), ', Imperfect Bandwidth'])
%sgtitle(['parameters (p,q,n,w_1,w_2) = (', num2str([p,q,sampleNum,GPScale(1),GPScale(2)]),')'])