load('mat_sample/sample_dat');
Y = dat(1);
Y = Y.spikes;
T = size(Y,2);
q = size(Y,1);

% latent dimension
p = 10;
% noise power
np = 0.05; 
% bin spike coumt
binwidth = 10;
Ybar = [];
for i = 1:T/binwidth
    z = sum(Y(:,(i-1) * binwidth + 1:i * binwidth ),2);
    Ybar = [Ybar,z];
end
T = T/binwidth;
Y = Ybar;
bary = reshape(Y,[q*T,1]);

% optimize hyperparameters
optTime = 200; % number of EM iteration
C = randn(q,p); scale = abs(randn(p,1)) + 10^(-1); R = diag(0.01 * ones(q,1)); d = randn(q,1); % parameter initialization
barR = kron(eye(T),R);

for t = 1:optTime   
    [Mean,Cov,Cov2] = EMexp(C,scale,R,d,Y);
    [C,scale,d] = EMmaxNoR(Mean,Cov,Cov2,Y);
    
    % calculate loglikelihood
    bard = kron(ones(T,1),d);
    barC = kron(eye(T),C);
    barK = [];
    for i = 1:T
        tempK = [];
        for j = 1:T
            v = [];
            if i == j
                for k = 1:p
                    v = [v;(1-10^(-3))*exp(-(i-j)^2/2/scale(k)^2)+10^(-3)];
                end
                tempK = [tempK,diag(v)];
            else
                for k = 1:p
                    v = [v;(1-10^(-3))*exp(-(i-j)^2/2/scale(k)^2)];
                end
                tempK = [tempK,diag(v)];
            end
        end
        barK = [barK;tempK];
    end
    %covK = barC * barK * barC' + barR;
    covK = barC * barK * barC' + barR + np^2 * eye(q*T);
    loglikelihood = -1/2 * (bary - bard)' * covK^(-1) * (bary - bard) - 1/2 * (q * T * log(2*pi) + log(det(covK)));
    disp(['Iteration ' num2str(t) ': logLikelihood = ' num2str(loglikelihood) ';']);
end

% get important modeling statistics to save repeated computations
[barC,barR,bard,barK,Bj,Bjc,Sigma] = ImpStat(C,scale,R,d);

Ytest = dat(1);
Ytest = Ytest.spikes;
T1 = size(Ytest,2);
Ybar = [];
for i = 1:T1/binwidth
    z = sum(Ytest(:,(i-1) * binwidth + 1:i * binwidth ),2);
    Ybar = [Ybar,z];
end
Ytest = Ybar;

[traj,preError] = testGP(Ytest,C,scale,R,d);

% visualization
figure
subplot(1,6,1);
hold on 
scatter(1:T,Y(1,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
hold on
plot(traj(:,1),'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')

subplot(1,6,2);
hold on 
scatter(1:T,Y(2,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
hold on
plot(traj(:,2),'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')

subplot(1,6,3);
hold on 
scatter(1:T,Y(3,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
hold on
plot(traj(:,3),'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')

subplot(1,6,4);
hold on 
scatter(1:T,Y(4,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
hold on
plot(traj(:,4),'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')

subplot(1,6,5);
hold on 
scatter(1:T,Y(5,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
hold on
plot(traj(:,5),'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')

subplot(1,6,6);
hold on 
scatter(1:T,Y(6,:),'DisplayName','real data');
%hold on
%plot(Y(1,:));
legend('real data')
hold on
plot(traj(:,6),'DisplayName','GP-approx.')
xlabel('Time (10ms)')
ylabel('Spike')





