clear all
clc
%% Function to Estimate
noise_std=0.2; 
N=260; % Number of samples 
x_init=linspace(0,4*pi,N)'; % Create a sin curve
fpure=@(v) cos(v).^2;
f=fpure(x_init)+noise_std*randn(size(x_init));
%%
plot(x_init,f)
xlabel('x')
ylabel('y')
title('y=cos(x)^2+noise')

%% Randomization of Data
n=numel(f);
A=[x_init,f]; 
A_rand=A(randperm(n),:);
x_rand=A_rand(:,1); y_rand=A_rand(:,2);
%% Data Splitting, 80% Train, 20% Test
y=y_rand(1:0.8*end); ys=y_rand(0.8*end:end);
x=x_rand(1:0.8*end); xs=x_rand(0.8*end:end);
plot(x,y,'o')
%% Transforming Data
y_init=y;
%[y, lambda]=boxcox(y);
ys_init=ys;
%ys=boxcox(lambda,ys);
%% Normalizing Outputs
ynorm= (y-mean(y))/(std(y)); ysnorm=(ys-mean(y))/(std(y));
%%
subplot(2,1,1);
histogram(ynorm)
xlabel('Function Value')
ylabel('Frequency')
title('Training Output')
subplot(2,1,2); 
histogram(ysnorm)
xlabel('Function Value')
ylabel('Frequency')
title('Testing Output')
%% Normalizing Inputs
xnorm=(x-mean(x))/(std(x)); xsnorm=(xs-mean(x))/(std(x));
%% Specifying the Gaussian Regression
meanfunc=[];
covfunc={@covSEard}; 
likfunc=@likGauss;                                        
inf=@infExact;
sn=0.1;
hyp0.cov=zeros(2,1);
hyp0.lik=log(sn);
Kx  = feval(covfunc{:}, hyp0.cov, xnorm);        % covariances between training data
Kxs = feval(covfunc{:}, hyp0.cov, xnorm,xsnorm); % cross covariances between training and test data
%%
hyp = minimize(hyp0, @gp, -100, inf, meanfunc, covfunc, likfunc, xnorm, ynorm);      
% hyp.cov: length parameters ln(l_1), ln(sf) sf is signal variance
% hyp.lik: ln(sn) , sn is the noise standard deviation
% large lengthscale corresponds to irrelevant feature for the regression!
             
[mu,s2] = gp(hyp, inf, meanfunc, covfunc, likfunc, xnorm,ynorm, xsnorm);   

Kxmin=   feval(covfunc{:}, hyp.cov, xnorm);     
Kxsmin = feval(covfunc{:}, hyp.cov, xnorm,xsnorm);  

% Trying to get proper scalings for comparison
mu_unscaled=std(y)*mu+mean(y);
%mu_unscaled=std(mu)*mu+mean(mu);
%mu2_real=exp((log(lambda*mu2_unscaled+1))/lambda);
y_compare=fpure(xs);
res=mu_unscaled-y_compare;
mspe = mean(res.^2);
% Predictions
%%
plotregression(y_compare,mu_unscaled,'o')
xlabel('True Values (Noise Free Function)')
ylabel('Actual Values')
legend('Ytrue=Yactual','Fit','Prediction')
%% Histogram of Residuals
hist_res=histogram(res);
xlabel('Residual Value[Predictions-Filtered function values]')
ylabel('Frequency')
%% THE EXPRESSION FOR THE PREDICTED VARIANCE (var)
%kTT = feval(covfunc{:}, hyp.cov, xsnorm);
%kTI = feval(covfunc{:}, hyp.cov, xsnorm,xnorm);
%kII = feval(covfunc{:}, hyp.cov, xnorm);
%sigma = exp(2*hyp.lik)*eye(max(size(xnorm))); 

% Posterior covariance matrix
%kPost = kTT - kTI/(kII+sigma)*kTI';

%var=diag(kPost+exp(2*hyp.lik)*eye(max(size(kTT)))); %s2
%% Determining Update Points
B=[xs,xsnorm,mu,s2];
n=4;      %  Number of points with largest predicted variance
Q=sortrows(B,4);
% corresponding inputs to highest vars:
in=Q(:,2);
in_max=in(length(in)-n+1:length(in));
%Creating M points in every broad confidence interval
M=30;
%  r = a + (b-a).*rand(M,1)

up=zeros((n-1)*M,1);    % (n-1) interpolations between n points

up(1:M)=in_max(1)+(in_max(2)-in_max(1)).*rand(M,1);

for j=2:n-1
    up((j-1)*M+1:j*M)=in_max(j)+(in_max(j+1)-in_max(j)).*rand(M,1);
end
% randomizing inputs corresponding to update points
m=numel(up);
up_rand=up(randperm(m),:);

xnorm_up=up_rand(1:0.8*end); xsnorm_up=up_rand(0.8*end:end);
xnorm_new=[xnorm;xnorm_up];
xsnorm_new=[xsnorm;xsnorm_up];
% Tracing back the output
x_up= std(x)*xnorm_up+mean(x);
xs_up=std(x)*xsnorm_up+mean(x);

y_up=fpure(x_up)+noise_std*randn(size(x_up));
y_new=[y_init;y_up];
ynorm_new=(y_new-mean(y))/(std(y));

ys_up=fpure(xs_up)+noise_std*randn(size(xs_up));
ys_new=[ys_init;ys_up];
ysnorm_new=(ys_new-mean(y))/(std(y));
%% Training and Prediction with UP
plot(xnorm_new,ynorm_new,'o')
%%
hyp2 = minimize(hyp0, @gp, -100, inf, meanfunc, covfunc, likfunc, xnorm_new, ynorm_new); 
[mu2,s22] = gp(hyp2, inf, meanfunc, covfunc, likfunc, xnorm_new,ynorm_new, xsnorm_new);
%%
Kxminup=   feval(covfunc{:}, hyp2.cov, xnorm_new);     
Kxsminup = feval(covfunc{:}, hyp2.cov, xnorm_new,xsnorm_new); 
%% Plotting Confidence Intervals
B2=[xsnorm_new,mu2,s22];
C2=sortrows(B2,1);

subplot(2,1,1);

interval2 = [C2(:,2)+2*sqrt(C2(:,3)); flip(C2(:,2)-2*sqrt(C2(:,3)),1)];
fill([C2(:,1); flip(C2(:,1),1)], interval2, [7 7 7]/8)
  hold on; plot(C2(:,1), C2(:,2)); plot(xnorm_new, ynorm_new, '+'), hold on
  xlabel('Input')
  ylabel('Output')
  legend('95 % Confidence Interval','Prediction','Training')
  title('With Update Points')
  
  subplot(2,1,2); 
  
C=sortrows(B,2);

  interval = [C(:,3)+2*sqrt(C(:,4)); flip(C(:,3)-2*sqrt(C(:,4)),1)];
fill([C(:,2); flip(C(:,2),1)], interval, [7 7 7]/8)
  hold on; plot(C(:,2), C(:,3)); plot(xnorm, ynorm, '+'), hold on
  xlabel('Input')
  ylabel('Output')
  title('No update points')
  legend('95 % Confidence Interval','Prediction','Training')
%% Plot without and with Update points
  subplot(2,1,1)
   plot(xsnorm,mu,'o')
 xlabel('Test Input')
 ylabel('Prediction')
 title('No update')
 subplot(2,1,2)
 plot(xsnorm_new,mu2,'o')
  xlabel('Test Input')
 ylabel('Prediction')
 title('With Update')
 %% Trying to get proper scalings for comparison
mu2_unscaled=std(y)*mu2+mean(y);
%mu2_real=exp((log(lambda*mu2_unscaled+1))/lambda);
xs_new=std(x)*xsnorm_new+mean(x);
 
y_compare_new=fpure(xs_new);
res2=mu2_unscaled-y_compare_new;
mspe2 = mean(res2.^2);
%%
histogram(res2);
xlabel('Residual Value[Predictions-Noise Free function values]')
ylabel('Frequency')
%%
plotregression(y_compare_new,mu2_unscaled,'o')
xlabel('True Values (Noise Free Function)')
ylabel('Actual Values')
legend('Ytrue=Yactual','Fit','Prediction')
%% Comparison with proper scalings
plot(xs_new,mu2_unscaled,'o')
hold on 
plot(x_init,f)
hold on 
plot(x_init,fpure(x_init),'k')
xlabel('x')
ylabel('y')
legend('Predictions','Function with Noise','Noise Free Function')
%% Noise standard deviation predictions
noise_std1=exp(2*hyp.lik);
noise_std2=exp(2*hyp2.lik);
