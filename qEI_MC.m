function EI = qEI_MC(new_x,model,fmin,n)
% Monte Carlo simulation for qEI
% predictions and errors
[u,~,~,Cov] = predictor(new_x,model);
sample = mvnrnd(u',Cov,n);
EI  = mean(max(fmin - min(sample,[],2),0));