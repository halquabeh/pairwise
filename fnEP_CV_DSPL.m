%iteration function
function optBest = fnEP_CV_DSPL(X_train, Y_train, nPass, nCV)
% Used for parameter selecting
% SOLAM: Stochastic Online AUC Maximization
%--------------------------------------------------------------------------
% Input:
%        X_train:    the training instances
%        Y_train:    the vector of lables for X_train
% Output:
%        optBest:    best option
%--------------------------------------------------------------------------
% maximal value
maxAUC = 0;
%% generate the cross idx
CVIdx = crossvalind('Kfold', size(X_train, 2), nCV);
% iterate the variable of eta
for m = -12:2:10
    eta = 2^m;
    for n = -10:2:2
        lambda = 10^n;
            
        curAUC = zeros(1, nCV);
        
        for k = 1:nCV
            
            X_tr = X_train(:,CVIdx~=k);
            Y_tr = Y_train(CVIdx~=k);
            X_te = X_train(:,CVIdx==k);
            Y_te = Y_train(CVIdx==k);
                 
            w = DSPL(X_tr,Y_tr,eta,lambda,nPass);

            % evaluate the method
            [auc, ~, ~] = fnEvaluate(X_te', Y_te, w');
            curAUC(1, k) = auc;
        end
        
        meanAUC = mean(curAUC);
        if meanAUC > maxAUC
            maxAUC = meanAUC;
            optBest.eta = 2^m;
            optBest.lambda = 2^n;
        end
    end
end

optBest.nPass = nPass;
