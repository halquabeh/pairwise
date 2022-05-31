%iteration function
function optBest = DSGD_grid(X_train, Y_train, B,nPass, nCV)
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
        lmd1 = 10^n;
        for v = -10:2:2
            lmd2 = 10^v;
            
            curAUC = zeros(1, nCV);
            
            for k = 1:1
                
                X_tr = X_train(:,CVIdx~=k);
                Y_tr = Y_train(CVIdx~=k);
                X_te = X_train(:,CVIdx==k);
                Y_te = Y_train(CVIdx==k);
                options.eta = eta; options.lmd1=lmd1; options.lmd2=lmd2;
                [AUC, ~] = DSGD_alg(X_tr,Y_tr,X_te,Y_te,B,options,nPass);
                
                curAUC(1, k) = AUC(end);
            end
            
            meanAUC = mean(curAUC);
            if meanAUC > maxAUC
                maxAUC = meanAUC;
                optBest.eta = eta;
                optBest.lmd1 = lmd1;
                optBest.lmd2 = lmd2;
            end
        end
    end
end

optBest.nPass = nPass;
