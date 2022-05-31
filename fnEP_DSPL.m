%iteration function
function [AUC, RT] = fnEP_DSPL(X_train, Y_train, X_test, Y_test, options, ID)
% Online pairwise AUC
%--------------------------------------------------------------------------
% Input:
%        X_train:    the training instances
%        Y_train:    the vector of lables for X_train
%         X_test:    the testing instances
%         Y_test:    the vector of lables for X_test
%        options:    a struct containing rho, sigma, C, n_label and n_tick;
%             ID:    a randomized ID list of training data
% Output:
%            AUC:    area under ROC curve
%             RT:    run time 
%--------------------------------------------------------------------------
eta = options.eta;
lambda = options.lambda;
nPass = options.nPass;

tS = clock;

w = DSPL(X_train,Y_train,eta,lambda, nPass);
    
% end point
tE = clock;

% evaluate the method
[AUC, ~, ~] = fnEvaluate(X_test', Y_test, w');

RT = etime(tE, tS);









