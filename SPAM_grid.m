function optBest = SPAM_grid(X_train, y_train, epoches, nCV,ID)
% Used for parameter selecting
% SOLAM: Cited Stochastic Online AUC Maximization
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
for b = -5:1:1
    beta = 10^b;
    for m = 0.1:0.1:1
        C = m;
        for b1 = -5:1:1
            beta1 = 10^b1;
            curAUC = zeros(1, nCV);
            for k = 1:1
                Xtrain = X_train(:,CVIdx~=k);
                ytrain = y_train(CVIdx~=k);
                Xtest = X_train(:,CVIdx==k);
                ytest = y_train(CVIdx==k);
                ID=1:size(ytrain,1);
                d = size(Xtrain, 1);
                
                ExP = zeros(d,1);
                ExN = zeros(d,1);
                
                Pold = 0;                   % \bar{P}(y=1|x)
                w = zeros(d + 0, 1);  % primal variable = [w;a;b]
                aold = 0;
                bold = 0;
                alpha_ = 0;                  % Dual variable (\alpha)
                Gold = 0;                  % Step size mod
                
                w_ = zeros(d + 0 , 1);   % New Primal Variable
                w_(1:d, 1) = zeros(d, 1); % Initialize in feassible region
                a = 0;
                b = 0;
                
                
                % iteration time.
                T = 1;
                % pass time.
                counter = 1;
                % recprd hist every topic
                while (1)
                    if (counter > epoches)
                        break;
                    end
                    
                    for j = 1:size(ID, 2)
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        id = ID(j);
                        x = Xtrain(:,id);
                        y = ytrain(id);
                        % get the step parameter
                        G = C / sqrt(T);
                        % two cases for either postive and negative samples.
                        if y == 1
                            %step 2
                            ExP =  (ExP + x) / 2 ;
                            P = ((T - 1)*Pold + 1) / T;
                            %
                            a = w_'*ExP;
                            alpha =  w_'*(ExN - ExP);
                            dw = 2*(1 - P)*(w_'*x - a)*x - ...
                                2*(1 + alpha)*(1 - P)*x;%-2*(1 - P)*(Ww'*x - a); 0];
                            w_ = w_ - G*dw;
                            w_ = Prox_Net(w_,beta,beta1,G);
                        else
                            %step 2
                            ExN =  (ExN + x) / 2 ;
                            P = (T - 1)*Pold / T;
                            b = w_'*ExN;
                            alpha =  w_'*(ExN - ExP);
                            dw = 2*P*(w_'*x - b)*x + ...
                                2*(1 + alpha)*P*x; 0;%-2*P*(Ww'*x - b)];
                            w_ = w_ - G*dw;
                            w_ = Prox_Net(w_,beta,beta1,G);
                        end
                        %update gamma_
                        Gsum = Gold + G;
                        %update v_
                        w = (Gold * w + G * w_) / Gsum;
                        a = (Gold * aold + G * a) / Gsum;
                        b = (Gold * bold + G * b) / Gsum;
                        %update alpha_
                        alpha_ = (Gold*alpha_ + G*alpha) / Gsum;
                        % update the information.
                        Pold = P;
                        aold = a;
                        bold = b;
                        Gold = Gsum;
                        %update counts
                        T = T + 1;
                        
                    end
                    counter = counter +1;
                end
                % evaluate the method
                [auc, ~, ~] = fnEvaluate(Xtest', ytest, w);
                curAUC(1, k) = auc;
            end
            meanAUC = mean(curAUC);
            if meanAUC > maxAUC
                maxAUC = meanAUC;
                optBest.beta = beta;
                optBest.beta1 = beta1;
                optBest.eta = C;
            end
        end
    end
end
optBest.nPass = epoches;
