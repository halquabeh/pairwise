%iteration function
function [AUCs, RTs] = SPAM(Xtrain, ytrian, Xtest, ytest, options, ID)
% SOLAM: Stochastic Online AUC Maximization
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
beta = options.beta;            % Constraint
beta1 = options.beta1;            % Constraint
C = options.eta;            % Step Size
epoches = options.nPass;      % Global pass over the data, epoches
% dimension of the data.
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
RTs = zeros(epoches,1);
AUCs = zeros(epoches,1);

tS = clock;
while (1)
    if (counter > epoches)
        break;
    end
        
    for j = 1:size(ID, 2)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        id = ID(j);
        x = Xtrain(:,id);
        y = ytrian(id);
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

	% end point
	tE = clock;
	% evaluate the method
	[AUC, ~, ~] = fnEvaluate(Xtest', ytest, w);
	RT = etime(tE, tS);

	AUCs(counter, 1) = AUC;
	RTs(counter, 1) = RT;
    
    counter = counter + 1;
end
