function [AUCs,RTs] = DSGD_alg(Xtrain,ytrain,Xtest,ytest,B,option,epochs)
%%  Input:
%    X, X_test   feature d x n for training and testing

%    Y,Y_test      binary labels 1 x n for training and testing

%    beta       adaptive expanding parameter for training size
% options has:
%    eta        the stepsize parameter

%    lmd1       squared L2 norm parameter

%    lmd2       L1 norm parameter

%	 epochs     iterate over n
% % outputs
%    w          solution for empire risk over the batch X,y
%% Checking if adatpive scheme
eta = option.eta;
lmd1 = option.lmd1;
lmd2 = option.lmd2;

[d, n]=size(Xtrain);
w=zeros(1,d);
idxP = find(ytrain==1);
idxN = find(ytrain==-1);
nP=length(idxP);
nN=length(idxN);
% iters = floor(num/2);

AUCs = zeros(epochs,1);
RTs = zeros(epochs,1);

split = 10;
total_num = 0;
tS = clock;
for epoch=1:epochs
    if B > 0
        stagenP = ceil(nP / B^(max(0, split-epoch)));
        stagenN = ceil(nN / B^(max(0, split-epoch)));
    else
        stagenP = nP;
        stagenN = nN;
    end
    
    rng(epoch);
	IdxP = idxP(randperm(stagenP));
	IdxN = idxN(randperm(stagenN));      
    for t=1:max(stagenP,stagenN)
        idP = IdxP(mod(t, stagenP)+1);
        idN = IdxN(mod(t, stagenN)+1);
        diff = Xtrain(:,idP) - Xtrain(:,idN);
        
        grad =  (w*diff-1)*diff';
        w = w - eta*grad;
        w = Prox_Net(w,lmd1,lmd2,eta);
    end
    total_num = total_num + max(stagenP, stagenN);

	tE = clock;
	% evaluate the method
	[AUC, ~, ~] = fnEvaluate(Xtest', ytest, w');
	RT = etime(tE, tS);
	RTs(epoch,1) = RT;
	AUCs(epoch,1) = AUC;

end


