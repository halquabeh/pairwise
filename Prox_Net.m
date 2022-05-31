function w = Prox_Net(w,beta,beta1,eta)
    
    c = beta1 * eta / (1+ eta*beta);
    w = sign(w) .* max(abs(w / (1+ eta*beta)) - c , 0);
    
    