function [sj, para, val] = icanfast(R)

    tic;
    sz = size(R);
    
    h = zeros(8,8);
    for i=1:sz(1)
        Q = R(i,:) .* [1 1 1 1 -1 -1 -1 -1];
        W = Q' * Q;
        h = h+W;
    end
    
    right = [0 0 0 0 1 1 1 1];
    left   = [1 1 1 1 0 0 0 0];
    
    both = [left;right];

    [para,val] = quadprog(2*h,[],[],[],right,[1],zeros(8,1));
    sj = toc;