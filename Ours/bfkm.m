function [label, C] = bfkm(X, k, F, p1, p2)
% X ... (d x n) data matrix
% k ... number of cluster
maxit = 1000;
n = size(X, 2);
label = randsrc(n,1,1:k);
last = zeros(n,1);
it = 1;

while any( label ~= last) && it<=maxit
    last = label;
    Y = label2binary(label);
    YY = Y'*Y; % cluster size
    C = X*Y/YY; % compute center of each cluster

    for i = 1:n
        minobj = 1e+14; % a very large number
        idx = 0;
        for c = 1:k
            Y(i, :) = 0;
            Y(i, c) = 1;
            YF = Y'*F;
            FF = F'*F;
            YY = Y'*Y;
            a = YF/FF - diag(YY/n);

            currobj = sum((X-C*Y').^2, 'all') + p1*sum(a.^2, 'all') + p2*sum(YY^(-1), 'all');
            if minobj > currobj
                minobj = currobj;
                idx = c;
            end
        end
        Y(i, :) = 0;
        Y(i, idx) = 1;
    end
    [~, label] = max(Y,[],2);
%     it = it+1;
end
end