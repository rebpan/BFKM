function [label, C] = k_means(X, k, label)
maxit = 100;
n = size(X, 2);
last = zeros(n,1);
it = 0;

while any( label ~= last) && it<maxit
    last = label;
    Y = label2binary(label);
    YY = Y'*Y; % cluster size
    C = X*Y/YY; % compute center of each cluster
    D = pdist2(X', C');
    [~, label] = min(D, [], 2); % assign every sample to its nearest center
end
end