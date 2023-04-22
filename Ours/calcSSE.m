function a = calcSSE(X, label)
Y = label2binary(label);
YY = Y'*Y; % cluster size
C = X*Y/YY; % compute center of each cluster
a = sum((X-C*Y').^2, 'all');
end