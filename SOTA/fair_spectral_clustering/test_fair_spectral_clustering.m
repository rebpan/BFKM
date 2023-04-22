X_raw = load("../../Datasets/2d-4c-no4.txt");
Color = load("../../Datasets/2d-4c-no4_Color.txt");
K = 4;

% Normalization and Standardization
X = normalize(X_raw, 1);
X = X./repmat(sqrt(sum(X.^2,2)),1, size(X,2));

% Contruct Similarity Matrix
distances = squareform(pdist(X));  
sigma = max(distances(:));
order = 2;
tmp = distances.^order/sigma;
W = exp(-tmp);

label = Fair_SC_normalized(W, K, Color);
% label = load('bank_FSCUN.txt');
Y = label2binary(label);
Y'*Y
bal = calcBAL(label, Color, K)
min(bal)
sse = calcSSE(X_raw', label)