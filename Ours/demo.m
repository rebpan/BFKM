close all;

X_raw = load("../Datasets/elliptical.txt");
Color = load("../Datasets/elliptical_Color.txt");
K = 2;

[F, v] = SensCNVT(Color);


% Standardization and Normalization
X = normalize(X_raw, 1);
X = X./repmat(sqrt(sum(X.^2,2)),1, size(X,2));

% label = kmeans(X, K);

distM=squareform(pdist(X));

% 
p1 = 400;
p2 = 0;
tic; [label,C] = bfkm(X', K, F, p1, p2); toc;

Y = label2binary(label);
YY = Y'*Y;
FYYY = F'*Y/YY;
AW = 0;
for i = 1:size(FYYY, 2)
    tmp = YY(i, i) * ws_distance(FYYY(:, i), v);
    AW = AW + tmp;
end
BAL = calcBAL(label, Color, K)
AW = AW/size(X,1)
DI = dunns(K, distM, label)
SSE = calcSSE(X_raw', label)
Draw(X_raw, label);