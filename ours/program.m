name = "ds577";
[data, color] = load_data(name);
k = 3;

F = indicator_matrix(color);

% preprocessing
X = normalize(data, 1);
X = X./repmat(sqrt(sum(X.^2, 2)), 1, size(X, 2));

% parameters setting
p1 = 1800;
p2 = 0;

tic; [label, center] = balanced_fair_kmeans(X', k, F, p1, p2); toc;

% evaluating
Y = indicator_matrix(label);
YY = diag(Y'*Y);
FYYY = F'*Y./YY';
n = size(X, 1);
f = sum(F)./n;
AW = 0;
for i = 1:k
	AW = AW + YY(i)*ws_distance(FYYY(:, i), f);
end
AW = AW/n;
BAL = calc_balance(label, color, k);

distM = squareform(pdist(X));
DI = dunns(k, distM, label);
SSE = calc_SSE(X', label);
