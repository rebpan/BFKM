function [label, center, sumd, D] = balanced_fair_kmeans(X, k, F, p1, p2)
	n = size(X, 2);
	last = zeros(n, 1);
	label = randsrc(n, 1, 1:k);
	max_iter = 100;
	iter = 0;
	err = 1;
	Y = indicator_matrix(label);
	YY = diag(Y'*Y); % cluster size
	FF = diag(F'*F); % group size
	YF = Y'*F;
	center = X*Y./YY';
	[D, sumd] = update_dist(X, center, k, label);
	while (err > 0.001  && iter <= max_iter && any(label ~= last))
		last = label;
		for i = 1:n
			idx_label = label(i);
			idx_group = find(F(i, :), 1);

			Y(i, :) = 0;
			YY(idx_label) = YY(idx_label) - 1;

			YF(idx_label, idx_group) = YF(idx_label, idx_group) - 1;
			YF_j = YF(:, idx_group);
			FF_j = FF(idx_group);

			min_cost = 1e+14;
			idx = 0;
			for c = 1:k
				YY(c) = YY(c) + 1;
				YF_j(c) = YF_j(c) + 1;
				YFFF = YF_j/FF_j;
				cost = sum((X(:, i) - center(:, c)).^2) + p1*sum((YFFF - YY/n).^2) + p2*sum(YY.^(-1), 'all');
				YF_j(c) = YF_j(c) - 1;
				YY(c) = YY(c) - 1;

				if (min_cost > cost)
					min_cost = cost;
					idx = c;
				end
			end
			Y(i, idx) = 1;
			YY(idx) = YY(idx) + 1;
			YF(idx, idx_group) = YF(idx, idx_group) + 1;
		end
		[~, label] = max(Y, [], 2);
		center = X*Y./YY';
		[D, new_sumd] = update_dist(X, center, k, label);
		err = abs(sum(sumd - new_sumd));
		sumd = new_sumd;
		iter = iter + 1;
	end

	% if not converge
	if (iter > max_iter && err > 0.001 && any(label ~= last))
		warning("failed to converge in %d iterations", max_iter);
	end
end

function [D, sumd] = update_dist(X, center, k, label)
	D = dist(X', center);
	sumd = zeros(k, 1);
	for i = 1:k        
		sumd(i) = sum(D(label == i, i));
	end
end
