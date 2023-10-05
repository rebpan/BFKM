function SSE = calc_SSE(X, label)
	Y = indicator_matrix(label);
	YY = diag(Y'*Y);
	C = X*Y./YY';
	SSE = sum((X-C*Y').^2, 'all');
end
