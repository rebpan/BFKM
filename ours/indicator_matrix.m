function H = indicator_matrix(y)
	uq = unique(y);
	c = length(uq);
	H = zeros(length(y), c);
	for i = 1:c
		H(y==uq(i), i) = 1;
	end
end
