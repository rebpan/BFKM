function B = label2binary(A)
uq = unique(A);
c = length(uq);
B = zeros(length(A), c);
for i = 1:c
    B((A==uq(i)), i) = 1;
end
end