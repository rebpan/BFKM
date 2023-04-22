function fairness = calcBal(label, Color, c)
%Calculate balance by checking fairness in each of the clusters.
%Fair algorithms for clustering, 2019, Bera, Suman, et al.
n = length(label);
[unique_labels, n_clusters] = unique_count(label);
nu_labels = length(unique_labels);
if nu_labels < c
    fairness = 0;
    return
end
[unique_sens, n_sens] = unique_count(Color);
nu_sens = length(unique_sens);
r_sens = n_sens/n;

curr_b = zeros(1, nu_labels);
cl_b = zeros(1, nu_sens);
for i = 1:nu_labels % cluster i
    for j = 1:nu_sens
        IDX = find(label==unique_labels(i));
        IDX1 = find(Color==unique_sens(j));
        r = length(intersect(IDX, IDX1)) / n_clusters(i);
        
        if r == 0
            continue
        else
            cl_b(j) = min(r_sens(j)/r, r/r_sens(j));
        end
    end
    curr_b(i) = min(cl_b);
    cl_b = zeros(1, nu_sens);
end
fairness = curr_b;
end


function [unique_a, cnt_unique] = unique_count(a)
unique_a = unique(a);
cnt_unique = [];
for i = 1:length(unique_a)
    IDX = a == unique_a(i);
    cnt_unique = [cnt_unique, sum(IDX)];
end
end