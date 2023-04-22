function [F, v] = SensCNVT(Color)
n = length(Color);
% converting sensitive to a vector with entries in [h] and building F %%%
sens_unique=unique(Color);
h = length(sens_unique);
sens_unique=reshape(sens_unique,[1,h]);

sensitiveNEW=Color;

temp=1;
for ell=sens_unique
    sensitiveNEW(Color==ell)=temp;
    temp=temp+1;
end
    
F=zeros(n,h-1);
v = [];

for ell=1:h
    temp=(sensitiveNEW == ell);
    F(temp,ell)=1; 
    groupSize = sum(temp);
    v = [v;groupSize/n];
end
end