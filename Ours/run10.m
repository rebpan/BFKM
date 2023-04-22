BAL_list = zeros(1,10)
AW_list = zeros(1, 10)
DI_list = zeros(1, 10)
SSE_list = zeros(1, 10)
for t = 1:10
demo
BAL_list(t) = min(BAL)
AW_list(t) = AW
DI_list(t) = DI
SSE_list(t) = SSE
t = t + 1
end
mean(BAL_list)
mean(AW_list)
mean(DI_list)
mean(SSE_list)