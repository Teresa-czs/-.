function [Z,min_w] = subproblem_Z_2_1_main(N,lamda,earning_iteration_Zi,A)
[movie_number,use_number] = size(A);
% 初始化 Z1 到 ZN 的值为 U
Z = use_number * ones(1, N);  %初始化Z
    % 从第 N+1 件商品开始，逐步优化 Z
for i = N+1:movie_number
    i
    [Z,min_w] = subproblem_Z_2_2(earning_iteration_Zi,Z,lamda,use_number,N);
end
Z;
min_w;
end