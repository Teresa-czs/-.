function [x_result_SP,ubd_value_SP] = solve_subproblem_new_Ru(A,lamda,k,indices_R0,Z_value)
[i,p] = size(A);
%记录索引

%生成系数
x_result_SP = zeros(i,p); %初始化x矩阵
A = A - lamda; %两个矩阵相减构成新矩阵,一定是正数
A(indices_R0) = -10000000  ;%取0，其他正常


%%遍历每列j，找到每列对应前K个最大系数aij
for m = 1:p
    [sorted_values,sorted_indices] = sort(A(:,m), 'descend');%降序排序
    sorted_indices1 = sorted_indices(1:k); % 找到k个最大索引
    x_result_SP(sorted_indices1,m) = 1 ;
end
ubd_value_SP = sum(sum(A.*x_result_SP)) + sum(lamda.*Z_value);

