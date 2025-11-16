function [x_result_SP,ubd_value_SP] = solve_subproblem_x_z(A,lamda,k,indices_R0,z)
[i,p] = size(A);
lamda2 = repmat(lamda,1,p);
x_result_SP = zeros(i,p); %初始化x矩阵
B_new = A + lamda2; %两个矩阵相减构成新矩阵,一定是正数
%找到符合最低得分的索引


B_new(indices_R0) = 0  ;%取0，其他正常


%%遍历每列j，找到每列对应前K个最大系数aij
for m = 1:p
    [sorted_values,sorted_indices] = sort(B_new(:,m), 'descend');%降序排序
    sorted_indices1 = sorted_indices(1:k); % 找到k个最大索引
    x_result_SP(sorted_indices1,m) = 1 ;
end
ubd_value_SP = sum(sum(B_new.*x_result_SP)) - sum(lamda*z);

