function [Z,min_w] = subproblem_Z_2_2(earning_iteration_Zi,Z,lamda,use_number,N)
%% 1.初始预备工作
%初始化新Z
Z = [Z, 0];
L_z = length(Z);
%先计算当前Zi降低的代价
Zi_lower_coef = zeros(1,L_z-1);
for t = 1:L_z-1
    Zi_number = Z(1,t); %拿到第t个对应的Zi数
    if Zi_number ~= 0
        Zi_lower_coef(1,t)= earning_iteration_Zi(1,Zi_number)+lamda(1,t);
    else
        Zi_lower_coef(1,t) = 100000; %如果为0，说明不能再减，把代价拉大
    end
end
%% 2.循环迭代
for tt=1:use_number
    %先获取下一步的迭代收益和最小迭代代价
    Judge_Zi = earning_iteration_Zi(1,tt)+lamda(1,L_z);%更新判断标准
    [sorted_values,sorted_indices] = sort(Zi_lower_coef, 'ascend');%升序排序
    sorted_indices1 = sorted_indices(1); %找出当前迭代代价下的最小值索引
    price_min = Zi_lower_coef(1,sorted_indices1);%找出当前迭代代价下的最小值
    %判断是否进行迭代
    if Judge_Zi>price_min
        %更新结果
        Z(1,L_z) = Z(1,L_z)+1;
        Z(1,sorted_indices1) = Z(1,sorted_indices1)-1;
        mm = Z(1,sorted_indices1);
        if Z(1,sorted_indices1)==0
            Zi_lower_coef(1,sorted_indices1) = 1000000;
        else
            Zi_lower_coef(1,sorted_indices1)=earning_iteration_Zi(1,mm)+lamda(1,sorted_indices1);
        end
        %确保每个Z一定≥0
    else
        break
    end
end
min_w = 0;
for t = 1:L_z
    if Z(1,t) ~= 0
        min_w = min_w-(Z(1,t)/(N*use_number))*(log(Z(1,t)/(N*use_number)))+Z(1,t)*lamda(1,t);
    end
end
Z;
min_w;
end