function [Z,min_w] = subproblem_Z_2_new(earning_iteration_Zi,Z,lamda,use_number,N)
%% 1.初始化工作  
% 初始化新Z  
global percent
L_z = length(Z);  
% 先计算当前Zi降低的代价  
Zi_price_coef = zeros(1, L_z);  
for t = 1:L_z  
    Zi_number = Z(1, t); % 获取到第t个对应的指标  
    if Zi_number ~= 0  
        Zi_price_coef(1, t) = percent*earning_iteration_Zi(1, Zi_number) +(1-percent) *lamda(1, t);  
    else  
        Zi_price_coef(1, t) = 100000; % 如果为0，说明不能再减，把代价拉大  
    end  
end  
%% 计算当前迭代收益  
Zi_benefit_coef = zeros(1, L_z);  
for t = 1:L_z  
    Zi_number = Z(1, t); % 获取到第t个对应的指标  
    if Zi_number ~= use_number  
        Zi_benefit_coef(1, t) =  percent*earning_iteration_Zi(1, Zi_number+1) +(1-percent) * lamda(1, t);  
    else  
        Zi_benefit_coef(1, t) = -100000; % 如果是use_number ，说明不能再增，把收益拉低  
    end  
end  

%% 找到最小代价和最大收益  
[minValue, minIndex] = min(Zi_price_coef);  
[maxValue, maxIndex] = max(Zi_benefit_coef);  

while maxValue > minValue  
    % 更新降低的变量  
    Z(minIndex) = Z(minIndex) - 1; % 更新对应指标  
    bb = Z(minIndex);  
    Zi_benefit_coef(minIndex) =  percent*earning_iteration_Zi(1, bb + 1) +(1-percent) * lamda(1, minIndex); % 更新降低后的收益  
    % 更新的变量  
    Z(maxIndex) = Z(maxIndex) + 1;   
    cc = Z(maxIndex);  
    Zi_price_coef(maxIndex) = percent*earning_iteration_Zi(1, cc) +(1-percent) * lamda(1, maxIndex);  
    %%当降到0时，不能再降----赋值代价为10000，保证不是最小代价
    if Z(minIndex) == 0  
        Zi_price_coef(minIndex) = 100000; % 保证不是最小价格  
    else  
        Zi_price_coef(minIndex) = percent*earning_iteration_Zi(1, bb) + (1-percent) *lamda(1, minIndex);  
    end 
    % 当升到U时，不能再升----赋值收益为-10000，保证不是最大收益  
    if Z(maxIndex) == use_number
        Zi_benefit_coef(maxIndex) = -100000; % 保证不是最大收益  
    else  
        Zi_benefit_coef(maxIndex) = percent*earning_iteration_Zi(1, cc + 1) +(1-percent) * lamda(1, maxIndex);  
    end 
    % 重新计算最大收益和最小代价  
    [minValue, minIndex] = min(Zi_price_coef);  
    [maxValue, maxIndex] = max(Zi_benefit_coef);  
end  
min_w = 0;
for t = 1:L_z
    if Z(1,t) ~= 0
        min_w = min_w-percent*(Z(1,t)/(N*use_number))*(log(Z(1,t)/(N*use_number)))+(1-percent) *Z(1,t)*lamda(1,t);
    end
end
Z;
min_w;
end