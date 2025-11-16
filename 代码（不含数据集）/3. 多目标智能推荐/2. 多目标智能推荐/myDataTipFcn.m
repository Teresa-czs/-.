function txt = myDataTipFcn(~, event)
    % 获取数据点的位置
    pos = event.Position;
    iteration = pos(1);
    bestBound = pos(2);
    
    fig = ancestor(event.Target, 'figure');
    
    Cumulative_time = getappdata(fig, 'Cumulative_time');
    
    % 获取对应的累计时间
    if iteration >=1 && iteration <= length(Cumulative_time)
        elapsedTime = Cumulative_time(iteration);
    else
        elapsedTime = NaN;
    end
    
    % 创建提示文本
    txt = {['迭代次数: ', num2str(iteration)], ...
           ['最优上界: ', num2str(bestBound)], ...
           ['累计时间: ', num2str(elapsedTime, '%.2f'), ' 秒']};
end
