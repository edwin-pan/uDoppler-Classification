function  helperPlotTrainData(trainData,trainLabel,T,F)
%helperPlotTrainData plots the training data sample from each category

%   Copyright 2019 The MathWorks, Inc.
    
    indexToPlot = [4,2,1,22,66]; % indexes of examples in data for each category
    
    figure
    for ii = 1:length(indexToPlot)
        subplot(2,3,ii)
        imagesc(T,F,trainData(:,:,:,indexToPlot(ii)))
        title(trainLabel(indexToPlot(ii)))
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        axis square xy    
    end
end