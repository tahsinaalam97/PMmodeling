function [predicted,performance] = svmfg(trainingData,testingData)


trainingPredictors = trainingData(:, 1:(end-1));
trainingResponse = trainingData(:,end);

testingPredictors = testingData(:, 1:(end-1));
testingResponse = testingData(:,end);
%actual = vertcat(trainingResponse,testingResponse);

% Train a regression model
% This code specifies all the model options and trains the model.
responseScale = iqr(trainingResponse);
if ~isfinite(responseScale) || responseScale == 0.0
    responseScale = 1.0;
end
boxConstraint = responseScale/1.349;
epsilon = responseScale/13.49;
regressionModel = fitrsvm(...
    trainingPredictors, ...
    trainingResponse, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 0.83, ...
    'BoxConstraint', boxConstraint, ...
    'Epsilon', epsilon, ...
    'Standardize', false);

testingPredictions = predict(regressionModel,testingPredictors);
trainingPredictions = predict(regressionModel,trainingPredictors);
predicted = vertcat(trainingPredictions, testingPredictions);

%Testing%

mdltest = fitlm(testingResponse,testingPredictions);
Ttest= predict(mdltest);
Rtest = sqrt(mdltest.Rsquared.Ordinary);

resultstest = figure('WindowState','maximized');
YMatrix2=[testingPredictions,Ttest];
% Create axes
axes1 = axes('Parent',resultstest);
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(testingResponse,YMatrix2,'Color',[0 0 0]);
set(plot1(1),'DisplayName','Data','Marker','+','LineStyle','none');
%change marker style in each figure
set(plot1(2),'DisplayName','Fit','LineWidth',3);

% Create plot
plot([min(testingResponse):max(testingResponse)],[min(testingResponse):max(testingResponse)],'DisplayName','Y = T','LineStyle','-.','Color',[0 0 0]);

Coefficients = mdltest.Coefficients.Estimate;
% Create ylabel
ylabel("Output = " + Coefficients(2,1) + "*Target + " + Coefficients(1,1));

% Create xlabel
xlabel('Observed PM2.5');


% Create title
title("R = " + Rtest + "(Testing)");

% Uncomment the following line to preserve the X-limits of the axes
%xlim(axes1,[0 1]);
% Uncomment the following line to preserve the Y-limits of the axes
%ylim(axes1,[0 1]);
box(axes1,'on');
hold(axes1,'off');
% Set the remaining axes properties
set(axes1,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 2]);
% Create legend
legend2 = legend(axes1,'show');
set(legend2,...
    'Position',[0.341752074847953 0.822324630973648 0.0592972175006811 0.0814132082480622]);

rmsetest = sqrt(mean((testingResponse - testingPredictions).^2));
maetest1 = mae(testingResponse-testingPredictions);
performance =[ Rtest , Rtest*Rtest , rmsetest, maetest1 ];
performance =array2table(performance);
performance.Properties.VariableNames = [ "R" , "Rsquare" , "RMSE" , "MAE"];