clear; close

% filter I/O
getInput = @(data) data(:, 1:end - 1);
getOutput = @(data) data(:, end);

% how to split data and cross validation:
% https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
% https://stackoverflow.com/questions/49160206/does-gridsearchcv-perform-cross-validation

% 60-20-20 split (training,validation,testing)
[trainData, checkData, testData] = prepareData('superconductivity.csv');

% grid search
numFeatures = [5 10 15 20];
radius = [0.3 0.45 0.55 0.65 0.85];
meanError = zeros(numel(numFeatures), numel(radius));

% feature selection
numNeighbors = 10;
ranks = relieff(getInput(trainData), getOutput(trainData), numNeighbors);

% k-fold cross validation split
k = 5;
rng(10);
out = getOutput(trainData);
cv = cvpartition(numel(out), 'Kfold', k);

%% Hyperparameter tuning cross validation - grid search
for i = 1:numel(numFeatures)
    for j = 1:numel(radius)
        for kth = 1:k
            trainIdx = find(cv.training(kth) == 1);
            checkIdx = find(cv.test(kth) == 1);
            trainFilt = [trainData(trainIdx, ranks(1:numFeatures(i))) out(trainIdx)];
            checkFilt = [trainData(checkIdx, ranks(1:numFeatures(i))) out(checkIdx)];

            % create fuzzy inference system
            opt = genfisOptions('SubtractiveClustering');
            opt.ClusterInfluenceRange = radius(j);
            fis = genfis(getInput(trainFilt), getOutput(trainFilt), opt);
            fprintf("Fold: %d, NumFeature: %d, Radius: %0.2f, Rules: %d",kth, numFeatures(i), radius(j),numel(fis.Rules))

            % train model
            opt = anfisOptions('InitialFIS', fis, 'EpochNumber', 100, 'ValidationData', checkFilt);
            opt.DisplayANFISInformation = 0;
            opt.DisplayErrorValues = 0;
            opt.DisplayStepSize = 0;
            opt.DisplayFinalResults = 0;
            [trainFis, trainError, ~, checkFis, checkError] = anfis(trainFilt, opt);

            meanError(i, j) = meanError(i, j) + mean(checkError);
            fprintf(", meanError: %0.5f\n",mean(checkError));
        end
        fprintf("Total meanError: %0.5f\n",meanError(i,j));
    end
end

% k-fold mean error
meanError = meanError ./ 5

%% Plots

% stem3, meanerror
figure
surf(radius, numFeatures, meanError); xlabel('radius','Interpreter','Latex'); ylabel('number of features','Interpreter','Latex')
zlabel('error','Interpreter','Latex')

%% Final model

% pick the optimal hyperparameters
[optimalNumFeaturesIdx, optimalRadiusIdx] = find(meanError == min(meanError(:)));
optimalNumFeatures = numFeatures(optimalNumFeaturesIdx)
optimalRadius = radius(optimalRadiusIdx)

% train for the best hyperparameters (radius, numFeatures)
trainOptimal = [trainData(:, ranks(1:optimalNumFeatures)) getOutput(trainData)];
checkOptimal = [checkData(:, ranks(1:optimalNumFeatures)) getOutput(checkData)];
testOptimal = [testData(:, ranks(1:optimalNumFeatures)) getOutput(testData)];

% create fuzzy inference system
opt = genfisOptions('SubtractiveClustering');
opt.ClusterInfluenceRange = optimalRadius;
fisOptimal = genfis(getInput(trainOptimal), getOutput(trainOptimal), opt);
fprintf("Rules: %d\n",numel(fisOptimal.Rules))

% train model
opt = anfisOptions('InitialFIS', fisOptimal, 'EpochNumber', 100, 'ValidationData', checkOptimal);
[trainFis, trainError, ~, checkFis, checkError] = anfis(trainOptimal, opt);

% evaluation, metrics
yHat = evalfis(checkFis, getInput(testOptimal));
y = getOutput(testOptimal);
error = y - yHat;
r2 = 1 - sum((y - yHat).^2) / sum((y - mean(y)).^2)
rmse = sqrt(mse(yHat, getOutput(testOptimal)))
nmse = 1 - r2
ndei = sqrt(nmse)

%% Plots

% learning curves
figure
plot([trainError checkError]); xlabel('Epochs','Interpreter','Latex'); ylabel('Error','Interpreter','Latex'); legend('trainError','validationError')

%% membership functions before vs after training
for j = 1:numel(fisOptimal.Inputs)
    figure
    subplot(211)
    % before
    [xmf,ymf] = plotmf(fisOptimal, 'input', j);
    plot(xmf,ymf); ylabel('Degree of membership','Interpreter','Latex')
    title('Before','Interpreter','Latex')
    subplot(212)
    % after training
    [xmf,ymf] = plotmf(checkFis, 'input', j);
    plot(xmf,ymf); xlabel('Input', 'Interpreter', 'Latex'); ylabel('Degree of membership','Interpreter','Latex')
    title('After','Interpreter','Latex')
    textTitle = ['$x_{' int2str(j) '}$'];
    sgtitle(textTitle,'Interpreter','Latex')
end

%% a sample of 100 elements of the model trying to fit to the data
figure
plot([y(1:100) yHat(1:100)]); xlabel('First 100 Test samples','Interpreter','Latex'); ylabel('Output','Interpreter','Latex'); legend('$y$','$\hat{y}$','Interpreter','Latex')

% prediction errors
figure
plot(error); xlabel('Test Samples','Interpreter','Latex'); ylabel('Error','Interpreter','Latex')
