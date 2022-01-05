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
radius = [0.25 0.45 0.65 0.85];
meanError = zeros(numel(numFeatures), numel(radius));

% feature selection
numNeighbors = 10;
ranks = relieff(getInput(trainData), getOutput(trainData), numNeighbors);

% k-fold cross validation split
k = 5;
rng(10);
out = getOutput(trainData);
cv = cvpartition(numel(out), 'Kfold', k);

% training 5-fold cross validation grid search
for i = 1:numel(numFeatures)
    for j = 1:numel(radius)
        for kth = 1:k
            trainIdx = find(cv.training(kth) == 1);
            checkIdx = find(cv.test(kth) == 1);
            trainFilt = [trainData(trainIdx, ranks(1:numFeatures(i))) out(trainIdx)];
            checkFilt = [trainData(checkIdx, ranks(1:numFeatures(i))) out(checkIdx)];

            % create fizzy inference system
            opt = genfisOptions('SubtractiveClustering');
            opt.ClusterInfluenceRange = radius(j);
            fis = genfis(getInput(trainFilt), getOutput(trainFilt), opt);

            % train model
            opt = anfisOptions('InitialFIS', fis, 'EpochNumber', 10, 'ValidationData', checkFilt);
            [trainFis, trainError, ~, checkFis, checkError] = anfis(trainFilt, opt);

            meanError(i, j) = meanError(i, j) + mean(checkError);
        end
    end
end

% k-fold mean error
meanError = meanError ./ 5;

% stem3, meanerror
figure
stem3(radius,numFeatures,meanError)


% pick the optimal hyperparameters
[optimalNumFeaturesIdx, optimalRadiusIdx] = find(meanError = min(meanError(:)));
optimalNumFeatures = numFeatures(optimalNumFeaturesIdx);
optimalRadius = radius(optimalRadiusIdx);

% train for the best hyperparameters (radius, numFeatures)
trainOptimal = [trainData(:, ranks(1:optimalNumFeatures)) getOutput(trainData)];
checkOptimal = [checkData(:, ranks(1:optimalNumFeatures)) getOutput(checkData)];
testOptimal = [testData(:, ranks(1:optimalNumFeatures)) getOutput(testData)];

% create fuzzy inference system
opt = genfisOptions('SubtractiveClustering');
opt.ClusteringInflenceRange = optimalRadius;
fisOptimal = genfis(getInput(trainOptimal), getOutput(trainOptimal), opt);

% train model
opt = anfisOptions('InitialFIS', fisOptimal, 'EpochNumber', 10, 'ValidationData', checkOptimal);
[trainFis, trainError, ~, checkFis, checkError] = anfis(trainOptimal, opt);

% evaluation, metrics
yHat = evalfis(checkFis, getInput(testOptimal));
y = getOutput(testOptimal);
r2 = 1 - sum((y - yHat).^2) / sum((y - mean(y)).^2);
rmse = sqrt(mse(yHat, getOutput(testOptimal)));
nmse = 1 - r2;
ndei = sqrt(nmse);
error = y - yHat;

% plots
figure
plot(error)

function [trainData, checkData, testData] = prepareData(name_dataset)
    data = load(name_dataset);

    % shuffle
    rng(10);
    shuffle = @(v) v(randperm(length(v)), :);
    data = shuffle(data);

    % split
    idxTrain = round(0.6 * length(data));
    idxCheck = round(0.8 * length(data));
    trainData = data(1:idxTrain, :);
    checkData = data(idxTrain + 1:idxCheck, :);
    testData = data(idxCheck + 1:end, :);

    % normalize min-max input
    trainData(:, 1:end - 1) = normalize(trainData(:, 1:end - 1), 'range');
    checkData(:, 1:end - 1) = normalize(checkData(:, 1:end - 1), 'range');
    testData(:, 1:end - 1) = normalize(testData(:, 1:end - 1), 'range');
end