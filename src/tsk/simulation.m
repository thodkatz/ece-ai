clear; close;

% prepare data
[trainData, checkData, testData] = prepareData('datasets/airfoil_self_noise.dat');

% filter I/O
getInput = @(data) data(:, 1:end - 1);
getOutput = @(data) data(:, end);

% create fuzzy models
opt = genfisOptions('GridPartition');
opt.InputMembershipFunctionType = 'gbellmf';

options.OutputMembershipFunctionType = 'constant';
opt.NumMembershipFunctions = 2;
tsk(1) = genfis(getInput(trainData), getOutput(trainData), opt);
opt.NumMembershipFunctions = 3;
tsk(2) = genfis(getInput(trainData), getOutput(trainData), opt);

options.OutputMembershipFunctionType = 'linear';
opt.NumMembershipFunctions = 2;
tsk(3) = genfis(getInput(trainData), getOutput(trainData), opt);
opt.NumMembershipFunctions = 3;
tsk(4) = genfis(getInput(trainData), getOutput(trainData), opt);

% training per model
for i = 1:numel(tsk)
    opt = anfisOptions('InitialFIS', tsk(i), 'EpochNumber', 10, 'ValidationData', checkData);
    % by default I think the weights are initialized randomly. That's why you get different results per run
    % the goal is to avoid bad local minimums
    [trainFis, trainError, ~, checkFis, checkError] = anfis(trainData, opt);

    % membership functions of optimized fuzzy inference system (fis)
    % for j = 1:numel(checkFis.Inputs)
    %     figure
    %     plotmf(checkFis,'input',j)
    % end

    % learning curves
    figure
    plot([trainError checkError])

    % evaluation, metrics
    yHat = evalfis(checkFis, getInput(testData));
    y = getOutput(testData);
    r2 = 1 - sum((y - yHat).^2) / sum((y - mean(y)).^2);
    rmse = sqrt(mse(yHat, getOutput(testData)));
    nmse = 1 - r2;
    ndei = sqrt(nmse);
    error = y - yHat;

    figure
    plot(error)
end

% evaluation

function [trainData, checkData, testData] = prepareData(name_dataset)
    data = load(name_dataset);

    % shuffle
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