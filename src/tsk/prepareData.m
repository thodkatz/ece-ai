function [trainData, checkData, testData] = prepareData(name_dataset)
    data = load(name_dataset);

    % shuffle
    rng(2);
    shuffle = @(v) v(randperm(length(v)), :);
    data = shuffle(data);

    % split
    idxTrain = round(0.6 * length(data));
    idxCheck = round(0.8 * length(data));
    trainData = data(1:idxTrain, :);
    checkData = data(idxTrain + 1:idxCheck, :);
    testData = data(idxCheck + 1:end, :);

    % normalize min-max input
    % trainData(:, 1:end - 1) = normalize(trainData(:, 1:end - 1), 'range');
    % checkData(:, 1:end - 1) = normalize(checkData(:, 1:end - 1), 'range');
    % testData(:, 1:end - 1) = normalize(testData(:, 1:end - 1), 'range');

    % normalization unit hypercube
    trnX = trainData(:, 1:end - 1);
    chkX = checkData(:, 1:end - 1);
    tstX = testData(:, 1:end - 1);
    xmin = min(trnX, [], 1);
    xmax = max(trnX, [], 1);
    trnX = (trnX - repmat(xmin, [length(trnX) 1])) ./ (repmat(xmax, [length(trnX) 1]) - repmat(xmin, [length(trnX) 1]));
    chkX = (chkX - repmat(xmin, [length(chkX) 1])) ./ (repmat(xmax, [length(chkX) 1]) - repmat(xmin, [length(chkX) 1]));
    tstX = (tstX - repmat(xmin, [length(tstX) 1])) ./ (repmat(xmax, [length(tstX) 1]) - repmat(xmin, [length(tstX) 1]));
    trainData(:, 1:end - 1) = trnX;
    checkData(:, 1:end - 1) = chkX;
    testData(:, 1:end - 1) = tstX;
end
