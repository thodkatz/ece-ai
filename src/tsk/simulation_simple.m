clear; close;

% filter I/O
getInput = @(data) data(:, 1:end - 1);
getOutput = @(data) data(:, end);

% prepare data
[trainData, checkData, testData] = prepareData('airfoil_self_noise.dat');

% create fuzzy models
opt = genfisOptions('GridPartition');
opt.InputMembershipFunctionType = 'gaussmf';
opt.OutputMembershipFunctionType = 'constant';
opt.NumMembershipFunctions = 2;
tsk(1) = genfis(getInput(trainData), getOutput(trainData), opt);
opt.NumMembershipFunctions = 3;
tsk(2) = genfis(getInput(trainData), getOutput(trainData), opt);
opt.OutputMembershipFunctionType = 'linear';
opt.NumMembershipFunctions = 2;
tsk(3) = genfis(getInput(trainData), getOutput(trainData), opt);
opt.NumMembershipFunctions = 3;
tsk(4) = genfis(getInput(trainData), getOutput(trainData), opt);

% training
for i = 1:numel(tsk)
    opt = anfisOptions('InitialFIS', tsk(i), 'EpochNumber', 100, 'ValidationData', checkData);
    opt.DisplayANFISInformation = 0;
    opt.DisplayErrorValues = 0;
    opt.DisplayStepSize = 0;
    opt.DisplayFinalResults = 0;
    [trainFis, trainError, ~, checkFis, checkError] = anfis(trainData, opt);

    % membership functions before vs after training
    for j = 1:numel(checkFis.Inputs)
        figure
        subplot(211)
        % before
        [xmf, ymf] = plotmf(tsk(i), 'input', j);
        plot(xmf, ymf); ylabel('Degree of membership', 'Interpreter', 'Latex')
        title('Before', 'Interpreter', 'Latex')
        subplot(212)
        % after training
        [xmf, ymf] = plotmf(checkFis, 'input', j);
        plot(xmf, ymf); xlabel('Input', 'Interpreter', 'Latex'); ylabel('Degree of membership', 'Interpreter', 'Latex')
        title('After', 'Interpreter', 'Latex')
        textTitle = ['$x_{' int2str(j) '}$'];
        sgtitle(textTitle, 'Interpreter', 'Latex')
        %exportgraphics(gcf, [int2str(j) '_tsk3.pdf'],'ContentType','Vector')
    end

    %showrule
    %showrule(checkfis)

    %learning curves
    figure
    plot([trainError checkError]); xlabel('Epochs', 'Interpreter', 'Latex'); ylabel('Error', 'Interpreter', 'Latex'); legend('trainError', 'validationError')
    %exportgraphics(gcf, 'learning_curves_tsk3.pdf','ContentType','Vector')
    
    % evaluation, metrics
    yHat = evalfis(checkFis, getInput(testData));
    y = getOutput(testData);
    error = y - yHat;
    r2 = 1 - sum((y - yHat).^2) / sum((y - mean(y)).^2)
    rmse = sqrt(mse(yHat, getOutput(testData)))
    nmse = 1 - r2
    ndei = sqrt(nmse)

    % a sample of 100 elements of the model trying to fit to the data
    figure
    plot([y(1:100) yHat(1:100)]); xlabel('First 100 Test samples', 'Interpreter', 'Latex'); ylabel('Output', 'Interpreter', 'Latex'); legend('$y$', '$\hat{y}$', 'Interpreter', 'Latex')
    %exportgraphics(gcf, 'prediction_real_tsk3.pdf','ContentType','Vector')

    % prediction errors
    figure
    plot(error); xlabel('Test Samples', 'Interpreter', 'Latex'); ylabel('Error', 'Interpreter', 'Latex')
    %exportgraphics(gcf, 'prediction_real_error_tsk3.pdf','ContentType','Vector')
end
