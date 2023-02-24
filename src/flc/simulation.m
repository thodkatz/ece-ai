clear; clc

% plant transfer function
%GP = tf(25, [1, 10.1, 1])
GP = zpk([], [-1 -9], 10)

% setup path
rootDir = fileparts(matlab.desktop.editor.getActiveFilename); % FIX: this requires to run within matlab desktop editor
addpath([rootDir '/simulink'])
addpath(rootDir)

%% LINEAR CONTROLLER

% controlSystemDesigner(GP)
% With Control System Toolbox, interactively we found the configuration of
% the PI "k*(s+c)/s" controler to satisfy our design requirements:
% 1) Rise time < 0.6 seconds
% 2) Overshoot < 8%

% The configuration is saved to "dcControllerTuning.mat"
% Uncomment the following line to launch the toolbox with the configuration:
% controlSystemDesigner("dcControllerTuning.mat")

c = 1.5;
k = 1;

% controller transfer function
%GC = tf([k * 1, k * c], [1, 0])
GC = zpk(-c, 0, k)

% closed loop transfer function
closedTf = feedback(GP * GC, 1)

% plots
figure; step(closedTf)
figure; rlocus(closedTf)

%% FUZZY CONTROLLER

% Design fuzzy controller
fis = mamfis('AndMethod', "min", 'ImplicationMethod', "min", 'AggregationMethod', "max", 'DefuzzificationMethod', "centroid");

numMfs = 9;
fis = addInput(fis, [-1 1], 'Name', "e", 'NumMFs', numMfs, 'MFType', "trimf");
fis = addInput(fis, [-1 1], 'Name', "De", 'NumMFs', numMfs, 'MFType', "trimf");
mfNames = ["NV", "NL", "NM", "NS", "ZR", "PS", "PM", "PL", "PV"];

% I/O fuzzy variables and membership functions
middleNode = -1;
leftNode = middleNode - 1/4;
rightNode = middleNode + 1/4;

for index = 1:numel(mfNames)
    fis.Input(1).MembershipFunctions(index).Name = mfNames(index);
    fis.Input(1).MembershipFunctions(index).Parameters = [leftNode middleNode rightNode];
    fis.Input(2).MembershipFunctions(index).Name = mfNames(index);
    fis.Input(2).MembershipFunctions(index).Parameters = [leftNode middleNode rightNode];

    middleNode = middleNode + 1/4;
    leftNode = middleNode - 1/4;
    rightNode = middleNode + 1/4;
end

fis = addOutput(fis, [-1 1], 'Name', "Du", 'NumMFs', numMfs, 'MFType', "trimf");
zeroIdxOutputMf = 5;

middleNode = -1;
leftNode = middleNode - 1/4;
rightNode = middleNode + 1/4;

for index = 1:numel(mfNames)
    fis.Output(1).MembershipFunctions(index).Name = mfNames(index);
    fis.Output(1).MembershipFunctions(index).Parameters = [leftNode middleNode rightNode];

    middleNode = middleNode + 1/4;
    leftNode = middleNode - 1/4;
    rightNode = middleNode + 1/4;
end

numPoints = 5000;
figure
subplot(311); plotmf(fis, 'input', 1, numPoints); title("Input e")
subplot(312); plotmf(fis, 'input', 2, numPoints); title("Input De")
subplot(313); plotmf(fis, 'output', 1, numPoints); title("Output Du")
sgtitle("Membership functions I/O")

% Rule base
rowDim = 9;
colDim = 9;
rules = strings(rowDim, colDim);

% iterate over array diagonally and fill the matrix based on Mc Vicar and Wheelan template
numDiagonals = 9;

for idxDiag = 1:numDiagonals
    rowIdx = idxDiag;
    for j = 1:numDiagonals - idxDiag + 1
        if idxDiag == 1
            rules(rowIdx, j) = sprintf("De==%s & e==%s => Du=%s", mfNames(numMfs - rowIdx + 1), mfNames(j), "ZR");
            %rules(rowIdx,j) = sprintf("%s,%s,%s", mfNames(numMfs - rowIdx + 1), mfNames(j),"ZR");
        elseif idxDiag <= zeroIdxOutputMf - 1
            rules(rowIdx, j) = sprintf("De==%s & e==%s => Du=%s", mfNames(numMfs - rowIdx + 1), mfNames(j), mfNames(zeroIdxOutputMf - idxDiag + 1));
            %rules(rowIdx, j) = sprintf("%s,%s,%s", mfNames(numMfs - rowIdx + 1), mfNames(j), mfNames(zeroIdxOutputMf - idxDiag + 1));
            
            rules(j, rowIdx) = sprintf("De==%s & e==%s => Du=%s", mfNames(numMfs - j + 1), mfNames(rowIdx), mfNames(zeroIdxOutputMf + idxDiag - 1));
            %rules(j, rowIdx) = sprintf("%s,%s,%s", mfNames(numMfs - j + 1), mfNames(rowIdx), mfNames(zeroIdxOutputMf + idxDiag - 1));
        else
            rules(rowIdx, j) = sprintf("De==%s & e==%s => Du=%s", mfNames(numMfs - rowIdx + 1), mfNames(j), "NV");
            %rules(rowIdx, j) = sprintf("%s,%s,%s", mfNames(numMfs - rowIdx + 1), mfNames(j), "NV");

            rules(j, rowIdx) = sprintf("De==%s & e==%s => Du=%s", mfNames(numMfs - j + 1), mfNames(rowIdx), "PV");
            %rules(j, rowIdx) = sprintf("%s,%s,%s", mfNames(numMfs - j + 1), mfNames(rowIdx), "PV");
        end
        rowIdx = rowIdx + 1;
    end
end

fis = addRule(fis, rules(:));
writeFIS(fis, "flc")
gensurf(fis)

% Uncomment these lines for GUI representation of fis object, fuzzy inference system
%plotfis(fis)
%ruleview(fis)
%fuzzy(fis)
%showrule(fis)

%% Simulink closed loop unity negative feedback system
open_system("simulink_simulation.slx")