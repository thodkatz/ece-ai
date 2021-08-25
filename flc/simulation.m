clear; clc

% plant transfer function
%GP = tf(25, [1, 10.1, 1])
GP = zpk([], [-0.1 -10], 25)

%% LINEAR CONTROLLER

% controlSystemDesigner(GP)
% With Control System Toolbox, interactively we found the configuration of
% the PI "k*(s+c)/s" controler to satisfy our design requirements:
% 1) Rise time < 0.6 seconds
% 2) Overshoot < 8%

% The configuration is saved to "dcControllerTuning.mat"
% Uncomment the following line to launch the toolbox with the configuration:
% controlSystemDesigner("dcControllerTuning.mat")

c = 0.2;
k = 2;

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

fis = addInput(fis, [-1 1], 'Name', "e", 'NumMFs', 7, 'MFType', "trimf");
fis = addInput(fis, [-1 1], 'Name', "De", 'NumMFs', 7, 'MFType', "trimf");
inputMfNames = ["NL", "NM", "NS", "ZR", "PS", "PM", "PL"];

middleNode = -1;
leftNode   = middleNode-0.5;
rightNode  = middleNode+0.5;
for index = 1:numel(inputMfNames)
    fis.Input(1).MembershipFunctions(index).Name = inputMfNames(index);
    fis.Input(1).MembershipFunctions(index).Parameters = [leftNode middleNode rightNode];
    fis.Input(2).MembershipFunctions(index).Name = inputMfNames(index);
    fis.Input(2).MembershipFunctions(index).Parameters = [leftNode middleNode rightNode];

    middleNode = middleNode + 1/3;
    leftNode   = middleNode-0.5;
    rightNode  = middleNode+0.5;
end

fis = addOutput(fis, [-1 1], 'Name', "Du", 'NumMFs', 9, 'MFType', "trimf");
outputMfNames = ["NV" inputMfNames "PV"];
zeroIdxOutputMf = 5;

middleNode = -1;
leftNode   = middleNode-3/8;
rightNode  = middleNode+3/8;
for index = 1:numel(outputMfNames)
    fis.Output(1).MembershipFunctions(index).Name = outputMfNames(index);
    fis.Output(1).MembershipFunctions(index).Parameters = [leftNode middleNode rightNode];

    middleNode = middleNode + 1/4;
    leftNode   = middleNode-3/8;
    rightNode  = middleNode+3/8;
end

figure
subplot(311); plotmf(fis, 'input', 1); title("Input e")
subplot(312); plotmf(fis, 'input', 2); title("Input De")
subplot(313); plotmf(fis, 'output', 1); title("Output Du")
sgtitle("Membership functions I/O")

% Rule base
rowDim = 7;
colDim = 7;
rules = strings(rowDim, colDim);

% iterate over array diagonally and fill the matrix based on Mc Vicar and Wheelan template
numDiagonals = 7;

for idxDiag = 1:numDiagonals
    rowIdx = idxDiag;

    for j = 1:numDiagonals - idxDiag+1

        if idxDiag == 1
            rules(rowIdx,j) = sprintf("e==%s & De==%s => Du=%s", inputMfNames(rowIdx), inputMfNames(j),"ZR");
        elseif idxDiag <= zeroIdxOutputMf-1
            rules(rowIdx, j) = sprintf("e==%s & De==%s => Du=%s", inputMfNames(rowIdx), inputMfNames(j), outputMfNames(zeroIdxOutputMf - idxDiag + 1));
            rules(j, rowIdx) = sprintf("e==%s & De==%s => Du=%s", inputMfNames(j), inputMfNames(rowIdx), outputMfNames(zeroIdxOutputMf + idxDiag - 1));
        else
            rules(rowIdx, j) = sprintf("e==%s & De==%s => Du=%s", inputMfNames(rowIdx), inputMfNames(j), "NV");
            rules(j, rowIdx) = sprintf("e==%s & De==%s => Du=%s", inputMfNames(j), inputMfNames(rowIdx), "PV");
        end

        rowIdx = rowIdx + 1;
    end

end

fis = addRule(fis,rules(:));
writeFIS(fis,"flc")

% GUI representation of fis object
% plotfis(fis)
% fuzzy(fis)

% Simulink closed loop unity negative feedback system
