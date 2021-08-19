clear
clc

GP = tf(25,[1 10.1 1])
controlSystemDesigner(GP)
% With Control System Toolbox, interactively we found the configuration of the PI controler to satisfy our design requirements