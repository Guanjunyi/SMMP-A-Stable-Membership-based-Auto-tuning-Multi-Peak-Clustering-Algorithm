
% The code was written by Junyi Guan in 2021.
% Please kindly cite the paper Junyi Guan, Sheng li, Xiongxiong He, Jinhui Zhu, Jiajia Chen, and Peng Si
% SMMP: A Stable-Membership-based Auto-tuning Multi-Peak Clustering Algorithm
% IEEE TPAMI,2022,Doi:10.1109/TPAMI.2022.3213574

clear;close all;clc;
%% load dataset
load data/D31
data_with_lable = D31;
%% deduplicate data
data_x = unique(data_with_lable,'rows');
if size(data_x,1) ~= size(data_with_lable,1)
    data_with_lable = data_x;
end
lable = data_with_lable(:,end);
data = data_with_lable(:,1:end-1);
%% SMMP clustering
[CL,C,runtime] = SMMP(data);
%% evaluation
[AMI,ARI,FMI] = Evaluation(CL,lable);
%% show result
resultshow(data,CL);
%% clustering result
result = struct;
result.C = C;
result.AMI = AMI;
result.ARI = ARI;
result.FMI = FMI;
result.runtime = runtime;
result
