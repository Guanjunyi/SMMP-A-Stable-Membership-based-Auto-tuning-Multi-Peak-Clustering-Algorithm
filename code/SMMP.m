% The code was written by Junyi Guan in 2021.
% Please kindly cite the paper Junyi Guan, Sheng li, Xiongxiong He, Jinhui Zhu, Jiajia Chen, and Peng Si
% SMMP: A Stable-Membership-based Auto-tuning Multi-Peak Clustering Algorithm
% IEEE TPAMI,2022,Doi:10.1109/TPAMI.2022.3213574

function [CL,NC,runtime] = SMMP(data,NC_input) % If you have prior knowledge, you can directly enter the cluster number 'NC_input'
close all;
eta = 0.1; %% used to determine the length of the similarity message vector
fprintf('SMMP Clustering :)!\n');
isshowresult = 1;
%% normalization
data=(data-min(data))./(max(data)-min(data));
data(isnan(data))=0;
tic;
%% fast search of KNN matrix based on kd-tree (when dimension is not large than 10)
[n,d]  = size(data);
if n>200
    max_k = ceil(sqrt(n));
else
    max_k = max(15,round(n/10));
end
if d<=11
    [knn,knn_dist] = knnsearch(data,data,'k',max_k*2);
else
    dist = pdist2(data,data,'euclidean');
    [knn_dist,knn] = sort(dist,2);
end

%% adaptive tuning of parameter k
[k] = adaptive_tuning_k(knn,knn_dist,max_k);

%% seting of parameter k_b for our border link detection
k_b = min(round(k/2),2*floor(log(n)));

%% denisty estimation
rho = k*sum(knn_dist(:,2:k).^1,2).^-1; %% within-surrounding-similarity-based density w.r.t 'k'

%% identify density peaks and calculate center-representativeness
theta = ones(n,1); %theta(i): the center-representativeness of point 'i' (initialization)
descendant = zeros(n,1); % descendant(i): the descendant node number of point 'i' (initialization)
[~,OrdRho]=sort(rho,'descend');
for i=1:n
    for j=2:k
        neigh=knn(OrdRho(i),j);
        if(rho(OrdRho(i))<rho(neigh))
            NPN(OrdRho(i))=neigh;%% NPN:neigbor-based parent node, i.e., nearest higher density point within the KNN area.
            theta(OrdRho(i)) = theta(neigh)* rho(OrdRho(i))/rho(neigh);
            descendant(neigh) = descendant(neigh)+1;
            break
        end
    end
end
pk = find(theta==1);%% find density peaks (i.e., sub-cluster centers)
n_pk = length(pk);%% the number of density peaks

%% generate sub-clsuters
sl=-1*ones(n,1); %% sl: sub-labels of points.
sl(pk) = (1:n_pk); %% give unique sub-labels to density peaks.
for i=1:n
    if (sl(OrdRho(i))==-1)
        sl(OrdRho(i))=sl(NPN(OrdRho(i)));%% inherit sub-labels from NPN
    end
end
for i = 1:n_pk
    child_sub= descendant(sl==i);
    edge(i) = length(find(child_sub==0)); %% edge(i): the edge number of sub-cluster 'i'
end

%% obtain cross-cluster border pairs
borderpair = obtain_borderpairs(sl,k_b,knn,knn_dist);

%% obtain border links
blink = obtain_borderlinks(borderpair);

%% if there is no border link, output sub-clustering result
if isempty(blink)
    CL = sl';
    NC = n_pk;
    runtime = toc;
    %% show result
    if isshowresult
        resultshow(data,CL);
    end
    return
end

%% else, calculate representativeness of border links for the similarity estimation between subclusters
n_blink = size(blink,1);
simimesgs = cell(n_pk,n_pk); %smeg(i,j): a set of all similarity messages bewteen density peak 'i' and 'j'
for i = 1:n_blink
    ii = blink(i,1);
    jj = blink(i,2);
    pk1 = sl(ii);
    pk2 = sl(jj);
    smesgs = simimesgs(pk1,pk2);
    smesgs{1} = [smesgs{1};(theta(ii)+theta(jj))/2];
    simimesgs(pk1,pk2) = smesgs;
    simimesgs(pk2,pk1) = smesgs;
end

%% similarity estimation between subclusters
sim = zeros(n_pk,n_pk);
sim_list = [];
for pk1=1:n_pk-1
    for pk2 =pk1+1:n_pk
        smesgs = simimesgs(pk1,pk2);
        smesgs = smesgs{:};
        max_smesg = max(smesgs);
        min_n_smesg = ceil(min(edge(pk1),edge(pk2))*eta); %% min_n_smesg: the minimum standard number of similarity message samples
        smesgs = sort([smesgs;zeros(min_n_smesg,1)],'descend');
        smesgs = smesgs(1:min_n_smesg);
        if max_smesg>0
            Gamma = mean(abs(smesgs - max_smesg))/max_smesg; %%
            sim(pk1,pk2) = max_smesg*(1-Gamma);
            sim(pk2,pk1) = max_smesg*(1-Gamma);
        end
        sim_list = [sim_list sim(pk1,pk2)];
    end
end

%% Single-linkage clustering of sub-clusters according to SIM
SingleLink = linkage(1-sim_list,'single');
if nargin >= 2 %% case: the number of cluster 'NC' is a priori
    NC = NC_input;
else
    bata = [0;SingleLink(:,end)];
    bata(bata<0)=0;
    bataratio = [[n_pk+1-(1:n_pk-1)]' diff(bata)];
    bataratio = sortrows(bataratio,2,'descend');
    NC  = bataratio(1,1); %% the stable number of cluster that with maximum bata-interval.
end
CL_pk = cluster(SingleLink,NC);

%% assign final cluster label
for i=1:n_pk
    CL(sl==i) = CL_pk(i); %% CL: the cluster label
end
runtime = toc;

fprintf('Finished!!!!\n');


function [k] = adaptive_tuning_k(knn,knn_dist,max_k)
n = size(knn,1);
n_k = zeros(n,1);  %% n_k: the number of different 'k' that satisfy the number of desnity peaks 'n_pk'
k_sum = zeros(n,1);  %% k_sum : the sum of different 'k' that satisfy the number of desnity peaks 'n_pk'
for cur_k = 2:ceil(max_k/20):max_k %% ceil(kmax/20): run about 20 times (to reduce computation time)
    cur_rho = cur_k*sum(knn_dist(:,2:cur_k).^1,2).^-1; %% within-surrounding-similarity-based density w.r.t 'cur_k'
    ispk = ones(n,1); %% density peak label
    for i=1:n
        for j=2:cur_k
            if cur_rho(i)< cur_rho(knn(i,j))
                ispk(i)=0; %% not a density peak
                break
            end
        end
    end
    n_pk = length(find(ispk==1)); %% n_pk: the number of denisty peak w.r.t 'cur_k'
    n_k(n_pk) = n_k(n_pk)+1;
    k_sum(n_pk) = k_sum(n_pk) + cur_k;
end
stb_n_pk = find(n_k==max(n_k)); %%stb_n_pk: the stable number of density peaks that with the maximum k-interval.
stb_n_pk = stb_n_pk(1);
k = round((k_sum(stb_n_pk)/n_k(stb_n_pk))); %% obtain our parameter $k$

function [borderpair] = obtain_borderpairs(sl,k_b,knn,knn_dist)
borderpair = [];
n = length(sl);
for i=1:n
    label_i = sl(i);
    for j = 2:k_b
        i_nei = knn(i,j);
        dist_i_nei = knn_dist(i,j);
        label_nei = sl(i_nei);
        if label_i ~= label_nei & find(knn(i_nei,2:k_b)==i)
            borderpair = [borderpair;[i i_nei dist_i_nei]];
            break
        end
    end
end

function [blink] = obtain_borderlinks(borderpair)
if isempty(borderpair)
    blink = [];
else
    borderpair(:,1:2) = sort(borderpair(:,1:2),2);
    [~,index] = unique(borderpair(:,3));
    borderpair = borderpair(index,:);
    borderpair = sortrows(borderpair,3);
    n_pairs = size(borderpair,1);
    blink = []; %% blink: border link
    for i = 1:n_pairs
        bp = borderpair(i,1:2);
        if isempty(intersect(bp,blink))
            blink = [blink;bp];
        end
    end
end





