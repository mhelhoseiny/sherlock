%% Tutorial for colAUC Function and Package
% *By Jarek Tuszynski*
% 
% Function colAUC calculates Area under ROC curve (AUC) for a vector or for 
% each column of a matrix. 
%
% The main properties of this code: 
%
% * Ability to work with multi-dimensional data.
%
% * Ability to work with multi-class datasets.
%
% * Speed - this code was written to calculate AUC's for large number of 
%    features, fast.
%
% * Two different algorithms are provided one based on integrating ROC
%    curves and one based on Wilcoxon Rank Sum Test aka. Mann-Whitney U
%    Test.
%
% * Function can be used to plot ROC curves.
%
% See http://en.wikipedia.org/wiki/Receiver_operating_characteristic
%
%% Change History
% * 2005 - original code written in MATLAB and R. Only R version released 
%          at http://cran.r-project.org/web/packages/caTools
% * 2011-02-15 - Release of MATLAB version

%% Licence
% The package is distributed under BSD License
format compact; % viewing preference
clear variables;
type('license.txt')

%% Read in test file
% Iris flower data set collected by Edgar Anderson to quantify 
% the geographic variation of Iris flowers in the Gaspé Peninsula.
% http://en.wikipedia.org/wiki/Iris_flower_data_set
[~, ~, iris] = xlsread('iris.csv'); 
X = cell2mat(iris(2:end, 1:end-1));
y = iris(2:end, end);
colLabel = iris(1, 1:end-1);

%% Convert it into 2-class label
msk = strcmp(y,y{1});
y2 = y;
y2( msk) = y(1);
y2(~msk) = {['not ' y{1}]};

%% Calculate 2-class AUC using by integration of ROC 
auc=colAUC(X,y2);
out = [colLabel; num2cell(auc)];
disp(out); 

%% Calculate 2-class AUC using "Wilcoxon rank sum test"
auc=colAUC(X,y2,'Wilcoxon');
out = [colLabel; num2cell(auc)];
disp(out); 

%% Plot ROC curves for all the features 
colAUC(X,y2)
legend(colLabel, 'Location', 'SouthEast');
title('ROC curves separating Setosa from other species of iris flowers')
xlabel('True Positive Rate');
ylabel('False Positive Rate');

%% Calculate 3-class AUC using by integration of ROC 
[auc lab]=colAUC(X,y);
rowLabel = textscan(sprintf('%s vs. %s|', lab{:,1}, lab{:,2}), '%s', 'Delimiter', '|');
out = cellfun(@num2str,num2cell(auc),'UniformOutput', false);
out = [[{''}; rowLabel{1}], [colLabel; out]];
format long; disp(out); format short;

%% Calculate 3-class AUC using "Wilcoxon rank sum test"
auc=colAUC(X,y,'Wilcoxon');
out = cellfun(@num2str,num2cell(auc),'UniformOutput', false);
out = [[{''}; rowLabel{1}], [colLabel; out]];
format long; disp(out); format short;

%% Plot 3-class ROC curves for feature #1 
colAUC(X(:,1),y)
title('ROC curves separating species of iris flowers using sepal length')
xlabel('True Positive Rate');
ylabel('False Positive Rate');

%% Test on a dataset with 100 samples and 100,000 features
n = 100;
X = rand(n,100000); % randomly distributed samples
y = ((1:n)>n/2);    % half of the samples split into 2 categories
tic; auc1=colAUC(X,y,'ROC'); toc;
hist(auc1,50);

%% Same calculation using Wilcoxon algorithm
tic; auc2=colAUC(X,y,'Wilcoxon'); toc;

%% Make sure results are the same
disp(max(auc1-auc2));

%% Look at 2 best features
% With enough features one can usually find a set which can be used to
% separate 2 classes, even when all the numbers are random.
[~, idx] = sort(auc1, 'descend');
scatter(X(:,idx(1)), X(:,idx(2)),[],y,'filled');

%% Show avrRank function
x = round(rand(1,10)*5);
r = avrrank(x);
disp([x;r])
