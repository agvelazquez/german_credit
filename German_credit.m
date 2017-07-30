clc;
clear all;
close all;

%{ 
Analisis para determinar si una persona va a hacer capaz de devolver o no
un credito
Se va a utilizar regresion logistica ya que la respuesta y es una variable
categorica binaria
Response:
y = 1 is considered a Good  credit risk
y = 0 is considered a Bad credit risk
There are 1000 loan applicants
%}
%% DATA
M = csvread('csvnew.csv',1,0);
[num,text,raw] = xlsread('german_credit_headers.xlsx');
X = M(:,2:end);
Xcategorical = [X(:,1) X(:,3:4) X(:,6:12) X(:,14:end)];
Xcontinuos = [X(:,2) X(:,5) X(:,13)];
textp = text';
Xcategoricallabels = [textp(:,2) textp(:,4:5) textp(:,7:13) textp(:,15:end)]';
Y = M(:,1);

%% EXPLORATORY DATA ANALYSIS
%Categeorical predictors
% 1 - No missing values
% 2 - Histograms
figure
a = subplot(1,3,1);
hist(X(:,1))
xlabel(a, 'Account Balance')
xlim([1 4])
a1 = subplot(1,3,2);
hist(X(:,12))
xlabel(a1, 'Most valuable available asset')
a2 = subplot(1,3,3);
hist(X(:,3))
xlabel(a2, 'Payment Status of Previous Credit')
%{ 
Most of the clients:
        -Ask for credits below 50 months
        -Their account balance is mostly above 200 DM
        -And their other credits are paid up
%}
figure
a = subplot(1,3,1);
hist(X(:,4))
xlabel(a, 'Purpose')
%xlim([1 4])
a1 = subplot(1,3,2);
hist(X(:,7))
xlabel(a1, 'Length of current employment')
a2 = subplot(1,3,3);
hist(X(:,6))
xlabel(a2, 'Value savings/stocks')
%{
Most of the clients:
        -Ask for credits for used or new cars mainly 
        -Ask for credits between 1 to 4 years of employement or up to 7
        -Don't have any stocks or savings
It is not continued the analisys of the predictors because it isn't the
pourpose of the work
%}
%% EXPLORATORY DATA ANALYSIS
%Categeorical predictors
%Boxplot for categorical predictors
figure
boxplot(Xcategorical(:,4),Y)
ylabel('Value Savings/Stocks')
%Most of the rejected don't have savings but there are some outliers with
%great accounts
figure
boxplot(Xcategorical(:,7),Y)
ylabel('Sex & Marital Status')
%There are not any kind of diference with the sex and marital status
%% EXPLORATORY DATA ANALYSIS
%Distribution of the continuos variables

%With a log transformation of the continuos variables that are normal
%skewed we achive a normalization of the variables 
figure
b1 = subplot(1,2,1);
nbins = 50;
hist(log(X(:,5)),nbins);
xlabel('Credit Amount');
ylabel('Frequency');
b2 = subplot(1,2,2);
hist(log(X(:,2)),nbins/3); %Al ser menos cantidad de categorias es mejor una cantidad de bins mayor
xlabel('Duration of Credit (month)');
ylabel('Frequency');

figure
h1 = subplot(1,2,1);
normplot(log(X(:,5)));
h2 = subplot(1,2,2);
normplot(log(X(:,2)));
xlim([0 5]);
%Looking at normal plots it is seen a marked skewness from normal
%distribution

%% EXPLORATORY DATA ANALYSIS
%Correlation matrix
r = corrcoef(X);
hmo = HeatMap(r,'Colormap',redbluecmap, 'COLUMNLABELS',text(2:end),'ROWLABELS',text(2:end));
%It exist an important correlation between Amount of the credit and
%duration. It leads future researchs for a posible multicollineality
%problem. But there isn't a big problem with this fact
%% PEARSON CHI SQUEARED TEST CATEGORICAL PREDICTORS

pvalueCP = zeros(size(Xcategorical,2),1);

for i = 1:size(Xcategorical,2);
    [toble,chi2,p] = crosstab(Xcategorical(:,i),Y);
    pvalueCP(i) = p;
end

Xscatter=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17];
figure
scatter(Xscatter,pvalueCP);
%set(gca,'xtick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17])
%set(gca,'xticklabel',Xcategoricallabels)
%set(gca,'XTickLabelRotation',45)
ylabel('pvalue')
hold on
hline = refline([0 0.1]);
hline.Color = 'r';
table(Xscatter',pvalueCP,'RowNames',Xcategoricallabels)
gname(Xcategoricallabels) %I can name the dots in the scatter plot

XselectedColumns = find(pvalueCP<0.1); %indices of columns selected as predictors from Xcategorical
Xselected = [Xcategorical(:,1:5) Xcategorical(:,7:8) Xcategorical(:,10:12) Xcategorical(:,17)];
%% CORRELATION CONTINUOUS VARIABLES

pvalueCV = zeros(3,1);

for i=1:3;
    mdl = fitlm(Xcontinuos(:,i),Y);
    pvalueCV(i) = mdl.Coefficients.pValue(2);
end

%% LOADING TRAINING DATA

training = xlsread ('Training50xls.xlsx','(A3:U502)');
Ytraining = training(:,1);
%the selection of the data is based in the results of the chi-square test
%Queda afuera: Installment, Duration in current adress, Concurrent
%credits,Occupation, no of dependants and telephone.
%
Xslected_training = [training(:,2) training(:,4:5) training(:,7:8) training(:,10:11) training(:,13) training(:,16:17) training(:,21)];
Xcontinuos_training = [training(:,3) training(:,6) training(:,14)];
%% PREDICTION
% Logistic regression TRAINING
%We make a simple partition 50:50
LogisticMDL = GeneralizedLinearModel.fit([Xslected_training Xcontinuos_training],Ytraining,'CategoricalVars',logical([1 1 1 1 1 1 1 1 1 1 1 0 0 0]) ,'Distribution','binomial')
%Given the p-value we have to select which variables to remove (but we have
%to remove all the dummy variables)
%We select the variables in the excel sheet "Logistic Training"
%% Second Training
% We train again the model only we the variables selected
training2 = xlsread ('Training2.xlsx','(A3:L502)');
Ytraining2 = training2(:,1);
Xtraining2 = training2(:,2:end);
LogisticMDL2 = GeneralizedLinearModel.fit(Xtraining2,Ytraining2,'CategoricalVars',logical([1 1 1 1 1 1 1 1 1 0 0]) ,'Distribution','binomial')
%% Final model trained
% Logistic regression Training
training3 = xlsread('Training3.xlsx','(A3:J502)');
Ytraining3 = training3(:,1);
Xtraining3 = training3(:,2:end);
LogisticMDLtest = GeneralizedLinearModel.fit(Xtraining3,Ytraining3,'CategoricalVars',logical([1 1 1 1 1 1 1 0 0]) ,'Distribution','binomial')

%% Testing
test = xlsread('Test50.xlsx', '(A3:J502)');
Xtest = test(:,2:end);
ypred = predict(LogisticMDLtest,Xtest);
