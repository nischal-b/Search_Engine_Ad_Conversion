# loading sources
source("Documents/4th_Sem/Biz_Intel/BabsonAnalytics.R")

# loading libraries
library(readr)
library(gmodels)
library(Hmisc)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

# loading data
df<-read.csv("Documents/Data_Set.csv")
df1 = df

# managing data
df1$Clicks = as.logical(df1$Clicks)
df1$Query.Search.Date.Time = NULL
df1$Query.Phrase = NULL
df1$Keyword.Entered.In.Auction.Process = NULL
df1$Path.Contains.Advertiser. = as.factor(df$Path.Contains.Advertiser.)
df1$Path.Contains.Competitor. = as.factor(df$Path.Contains.Competitor.)
#df1$User.Search.Path.Number = NULL
#df1$Order.Of.User.Search.In.Path.From.End.Point = NULL
df1$Is.Account.Competitor. = as.factor(df$Is.Account.Competitor.)
df$Delivered.Match.Type = as.factor(df$Delivered.Match.Type)
df1$Impressions = NULL
#df1$Keyword.Bid.Amount = NULL
df1$Served.Ad.Position = as.factor(df1$Served.Ad.Position)
df1$User.Device = as.factor(df1$User.Device)
#df1$User.Device = NULL

# slicing data
N = nrow(df1)
trainingSize = round(N*0.8)
trainingCases = sample(N, trainingSize)
training = df1[trainingCases, ]
test = df1[-trainingCases, ]

## Logistic Regression Begin

# build model
model = glm(Clicks ~ ., data=training, family=binomial)
model = step(model)
summary(model)

# predictions
pred = predict(model, test, type="response")
predTF = pred > 0.5

# error rate
error_logistic = sum(predTF!=test$Clicks)/nrow(test)
CrossTable(predTF, test$Clicks, expected = F, prop.r = F, prop.c = F, prop.t = F, prop.chisq = F)

error_bench_log = benchmarkErrorRate(training$Clicks, test$Clicks)

# charts
ROCChart(test$Clicks, pred)
liftChart(test$Clicks, pred)

## Logistic Regression End

## Classification Trees Begin

# manage
df2 = df1
df2$Clicks = as.factor(df2$Clicks)

# slice
N = nrow(df2)

trainingSize = round(N * 0.8)
trainingCases = sample(N,trainingSize)
training = df2[trainingCases,]
test = df2[-trainingCases,]

# pruning
stoppingRules = rpart.control(minsplit = 2, minbucket = 1, cp = 0)
overfit = rpart(Clicks~., data = training, control = stoppingRules)
pruned = easyPrune(overfit)
rpart.plot(pruned)

# predictions
pred = predict(pruned, test, type = "class")
error_tree = sum(pred!=test$Clicks)/nrow(test)
error_bench_tree = benchmarkErrorRate(training$Clicks,test$Clicks)

## Classification Trees End

## Ensembling Begin

rf = randomForest(Clicks ~ ., data= training, ntree=500)
pred_rf = predict(rf, test)
pred_rf = (pred_rf > 0.5)

error_rf = sum(pred_rf!=test$Clicks)/nrow(test)

## Ensembling End