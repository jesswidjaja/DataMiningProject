---
# Classification to Predict Bank Marketing Success  
- Jesslyn Widjaja
- IN PROGRESS

## Abstract

In this project, classification is utilized to predict the success of telemarketing calls in selling bank long-term deposits. To classify whether or not each call led to a deposit, I analyzed 15 features relating to bank clients using 2 different models: logistic regression and decision trees. Overall, decision trees proved to be the best in determining telemarketing success with an 89% accuracy rate.        

## Introduction

Banks use direct marketing to target a group of potential customers and meet a defined goal, one of which is to secure long-term deposits. However, telemarketing can prove to be a hassle for both the bank and its potential clients, such as when the bank's calls are constantly being ignored or clients receive multiple calls advertising a service they have no interest in. To remedy the situation, I aim to build a model that will predict whether telemarketing calls lead to long-term deposits and maximize the chances of reaching the right clients. Through my results, I hope to figure out which features make a person more likely to subscribe to long-term deposits or whether the bank should try another marketing strategy. 

I focus on analyzing a data set of telemarketing calls made by a Portuguese retail bank taken from the UCI Machine Learning Repository. This data set includes 4500 calls and 17 attributes, including categorical variables, such as the type of contact communication and the last month of contact, and numerical variables, such as the number of days after the client was last contacted. Most of these attributes will be used to predict y, the deposit subscription status, through classifying whether a call led to a "yes" (subscription) or "no" (no subscription).

Two models were built using the R programming language in the course of this study, the first being logistic regression. I used logistic regression to fit all the attributes and gauge how good of a model this would be in predicting subscription status. Overall, the model had an accuracy rate of around 90%, but upon closer inspection, I realized that it was much better at predicting the chance that a client would not subscribe as opposed to the chance that a client will subscribe. Because it has a low rate of predicting subscriptions, I then use the ROC curve to improve its rate. This leads to a lower prediction accuracy overall, but the probabilities of correctly predicting subscriptions and non-subscriptions each are fairly high at around 80%. The last model I use is a decision tree, and after pruning, it has an 89% accuracy rate for predicting subscription status overall. Because it has the highest accuracy rate and selects specific features that create more accurate predictions, the decision tree is concluded to be the best model to use to predict whether telemarketing calls lead to long-term deposits.         
## Data Analysis

Preprocessing

The data set provided by the UCI Machine Learning Repository is in the format of an Excel spreadsheet. To read the data into R, I use the code below and check the structure of the data set.     
```{r}
bank <- read.csv("bank/bank.csv", header=T)
str(bank) 
```
From the output, it is seen that there are 7 numerical variables and 9 categorical variables, excluding the variable that represents the subscription status, y, which will be used as the response variable the models are attempting to predict. 

Since there appear to be no missing values, I proceed to create a few visualizations to explore the data set and gain some initial insights.

## Explanatory Data Analysis

For numerical variables such as the age of the client, I use boxplots to visualize the effect they might have on subscribing to a long-term deposit. These will enable us to best see the distribution of clients and whether this will differ between those who subscribed and those who did not. 

```{r}
library(ggplot2)
qplot(bank$y, bank$age, data=bank, geom="boxplot", xlab="Subscription Status", ylab="Age")
```
From this boxplot, I can observe that 50% of those who declined to subscribe to a long-term deposit fall around 32-48 years of age whereas 50% of those who subscribed fall around 31-50 years of age. The distribution of age is slightly larger for subscriptions than non-subscriptions, but they are otherwise similar to each other with a similar median age of about 40.

```{r}
qplot(bank$y, bank$day, data=bank, geom="boxplot", xlab="Subscription Status", ylab="Day of Month Contacted")
```
Subscription status based on day of month contacted shows us that the distribution is slightly higher for those who subscribed than those who did not. 50% of subscriptions lie between 9th-24th day of the month while 50% of non-subscriptions lie between 7th-22nd day of the month. Because the ranges for subscriptions and non-subscriptions overlap, these boxplots  do not necessarily tell us which days would be better to contact clients. Thus, I proceed to model the data to get a better glimpse of what might lead to client subscription.

## Modeling the Data

Before building the models, I first make training and test sets to run them on. The library caret is used to split 75% of the observations into the training set and the remaining 25% into the test set.

```{r}
library(caret)
#set random seed for the results to be reproducible
set.seed(123) 
# sample 75% of observations as the training set
trainX <- createDataPartition(bank$y,p=0.75,list=FALSE) 
train <- bank[trainX,]
# the rest 25% as the test set
test <- bank[-trainX,]
```

## Logistic Regression

My goal in this analysis is to build a model that will predict whether or not telemarketing calls lead potential clients to subscribe to a long-term deposit at the Portuguese bank. Because this is a classification problem and the subscription status is a binary variable dependent on the rest of the attributes, I initially decide to use logistic regression to create a model. 

It is stated by the researchers who have collected the data that duration, which corresponds to the 11th attribute in the data set, highly affects the outcome and should be discarded in order to create a realistic predictive model. Due to this reason, I will remove duration from the model.

```{r}
bank.fit <- glm(y~.,data=train[,-11], family=binomial) #take away 11th column
summary(bank.fit)
```

From the above results, blue collar job, housing, loan, unknown contact, campaign, and poutcome are statistically significant at level 0.05. Interpreting the change in coefficient for categorical variables such as blue collar job, housing, loan, and unknown contact will not be useful because these variables are not quantitative. 

However, for every one unit change in campaign (the number of times a client was contacted), the log odds of subscribing to a long-term deposit decreases by 0.08179. For every one unit change in poutcomesuccess (successful outcome of previous marketing campaign), the log odds of subscribing increases by 2.061. Lastly, for every unit change in poutcomeunknown (unknown outcome of previous marketing campaign), the log odds of subscribing decreases by 0.694. This might be because an unknown outcome might have actually been a negative outcome, but I cannot make this claim with certainty.

I next want to gauge how good the logistic regression model is at predicting the probabilities of subscribing. To do so, I construct a confusion matrix on the training set. 

```{r}
# specify type="response" to get the estimated probabilities
prob <- round(predict(bank.fit,type="response"), digits=3)

# store the predicted labels using 0.5 as a threshold
library(dplyr)
train = train %>%
  mutate(predy=as.factor(ifelse(prob<=0.5, "No", "Yes")))
# confusion matrix
table(pred=train$predy, true=train$y)
```
Out of 3391 cases in total, the model classifies 3062 correctly, which is 90.3%. This seems to be an excellent rate of accuracy, but taking a closer look at the results shows otherwise. Out of the 3000 calls that did not lead to a subscription to a term deposit, the model classifies 2943 correctly (98.1%) - still a very high rate of accuracy. The problem, however, lies in predicting the clients who have subscribed. Out of the 391 clients who subscribed to a term deposit, the model only classifies 119 correctly (30.43%).

```{r}
Prob <- round(predict(bank.fit, test, type="response"),digits=3)
test = test %>%
  mutate(Probability=Prob)
test
```
Overall, the probabilities of subscribing to a long-term deposit seem to be very low, but this might be because the accuracy rate of predicting whether a client subscribes is low. Thus, I use the ROC curve to increase the probability of determining accurate results for subscriptions.

```{r}
#ROC curve because low true positive rate
library(ROCR)
pred = prediction(prob, train$y)
#TPR on the y axis and FPR on the x axis
perf = performance(pred, measure="tpr", x.measure="fpr")
plot(perf, col=2, lwd=3, main="ROC curve")
abline(0,1)
```

```{r}
auc <- performance(pred, "auc")@y.values
auc
```
The area under the curve (AUC) is used to summarize the performance of the model, and it is currently about 0.8832. This is a pretty high value, which is an indicator of a good performance. However, its performance can be better. As observed from the graph of the ROC curve, the optimum for the true positive rate is 0.82. Our current true positive rate, or the rate of accurately predicting subscriptions, is 0.304, which is much lower than what it can be.

To increase the true positive rate, I change the threshold values by ensuring that the false positive (fpr) and false negative rates (fnr) are as small as possible. Specifically, I will choose the probability threshold that leads to the shortest distance from (fpr,fnr) to (0,0). I calculate the Euclidean distance to find this distance.  

```{r}
# FPR
fpr = performance(pred, "fpr")@y.values[[1]]
cutoff = performance(pred, "fpr")@x.values[[1]]
# FNR
fnr = performance(pred,"fnr")@y.values[[1]]

#plot the FPR and FNR versus threshold values
matplot(cutoff, cbind(fpr,fnr), type="l",lwd=2, xlab="Threshold",ylab="Error Rate")
legend(0.4, 1, legend=c("False Positive Rate","False Negative Rate"),
       col=c(1,2), lty=c(1,2))

# calculate the euclidean distance between (fpr,fnr) and (0,0)
rate = as.data.frame(cbind(Cutoff=cutoff, FPR=fpr, FNR=fnr))
rate$distance = sqrt((rate[,2])^2+(rate[,3])^2)

# select the probability threshold with the smallest euclidean distance
index = which.min(rate$distance)
best = rate$Cutoff[index]
best
abline(v=best, col=3, lty=3, lwd=3)
```

Therefore, our best cutoff value is 0.099 which corresponds to the smallest Euclidean distance of 0.20. That being said, assigning subscription probabilities less than 0.099 to No and higher than 0.099 to Yes makes (fpr,fnr) and (0,0) closest.

I then calculate the subscription probabilites again using the threshold value to see if there was an improvement.

```{r}
#calculate subscription probabilities again with threshold value
new.train = train %>%
  mutate(predy=as.factor(ifelse(prob<=0.099, "No", "Yes")))
# confusion matrix
table(pred=new.train$predy, true=train$y)
```

Out of 3391 cases, the model classifies 2686 correctly (79.2%), which is lower than the previous accuracy rate at about 90%. However, out of the 3000 not subscribed to a term deposit, the model classifies 2360 correctly (78.67%) and out of 391 subscribed to a term deposit, 326 calls are classified correctly (83.38%). We see that this is a huge increase in the true positive rate and a much more accurate model overall.


## Decision Trees

The next model I will use to classify the bank's calls into subscriptions or non-subscriptions is the decision tree. Among its advantages are that it can implicitly perform feature selection, nonlinear relationships between the parameters - which we might have here due to the categorical variables - will not affect the performance of the tree, and it is beneficial to use for classification. All of the above prove to be enough reasons to give decision trees a try, so I proceed to do so in the following steps. 

```{r}
bank <- bank[,-11] 

library(ISLR)
library(tree)

tree.bank = tree(y~.-poutcome, data=bank)
summary(tree.bank)

plot(tree.bank)
text(tree.bank, pretty=0, cex=1, col="purple")
title("Classification Tree")
```
The output provided by the summary shows that the misclassification error rate is 0.1086, meaning that the model will incorrectly classify 10.86% - or correctly classify 89% - of the calls. In comparison to the previously improved logistic regression model, which classified 79.2% of the calls correctly, this is significantly better. The plot also gives us a visualization of which features it used to create this more accurate model, which includes unknown contact, an age younger than 60, and less than 99 days since the client was last contacted.

I will next create training and test sets to test the decision trees on. As previously stated, the library caret will be used to split 75% of the data set into the training set and the remaining 25% into the test set.

```{r}
library(caret)
#set random seed for the results to be reproducible
set.seed(123) 
# sample 75% of observations as the training set
trainX <- createDataPartition(bank$y,p=0.75,list=FALSE) 
train <- bank[trainX,]
# the rest 25% as the test set
test <- bank[-trainX,]

y.test <- test$y
```

The model curated by the decision tree will then be fitted on the training set and used to predict on the test set.  

```{r}
# fit model on training set
tree.bank = tree(y~.-poutcome, data = train)
# predict on test set
tree.pred = predict(tree.bank, test, type="class")
# get confusion matrix
error = table(tree.pred, y.test)
error
#accuracy rate
accuracy = sum(diag(error))/sum(error)
accuracy
```

This approach leads to correct predictions for around 88% of the calls in the test data set, which is a significantly high accuracy rate. To see if the model still has room for improvement, I prune the decision tree.

### Pruning the Decision Tree

I determine the best size by using cross validation, which will use the classification error rate to guide the pruning process. The best size will thus be the size that minimizes this error rate, and this is shown through the code below.

```{r}
#determine best size using cross validation
set.seed(123)

cv.bank <- cv.tree(tree.bank, FUN=prune.tree, K=10, method="misclass")
names(cv.bank)
class(cv.bank)

best.size = cv.bank$size[which.min(cv.bank$dev)]
cv.bank

par(mfrow=c(1,1))
prune.bank = prune.misclass(tree.bank, best=best.size)
plot(prune.bank)
text(prune.bank, pretty=0, col = "blue", cex = .8)
title("Pruned Tree")
```

As seen from the above visualization, the best tree is includes 5 nodes and the features of unknown contact, an age younger than 60, 1 time of contact before this telemarketing campaign, and not having been contacted less than 185 days since being last contacted. This provides a very clear outlook of what the bank should do to maximize their chances of reaching clients interested in subscribing, and I then use the decision tree to predict the calls from the test set to see if it is just as accurate.

```{r}
# predict on test set
prune.pred = predict(prune.bank, test, type="class")
# get confusion matrix
error = table(prune.pred, y.test)
error
#accuracy rate
accuracy = sum(diag(error))/sum(error)
accuracy
```

The pruned decision tree leads to 89% of correct predictions, which is a bit more accurate than the original decision tree at 88%. It is also the most accurate prediction and takes the least amount of computing time out of all the models, so this is the model I will ultimately choose to determine telemarketing success.   

## Conclusion

In conclusion, the decision tree ultimately led me to find a satisfactory model to predict whether the telemarketing calls from the bank lead to a client subscribing to a long-term deposit. It also identified which features would more likely lead to a subscription, such as unknown contact, an age younger than 60, 1 time of contact before this telemarketing campaign, and not having been contacted less than 185 days since being last contacted. Because of this specific set of potential criteria, the Portuguese retail bank can then focus more of its telemarketing efforts on clients with these attributes. 

For further study, I can also utilize a random forest or logistic regression given the features selected from the decision tree to see if they gives a higher probability of correctly predicting subscriptions and non-subscriptions. 

Last but certainly not least, I would like to give my thanks to Professor Sang-Yun Oh and Kevin Zou for their time and resources in helping me to complete this project.

## Works Cited

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

Rickert, Joseph. "Using Caret to Compare Models." R-bloggers. Foundation for Open Access Statistics, 01 June 2016. Web. 10 Mar. 2017.


