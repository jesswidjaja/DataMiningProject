Sys.getenv("PATH")
Sys.setenv(PATH="C:/Program Files/MiKTeX 2.9/miktex/bin/x64/",sep=";")
bank <- read.csv("C:/Users/xiao cucu/Documents/classes - ucsb/PStat 131/bank/bank.csv", header=T)
str(bank)

#check if there are missing values - since there are none, proceed
summary(bank)

#make training and test sets - used caret package because the usual way was giving me a vector for the training set, which is not what I wanted

library(caret)
#set random seed for the results to be reproducible
set.seed(123) 
# sample 75% of observations as the training set
trainX <- createDataPartition(bank$y,p=0.75,list=FALSE) 
train <- bank[trainX,]
# the rest 25% as the test set
test <- bank[-trainX,]

qplot(bank$y, bank$age, data=bank, geom="boxplot")

qplot(bank$y, bank$balance, data=bank, geom="boxplot", ylim=c(0,30000))
qplot(bank$y, bank$balance, data=bank, geom="boxplot") #orig

qplot(bank$y, bank$day, data=bank, geom="boxplot")

qplot(bank$y, bank$duration, data=bank, geom="boxplot")

qplot(bank$y, bank$campaign, data=bank, geom="boxplot") #minor difference

qplot(bank$y, bank$pdays, data=bank, geom="boxplot", ylim=c(0,1000)) #diff
qplot(bank$y, bank$pdays, data=bank, geom="boxplot") #orig

qplot(bank$y, bank$previous, data=bank, geom="boxplot") #bad visualization

prev <- table(bank$previous, bank$y)
barplot(prev, main="Subscription Based on Previous Number of Contacts Performed",
        xlab="Previous Number of Contacts", col=c("purple"),
        beside=TRUE, ylim=c(0,3500))
#People more likely to say no than yes overall, which is expected.
#But there are fewer people who declined to subscribe the more they were contacted
#and fewer people who subscribed the more they were contacted - which does not reveal any useful insight.


#Logistic Regression
bank.fit <- glm(y~.,data=train[,-11], family=binomial) #take away 11th column
summary(bank.fit)

# specify type="response" to get the estimated probabilities
prob <- round(predict(bank.fit,type="response"), digits=3)

# store the predicted labels using 0.5 as a threshold
library(dplyr)
train = train %>%
  mutate(predy=as.factor(ifelse(prob<=0.5, "No", "Yes")))
# confusion matrix
table(pred=train$predy, true=train$y)
#Out of 3391 cases, model classifies 3062 correctly (90.3%)
#Out of 3000 not subscribed to a term deposit, model classifies 2943 correctly (98.1%)
#Out of 391 subscribed to a term deposit, model classifies 119 correctly (30.43%)

Prob <- round(predict(bank.fit, test, type="response"),digits=3)
test = test %>%
  mutate(Probability=Prob)
test

#ROC curve because we see that there's small probability of determining accurate results for subscriptions
#low true positive rate
library(ROCR)
pred = prediction(prob, train$y)
# we want TPR on the y axis and FPR on the x axis
perf = performance(pred, measure="tpr", x.measure="fpr")
plot(perf, col=2, lwd=3, main="ROC curve")
abline(0,1)

auc <- performance(pred, "auc")@y.values
auc

#optimum for TPR is 0.82. to increase true pos rate, we have to change threshold values
# FPR
fpr = performance(pred, "fpr")@y.values[[1]]
cutoff = performance(pred, "fpr")@x.values[[1]]
# FNR
fnr = performance(pred,"fnr")@y.values[[1]]

# use matplot to plot the FPR and FNR versus threshold values
matplot(cutoff, cbind(fpr,fnr), type="l",lwd=2, xlab="Threshold",ylab="Error Rate")
# add legend to the plot
legend(0.4, 1, legend=c("False Positive Rate","False Negative Rate"),
       col=c(1,2), lty=c(1,2))

# calculate the euclidean distance between (FPR,FNR) and (0,0)
rate = as.data.frame(cbind(Cutoff=cutoff, FPR=fpr, FNR=fnr))
rate$distance = sqrt((rate[,2])^2+(rate[,3])^2)

# select the probability threshold with the smallest euclidean distance
index = which.min(rate$distance)
best = rate$Cutoff[index]
best
abline(v=best, col=3, lty=3, lwd=3)
#Therefore, our best cutoff value is 0.099 which corresponds to the smallest Euclidean distance 0.20.
#That being said, assigning hiring probabilities less than 0.099 to No and higher than 0.099 to Yes makes
#(FPR, FNR) and (0, 0) closest.

#calculate subscription probabilities again with threshold value
new.train = train %>%
  mutate(predy=as.factor(ifelse(prob<=0.099, "No", "Yes")))
# confusion matrix
table(pred=new.train$predy, true=train$y)
#Out of 3391 cases, model classifies 2686 correctly (79.2%)
#Out of 3000 not subscribed to a term deposit, model classifies 2360 correctly (78.67%)
#Out of 391 subscribed to a term deposit, model classifies 326 correctly (83.38%) 
#We see that this is a huge increase in true pos rate and a much more accurate model overall.

#Decision Trees
bank <- bank[,-11] #move this to the very beginning? & make appropriate changes to training set

library(ISLR)
library(tree)

tree.bank = tree(y~.-poutcome, data=bank)
summary(tree.bank)

plot(tree.bank)
text(tree.bank, pretty=0, cex=1, col="purple")
title("Classification Tree")

#make training and test sets - used caret package because the usual way was giving me a vector for the training set, which is not what I wanted

library(caret)
#set random seed for the results to be reproducible
set.seed(123) 
# sample 75% of observations as the training set
trainX <- createDataPartition(bank$y,p=0.75,list=FALSE) 
train <- bank[trainX,]
# the rest 25% as the test set
test <- bank[-trainX,]

y.test <- test$y

# fit model on training set
tree.bank = tree(y~.-poutcome, data = train)
# predict on test set
tree.pred = predict(tree.bank, test, type="class")
# get confusion matrix
error = table(tree.pred, y.test)
error
#accuracy rate
accuracy = sum(diag(error))/sum(error)


#This approach leads to correct predictions for around 88% of the locations in the test data set.

#Pruning Tree
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

#make training and test sets - used caret package because the usual way was giving me a vector for the training set, which is not what I wanted
# predict on test set
prune.pred = predict(prune.bank, test, type="class")
# get confusion matrix
error = table(prune.pred, y.test)
error
#accuracy rate
accuracy = sum(diag(error))/sum(error)
accuracy
#The pruned decision tree leads to 89% of correct predictions, which is a bit more accurate than the decision tree at 88%. 
#It's also the most accurate prediction and takes less computing time out of all the models (and also provides which features affect the subscription the most, which other models don't), so this is the one we choose.

#take it a step further: could do randomForest or logistic regression given the features selected from the decision tree to see if it gives a more accurate prediction/higher probability of correctly predicting subscriptions