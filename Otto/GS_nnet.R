setwd('C:/Projects/kaggle/otto')
library(nnet)
library(readr)
train <- read_csv("train.csv")
test <- read_csv("test.csv")
train.features <- train[,c(-1,-95)]
train.class <- as.factor(train$target)
test.features <- test[,c(-1,-95)]
load('varSelRF.RData')


wt <- c(3,1,2,3,3,1,3,2,1);
wt <- wt / sum(wt); #wt
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
classes <- colnames(submission)[c(-1)]
names(wt) <- classes
weight <- sapply(train$target, function(x){wt[x]})
# nnet.classifier.01 <- nnet(train.features, train$target, size=50, weights=weight)
# train.data <- data.frame(train[,rf.vs.varSelRF], target=as.factor(train$target))

#nnet is not entirely terrible, but still not as good as svm.
# nnet.classifier.01 <- nnet(target~., data=train.data, size=50, weights=weight, MaxWts=5000)
# nnet.predict.class <- predict(nnet.classifier.01, test, type="class")
# nnet.predict.class <- predict(train[,rf.vs.varSelRF], train, type="class")
# table(nnet.predict.class, train$target)
# nnet.predict <- predict(nnet.classifier.01, test, type="raw")

#try to use contrasts
train.data <- data.frame(train.features[,rf.vs.varSelRF], target=as.factor(train$target))
# mask <- colnames(train.features) %in% rf.vs.varSelRF #not as well described as in the documentation!!
nnet.classifier.02 <- nnet(target~., data=train.data, size=100, weights=weight, MaxNWts=50000, entropy=TRUE, maxit = 500)
# nnet.predict.class.02 <- predict(nnet.classifier.02, train, type="class")
nnet.predict.02 <- predict(nnet.classifier.02, train, type="class")
nnet.predict.02.test <- predict(nnet.classifier.02, test, type="raw")
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
submission[,2:10] <- nnet.predict.02.test
#single layer nnet. May not be as useful.
#not overfitting as bad as the last one.
#This is a less overfitted model.





train.data <- data.frame(train.features[,rf.vs.varSelRF], target=as.factor(train$target))
nnet.classifier.03 <- nnet(target~., data=train.data, size=500, weights=weight, MaxNWts=50000, entropy=TRUE, maxit = 500)


################################################################################################################################################
#this model is extremly overfitted. As a matter of fact, does not work on testing data at all.
#the best way is to do a grid search overnight...
train.data <- data.frame(train.features[,rf.vs.varSelRF], target=as.factor(train$target))
nnet.classifier.04 <- nnet(target~., data=train.data, size=250, weights=weight, MaxNWts=50000, entropy=TRUE, maxit = 500)
# the error was actually reduced to 93! after 500 iterations.
nnet.predict.04 <- predict(nnet.classifier.04, train, type="class")
# this model overfits tremendously ... got to see how it works with overfitting like this.
# extreme overfitting to the training data. - not a very robust model.
nnet.predict.04.test <- predict(nnet.classifier.04, test, type="raw")
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
submission[,2:10] <- nnet.predict.04.test
