#NNET Artificial Neural Network Approach 
library(nnet)
wt <- c(3,1,2,3,3,1,3,2,1);
wt <- wt / sum(wt); #wt
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
classes <- colnames(submission)[c(-1)]
names(wt) <- classes
weight <- sapply(train$target, function(x){wt[x]})
# nnet.classifier.01 <- nnet(train.features, train$target, size=50, weights=weight)
train.data <- data.frame(train[,rf.vs.varSelRF], target=as.factor(train$target))

#nnet is not entirely terrible, but still not as good as svm.
nnet.classifier.01 <- nnet(target~., data=train.data, size=50, weights=weight, MaxWts=5000)
nnet.predict.class <- predict(nnet.classifier.01, test, type="class")
nnet.predict.class <- predict(train[,rf.vs.varSelRF], train, type="class")
table(nnet.predict.class, train$target)

nnet.predict <- predict(nnet.classifier.01, test, type="raw")
