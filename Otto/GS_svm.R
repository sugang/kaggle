#svm code
setwd('C:/Projects/kaggle/otto')
library(e1071)
library(readr)
train <- read_csv("train.csv")
test <- read_csv("test.csv")
train.features <- train[,c(-1,-95)]
train.class <- as.factor(train$target)
test.features <- test[,c(-1,-95)]
load('varSelRF.RData')

#artificial weight, avoid some classes getting overly under sampled
wt <- c(4,1,2,4,4,1,4,2,1);
wt <- wt / sum(wt); #wt
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
classes <- colnames(submission)[c(-1)]
names(wt) <- classes


train.features.hasValue <- apply(train.features, 1, function(x){sum(x > 0)})

#train with selected samples, selected feature, and increased nu
svm.classifier.04 <- svm(
	train.features[ train.features.hasValue >= 10, rf.vs.varSelRF ], 
	train.class[ train.features.hasValue >= 10 ], 
	probability=TRUE, 
	class.weights=wt, 
	type="nu-classification", 
	nu=0.10
)

predict.svm.classifier.04.train <- predict(svm.classifier.04, train[,rf.vs.varSelRF], probability=FALSE) #try on the train to see what happens.
predict.svm.classifier.04 <- predict(svm.classifier.04, test[,rf.vs.varSelRF], probability=TRUE)
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
submission[,2:10] <- attr(predict.svm.classifier.04, 'probabilities')
