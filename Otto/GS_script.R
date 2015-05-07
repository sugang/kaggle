#use deeplearning.

library(randomForest)
library(readr)
library(plyr)
setwd('C:/Projects/kaggle/otto')
train <- read_csv("train.csv")
test <- read_csv("test.csv")
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
classes <- colnames(submission)[c(-1)]

train.features <- train[,c(-1,-95)]
train.class <- as.factor(train$target)
test.features <- test[,c(-1,-95)]

#quick run, error ~ 22%
rf.classifier.full <- randomForest(train.features, train.class, ntree=50, importance=TRUE)
rf.classifier.importance <-  importance(rf.classifier.full, type=1)

#choose importance >= 10
rf.vs.impSel <- rf.classifier.importance[,1] >= 10
rf.classifier.vs.impSel <- randomForest(train.features[,rf.vs.impSel], train.class, ntree=50) #retrain using selected features

rf.classifier.vs.impSel.predict <-predict(rf.classifier.vs.impSel, test[,c(-1)][,rf.vs.impSel], type="prob")
submission[,2:10] <- rf.classifier.vs.impSel.predict #varsel w/ rF 20% error




#variable selection -simple
library(varSelRF)
rf.vs <- varSelRF(train.features, train.class) #this takes more than several hours. Tree number maybe too large.

###############################################################################################################
#svm
library(e1071)
#svm with default parameters and all features.
svm.classifier.full <- svm(train.features, train.class)
svm.predict.full <- predict(svm.classifier.full, train.features)
table(svm.predict.full, train.class) #will test on feature selection.

svm.predict.full.test <- predict(svm.classifier.full, test.features)

svm.predict.output <- adply(as.vector(svm.predict.full.test), 1, function(x){(classes == x)*1}) #plyr function to 
colnames(svm.predict.output) <- c('id',classes)

#svm using nu-classification
svm.classifier.01 <- svm(train.features, train.class, type="nu-classification", kernal='linear', nu=0.05, cross=5)

#consider weight balancing issue:
wt <- 10000/table(train$target) # run with weight




###############################################################################################################
# library(penalizedSVM) #this only works on single class classification methods.
# svm.classifier.p <- lpsvm(train.features, train.class, k=5, nu=0.05, output=1)




#deep-learning, need to set up using h2o.
#########################################
library(h2o)
localH2O <- h2o.init(max_mem_size = "16g")

#load the training data
h2o.train <- h2o.importFile(
	localH2O,
	path = "C:/Projects/kaggle/otto/train.csv",
	key = "h2o.train"
)

h2o.train.features <- h2o.train[,c(-1,-95)]
h2o.train.class <- as.factor(h2o.train$target)

dl.classifier.full <- h2o.deeplearning(
	x = colnames(h2o.train.features),
	y = colnames(h2o.train.class),
	data = h2o.train,
	classification = T,
	activation=c("Tanh"),
	hidden=c(100,200,200,200,100),
	epochs=20,
	l1 = 1e-7,
	variable_importances=T,
	nfolds=5
)
#very large error rate.


dl.vs <- dl.classifier.full@model$varimp

#h2o random forest
#dl classification doesn't quite work. for this dataset.


rf.classifier.full <- h2o.randomForest(
	x = colnames(h2o.train.features),
	y = colnames(h2o.train.class),
	data = h2o.train,
	classification = TRUE,
	ntree = 500,
	depth = 10,
	balance.classes = TRUE,
	importance = TRUE,
	oobee = TRUE
	# nfolds = 5
)





