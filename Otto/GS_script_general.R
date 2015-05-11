#use deeplearning.

library(randomForest)
library(readr)
library(plyr)
setwd('C:/Projects/kaggle/otto')
train <- read_csv("train.csv")
test <- read_csv("test.csv")
#load('varSelRF.RData')
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
classes <- colnames(submission)[c(-1)]

train.features <- train[,c(-1,-95)]
train.class <- as.factor(train$target)
test.features <- test[,c(-1,-95)]

#quick run, error ~ 22%
rf.classifier.full <- randomForest(train.features, train.class, ntree=50, importance=TRUE)
rf.classifier.importance <-  importance(rf.classifier.full, type=1)

#choose importance >= 10, ~ 20% error
rf.vs.impSel <- rf.classifier.importance[,1] >= 15 # first try >= 10

rf.classifier.vs.impSel <- randomForest(train.features[,rf.vs.impSel], train.class, ntree=750) #retrain using selected features with 50 trees, about 20% oob error
rf.classifier.vs.impSel.predict <-predict(rf.classifier.vs.impSel, test[,c(-1)][,rf.vs.impSel], type="prob")
submission[,2:10] <- rf.classifier.vs.impSel.predict #varsel w/ rF 20% error




#variable selection -simple
library(varSelRF)
rf.vs.varSelRF <- varSelRF(train.features, train.class) #this takes more than several hours. Tree number maybe too large.

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

#svm using nu-classification, however, should outcome
#need to enable probability from the model
svm.classifier.01 <- svm(train.features, train.class, type="nu-classification", kernal='linear', nu=0.05, cross=5)
svm.predict.01 <- predict(svm.classifier.01, train.features) #accurary is about 77.7%
table(svm.predict.01, train.class)

#consider weight balancing issue:
wt <- 10000/table(train$target) # run with weight

#make sure get probability
#this is no good enough!
svm.classifier.02 <- svm(train.features, train.class, probability=TRUE, class.weights=wt)
# predict(svm.classifier.02, train.features)
table(fitted(svm.classifier.02), train.class)

#do a feature selection and see what happens.
svm.classifier.03 <- svm(train.features, train.class, probability=TRUE, class.weights=wt, type="nu-classification", nu=0.05)
table(fitted(svm.classifier.02), train.class)

#use this as the svm baseline.
svm.classifier.03.predict <- predict(svm.classifier.03, test.features, probability=TRUE)
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
submission[,2:10] <- attr(svm.classifier.03.predict, 'probabilities') #create the SVM baseline.




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


###########################################################################################################
###########################################################################################################
# use subsampling to balance data
# use feature selection
# use more trees.

# need a sample function to sample 1500 rows from class_1, class_4, class_5, class_7, 4500 rows from class_2, class_3, class_6, class_8

# sTrain <- function(train){

# }

#use subsample. Another possiblility is to do multiple subsamples and add up the predicted results. Rerun a number of times, add them up, then scale.
#results are auto rescaled. Let's say run this at 500 trees 20 times.
#svm is so damn slow though...
getBalancedTrain <- function(train, sSize, vs){

	train.t <- data.frame(id=train$id, target=train$target)

	sampledIDs <- c(
		sample(train.t[train.t$target == 'Class_1',1],sSize[1]),
		sample(train.t[train.t$target == 'Class_2',1],sSize[2]),
		sample(train.t[train.t$target == 'Class_3',1],sSize[3]),
		sample(train.t[train.t$target == 'Class_4',1],sSize[4]),
		sample(train.t[train.t$target == 'Class_5',1],sSize[5]),
		sample(train.t[train.t$target == 'Class_6',1],sSize[6]),
		sample(train.t[train.t$target == 'Class_7',1],sSize[7]),
		sample(train.t[train.t$target == 'Class_8',1],sSize[8]),
		sample(train.t[train.t$target == 'Class_9',1],sSize[9])
	)

	train[train$id %in% sampledIDs, c(vs, 'target')]
}

#this is then with reduced sample size / balanced
train.balanced <- getBalancedTrain(train, c(1500,4500,4500,1500,1500,4500,1500,4500,4500), rf.vs.varSelRF)

#also remove samples that have many 0s. Won't be very helpful.
#check how many missing values per observation
#still a lot of rows with very sparse data. Bad for training. - contain little information.
#possible diagostics is to remove the way.
hist(apply(train.balanced[,rf.vs.varSelRF], 1, function(x){sum(x!=0)}), breaks=100)

#also try on full data with the weight operator
#at 500 tree, gets 19.88% accuracy
#at 1500 tree, gets
#1500 trees to 500 trees, increase 2% of accuracy
rf.classifier.01 <- randomForest(train.balanced[,rf.vs.varSelRF], as.factor(train.balanced$target), ntree=1500) #crash? - if class is null, the program simply crashes.
rf.classifier.01.predict <-predict(rf.classifier.01, test[,c(-1)][,rf.vs.varSelRF], type="prob")
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
submission[,2:10] <- rf.classifier.01.predict
write.csv(submission, file="rf.01.csv", row.names=F)

###############################################################################################################
#incorporate weight on the original dataset
wt <- 10000/table(train$target) 
wt <- wt / sum(wt)

rf.classifier.02 <- randomForest(train.features, train.class, ntree=1500, classwt=wt) #19.05% error

#with variable selection, the result didn't get better.
rf.classifier.03 <- randomForest(train.features[,rf.vs.varSelRF], train.class, ntree=1500, classwt=wt)

#the wt need to be adjusted. Right now the weight is a bit skewed for class 2,3,6,8
#create an artificial weight
wt <- c(3,1,2,3,3,1,3,2,1);
wt <- wt / sum(wt); #wt

train.features.hasValue <- apply(train.features, 1, function(x){sum(x > 0)})

#remove samples with little value and see what happens, new weight
#remove those sparse samples
#this will definitely improve classification performance.
rf.classifier.04 <- randomForest(train.features[train.features.hasValue >= 10,], train.class[train.features.hasValue >= 10], ntree=1500, classwt=wt)
rf.classifier.04.predict <-predict(rf.classifier.04, test[,c(-1)], type="prob")
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
submission[,2:10] <- rf.classifier.04.predict

### try regularized RF
install.packages('RRF')
wt <- c(3,1,2,3,3,1,3,2,1);
wt <- wt / sum(wt); #wt
rf.classifier.05 <- RRF(train.features, train.class, ntree=1500, classwt=wt)
##################################################################################
#nnet
#try nnet
library(nnet)


wt <- c(3,1,2,3,3,1,3,2,1);
wt <- wt / sum(wt); #wt
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
classes <- colnames(submission)[c(-1)]
names(wt) <- classes
weight <- sapply(train$target, function(x){wt[x]})
# nnet.classifier.01 <- nnet(train.features, train$target, size=50, weights=weight)
train.data <- data.frame(train[,rf.vs.varSelRF], target=as.factor(train$target))
nnet.classifier.01 <- nnet(target~., data=train.data, size=50, weights=weight, MaxWts=5000)
nnet.predict <- predict(nnet.classifier.01, test, type="raw")



##################################################################################
#rpart?
library(readr)
library(plyr)
setwd('C:/Projects/kaggle/otto')
train <- read_csv("train.csv")
test <- read_csv("test.csv")
train.features <- train[,c(-1,-95)]
train.class <- as.factor(train$target)
test.features <- test[,c(-1,-95)]
wt <- c(3,1,2,3,3,1,3,2,1);
wt <- wt / sum(wt); #wt
submission <- data.frame(id=test$id, Class_1=NA, Class_2=NA, Class_3=NA, Class_4=NA, Class_5=NA, Class_6=NA, Class_7=NA, Class_8=NA, Class_9=NA)
classes <- colnames(submission)[c(-1)]
names(wt) <- classes
weights <- sapply(train$target, function(x){wt[x]})



load('varSelRF.RData')

train.data <- data.frame(train[,rf.vs.varSelRF], target=train$target)
rpart.classifier.01 <- rpart(target~., data=train.data, weights=weights)
rpart.classifier.01.cv <- xpred.rpart(rpart.classifier.01, xval = 10)
predict.rpart(rpart.classifier.01, test, type='class')
#a fast method, but it doesn't quite work.
#terrible performance





#################################################################################
#knn3?






#################################################################################
#extreme learning machine?
# does not work, only produces class labels.





#################################################################################
#try to maybe remove data with little information - but this could introduce bias?

#Use Caret to build ensemble methods.
#try caret, caretEnsemble
