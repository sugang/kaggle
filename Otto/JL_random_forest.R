#Splits the training data into 2 sets,
#retaining the class distribution. Builds a
#randomForest on the first set, and a gbm
#on the randomForest predictions for the
#second set. Play with below parameters to
#improve performace - my best LB score using
#this approach is 0.49635

#The evaluation metric function used by caret to score
#a fold. Not optimal - needs vectorization

##### for multiple process
library(doMC)
registerDoMC(cores = 4)


MCLogLoss <- function(data, lev = NULL, model = NULL)  {

  obs <- model.matrix(~data$obs - 1)
  preds <- data[, 3:(ncol(data) - 1)]

  err = 0
  for(ob in 1:nrow(obs))
  {
    for(c in 1:ncol(preds))
    {
      p <- preds[ob, c]
      p <- min(p, 1 - 10e-15)
      p <- max(p, 10e-15)
      err = err + obs[ob, c] * log(p)
    }
  }

  out <- err / nrow(obs) * -1
  names(out) <- c("MCLogLoss")
  out
}



library(caret)
#Prepare training\testing data, extract
#target and save test id's
train <- read.csv("./Data/train.csv")
train <- train[, -which(names(train)=="id")]
target <- train$target
train <- train[, -which(names(train)=="target")]
test <- read.csv("./Data/test.csv")
id <- test$id
test <- test[, -which(names(test)=="id")]


#TUNE HERE
#How much of the data to use to build the randomForest
TRAIN_SPLIT = 0.7
RF_MTRY = 20
RF_TREES = 225
GBM_IDEPTH = 2:8
GBM_SHRINKAGE  = seq(0.1, 0.5, by = 0.1)
GBM_TREES = 20:30
GBM_MINOBS = 10:30
#TUNE HERE

#Split training data into two sets(keep class distribution)
set.seed(20739)
trainIndex <- createDataPartition(target, p = TRAIN_SPLIT, list = TRUE, times = 1)
allTrain <- train
allTarget <- target
train <- allTrain[trainIndex$Resample1, ]
train2 <- allTrain[-trainIndex$Resample1, ]
target <- allTarget[trainIndex$Resample1]
target2 <- allTarget[-trainIndex$Resample1]

#Build a randomForest using first training set
fc <- trainControl(method = "repeatedCV",
                   number = 3,
                   repeats = 3,
                   verboseIter=FALSE,
                   returnResamp="all",
                   classProbs=TRUE,
                   summaryFunction=MCLogLoss)
tGrid <- expand.grid(mtry = 2:RF_MTRY)
model <- train(x = train, y = target, method = "rf", trControl = fc, tuneGrid = tGrid, metric = "MCLogLoss", ntree = RF_TREES)
#Predict second training set, and test set using the randomForest
train2Preds <- predict(model, train2, type="prob")
testPreds <- predict(model, test, type="prob")
model$finalModel


# for RRF
fc <- trainControl(method = "repeatedCV",
                   number = 3,
                   repeats = 3,
                   verboseIter=TRUE,
                   returnResamp="all",
                   classProbs=TRUE,
                   summaryFunction=MCLogLoss)
tGrid <- expand.grid(mtry = 2:7, coefReg = seq(0.01, 1, length = 5), coefImp = seq(0, 1, length = 5))
model <- train(x = train, y = target, method = "RRF", trControl = fc, tuneGrid = tGrid, metric = "MCLogLoss", ntree = RF_TREES)
model <- train(target ~ . , data = subset(train, select = -id), method = "RRF", trControl = fc, tuneGrid = tGrid, metric = "MCLogLoss", ntree = RF_TREES)
model <- train(target ~., data = train, method = "RRF", trControl = fc, tuneLength = 5, metric = "MCLogLoss", ntree = 600)

#Predict second training set, and test set using the randomForest
train2Preds <- predict(model, train2, type="prob")
testPreds <- predict(model, test, type="prob")
model$finalModel


# for extra tree
fc <- trainControl(method = "repeatedCV",
                   number = 3,
                   repeats = 3,
                   verboseIter=FALSE,
                   returnResamp="all",
                   classProbs=TRUE,
                   summaryFunction=MCLogLoss)
tGrid <- expand.grid(mtry = 2:7, coefReg = seq(0.01, 1, length = 5), coefImp = seq(0, 1, length = 5)))
test_eTree <- train(target~., data=train, method = "extraTrees", tuneLength = 4, trControl = fitControl, preProc = "YeoJohnson", ntree = 1000)



#Build a gbm using only the predictions of the
#randomForest on second training set
fc <- trainControl(method = "repeatedCV",
                   number = 10,
                   repeats = 1,
                   verboseIter=FALSE,
                   returnResamp="all",
                   classProbs=TRUE,
                   summaryFunction=MCLogLoss,)
tGrid <- expand.grid(interaction.depth = GBM_IDEPTH, shrinkage = GBM_SHRINKAGE, n.trees = GBM_TREES, n.minobsinnode = GBM_MINOBS)
model2 <- train(x = train2, y = target2, method = "gbm", trControl = fc, tuneGrid = tGrid, metric = "MCLogLoss", verbose = FALSE)
model2
hist(model2$resample$MCLogLoss)

####### extra trees
#Build a randomForest using first training set
fc <- trainControl(method = "repeatedCV",
                   number = 3,
                   repeats = 3,
                   verboseIter=FALSE,
                   returnResamp="all",
                   classProbs=TRUE,
                   summaryFunction=MCLogLoss)
tGrid <- expand.grid(mtry = 2:RF_MTRY)
test_eTree <- train(target~., data=train, method = "extraTrees", trControl = fc, tunningLength = 4, metric = "MCLogLoss", preProc = "YeoJohnson", ntree = 1000)


#Build submission
submit <- predict(model2, testPreds, type="prob")
# shrink the size of submission
submit <- format(submit, digits=2, scientific = FALSE)
submit <- cbind(id=1:nrow(testPreds), submit)
write.csv(submit, "submit.csv", row.names=FALSE)
