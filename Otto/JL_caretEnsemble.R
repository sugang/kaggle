setwd("~/Documents/Github/kaggle_OttoGroup/Data")

train <- read.csv("./Data/train.csv")
test <- read.csv("./Data/test.csv")

##### for multiple process
# not working for windows
library(doMC)
registerDoMC(cores = 4)
# for windows
library(doSNOW)
registerDoSNOW(cl = 3)

##### caret ensemble package ####
library("caretEnsemble")

MCLogLoss <- function(dat, lev = NULL, model = NULL)  {

  obs <- model.matrix(~dat$obs - 1)
  preds <- dat[, 3:(ncol(dat) - 1)]

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
  print(out)
  out
}

# Pre-define traincontrol for repeated cross validation
fitControl <- trainControl(
  method = "repeatedcv",
  number = 3,
  repeats = 1,
  classProbs = TRUE,
  summaryFunction=MCLogLoss,
  savePredictions=TRUE,
  returnData = TRUE,
  verboseIter = TRUE)

specifiedTuneList =list(
  rf1=caretModelSpec(method='rf', tuneGrid=data.frame(.mtry=2)),
  rf2=caretModelSpec(method='rf', tuneGrid=data.frame(.mtry=10), preProcess='pca'),
  nn=caretModelSpec(method='nnet', tuneLength=10, trace=FALSE)
)

model_list <- caretList(
  target~., data=train,
  trControl=fitControl,
  tuneList=specifiedTuneList,
  methodList=c('rpart', 'nnet')
)
caretStack(models, method='glm')
greedy_ensemble <- caretEnsemble(model_list)
summary(greedy_ensemble)


model_list$glm
model_list$rpart

####### be careful about this part!!!
models <- caretList(
  x=iris[1:50,1:2],
  y=iris[1:50,3],
  trControl=trainControl(method='cv'),
  methodList=c('rpart', 'glm')
)
caretStack(models, method='glm')


xyplot(resamples(model_list))

modelCor(resamples(model_list))

greedy_ensemble <- caretEnsemble(model_list)
summary(greedy_ensemble)



## for only caret
library("caret")
fitControl <- trainControl(
  method = "repeatedcv",
  number = 3,
  repeats = 3,
  summaryFunction=MCLogLoss,
  savePredictions=TRUE,
  returnData = TRUE,
  verboseIter = TRUE)
test_eTree <- train(target~., data=train, method = "extraTrees", tuneLength = 3, trControl = fitControl, preProc = "YeoJohnson", ntree = 600)

test_eTree <- train(target~., data=subset(train, select = -id), method = "rf", tuneLength = 3, trControl = fitControl, preProc = "YeoJohnson", ntree = 600)

model <- train(x = train, y = target, method = "rf", trControl = fc, tuneGrid = tGrid, metric = "MCLogLoss", ntree = RF_TREES)


fitControl <- trainControl(
  method = "repeatedcv",
  number = 3,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction=MCLogLoss,
  savePredictions=TRUE,
  returnData = TRUE,
  verboseIter = TRUE)
rforest <- train(target~., data=train, method = "rf", trControl = fitControl, preProc = "YeoJohnson", ntree = 200)

pred = predict(rforest$finalModel,test, type="prob")
pred_result = data.frame(1:nrow(pred),pred)
names(pred_result) = c('id', paste0('Class_',1:9))
write.csv(pred_result,file='submission5.csv', quote=FALSE,row.names=FALSE)


###### Over sampling of minority class
require(DMwR)

ncols<-ncol(train)
dm<-cbind(train[,2:ncols],train[,1])
dmSmote<-SMOTE(target ~ . , dm,k=5,perc.over = 1400,perc.under=140)
dm<-cbind(dmSmote[ncols],dmSmote[1:ncols-1])
