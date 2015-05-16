setwd("~/Documents/Github/kaggle_OttoGroup/Data")

train <- read.csv("./Data/train.csv")
test <- read.csv("./Data/test.csv")

##### caret ensemble package ####
library("caretEnsemble")

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

# Pre-define traincontrol for repeated cross validation
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  classProbs = TRUE,
  summaryFunction=MCLogLoss, 
  savePredictions=TRUE,
  returnData = TRUE,
  verboseIter = TRUE)

specifiedTuneList =list(
  rf1=caretModelSpec(method='rf', tuneGrid=data.frame(.mtry=1:5)),
  rf2=caretModelSpec(method='rf', tuneGrid=data.frame(.mtry=5:10), preProcess='pca'),
  nn=caretModelSpec(method='nnet', tuneLength=10, trace=FALSE)
)

model_list <- caretList(
  target~., data=train,
  trControl=fitControl,
  tuneList=specifiedTuneList,
  methodList=c('rpart', 'nnet')
)

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