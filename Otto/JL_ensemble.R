#
#
#
##################### first attempt for ensemble
dl = read.csv('./Data/20150514_deep_learning_final.csv',header=TRUE,stringsAsFactors = F)
gb = read.csv('./Data/20150511_xgboost_slow_learning_final.csv',header=TRUE,stringsAsFactors = F)
ensemble_d_g = (dl + 3*gb) / 4

pred_result = data.frame(id = test_data$id,ensemble_d_g[,-1])


write.csv(pred_result,file='submission5.csv', quote=FALSE,row.names=FALSE)


##################### second attempt for ensemble
####### parameters
TRAIN_SPLIT = 0.7

# so begin with segaration of data
library(caret)
train <- read.csv("./Data/train.csv")
test <- read.csv("./Data/test.csv")
alltrain <- subset(train, select = -id)
alltarget <- train$target
trainIndex <- createDataPartition(train$target, p = TRAIN_SPLIT, list = TRUE, times = 1) 
train <- alltrain[trainIndex$Resample1, ] 
train2 <- alltrain[-trainIndex$Resample1, ] 
target <- alltarget[trainIndex$Resample1] 
target2 <- alltarget[-trainIndex$Resample1] 

train_xg <- subset(train, select = -target)
train_nn <- train
train2_bk <- train2
###########################################################
##########   The part for neural network        ###########
###########################################################
library("h2o")

localH2O <- h2o.init(nthread = 6)

for(i in 1:93){
  train[,i] <- as.numeric(train[,i])
  train[,i] <- sqrt(train[,i]+(3/8))
}

for(i in 1:93){
  train2[,i] <- as.numeric(train2[,i])
  train2[,i] <- sqrt(train2[,i]+(3/8))
}

test <- read.csv("./Data/test.csv")
test <- test[,-1]

for(i in 1:93){
  test[,i] <- as.numeric(test[,i])
  test[,i] <- sqrt(test[,i]+(3/8))
}

train.hex <- as.h2o(localH2O,train)
train2.hex <- as.h2o(localH2O,train2)
test.hex <- as.h2o(localH2O,test)

predictors <- 2:(ncol(train.hex)-1)
response <- ncol(train.hex)

train2_pred <- matrix(0, nrow(train2), 9)
test_pred <- matrix(0, nrow(test), 9)

nloop = 5
for(i in 1:nloop){
  print(i)
  model <- h2o.deeplearning(x=predictors,
                            y=response,
                            data=train.hex,
                            classification=T,
                            activation="TanhWithDropout",
                            hidden=c(1024,512,256),
                            hidden_dropout_ratio=c(0.5,0.5,0.5),
                            input_dropout_ratio=0.05,
                            epochs=50,
                            l1=1e-5,
                            l2=1e-5,
                            rho=0.99,
                            epsilon=1e-8,
                            train_samples_per_iteration=2000,
                            max_w2=10,
                            seed=1)
  train2_pred <- train2_pred + as.data.frame(h2o.predict(model,train2.hex))[,2:10]
  test_pred <- test_pred + as.data.frame(h2o.predict(model,test.hex))[,2:10]
  print(i)
}      

train2Pred_nn <- train2_pred / nloop
test2Pred_nn <- test_pred / nloop

###########################################################
##########   The part for xgboost               ###########
###########################################################
library(xgboost)
train = train_xg
train2 = train2_bk
y = train[,ncol(train)]
y = gsub('Class_','',target)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

test <- read.csv("./Data/test.csv")
test <- test[,-1]

x = rbind(train,subset(train2, select = -target), test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):(nrow(train)+nrow(train2))
testind = (nrow(train)+nrow(train2)+1) : nrow(x)

# parameter from forum
param_f <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "gamma" = 0,
              "nthread" = 4,
              "eta" = 0.05,
              "max_depth" = 12,
              "min_child_weight" = 4,
              "subsample" = .9,
              "colsample_bytree" = .8)

# for test set
pred_test   = data.frame(matrix(0,length(testind), 9))
# for train2
pred_result = data.frame(matrix(0,length(teind), 9))

bst.cv <- xgb.cv(param= param_f, data = x[trind,], label = y, nfold = 3, nrounds=800)
#bst.cv <- xgb.cv(param= param_f, data = x[trind,], label = y, nfold = 3, nrounds=200)
n_loop = 30
for(i in 1:n_loop){
  bst = xgboost(param=param_f, data = x[trind,], label = y, nrounds= which.min(bst.cv[,test.mlogloss.mean]))
  pred = predict(bst,x[teind,])
  pred = matrix(pred,ncol = 9,byrow = T)
  pred_result = pred + pred_result
  pred = predict(bst,x[testind,])
  pred = matrix(pred,ncol = 9,byrow = T)
  pred_test = pred + pred_test
}

train2Pred_xg = pred_result / n_loop
testPred_xg = pred_test / n_loop

####################################################
######### Ensemble by gbm                 ##########
####################################################
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

test <- read.csv("./Data/test.csv")
test <- test[,-1]

#Build a gbm using only the predictions of the
#randomForest on second training set
fc <- trainControl(method = "repeatedCV", 
                   number = 10, 
                   repeats = 1, 
                   verboseIter=FALSE, 
                   returnResamp="all", 
                   classProbs=TRUE, 
                   summaryFunction=MCLogLoss,) 
model2 <- train(x = data.frame(cbind(train2_bk, train2Pred_xg, train2Pred_nn)), y = target2, method = "gbm", trControl = fc, tuneLength = 5, metric = "MCLogLoss", verbose = FALSE)



model2 <- train(x = data.frame(cbind(train2_bk, train2Pred_xg, train2Pred_nn)), y = target2, method = modelinfo, trControl = fc, tuneLength = 10, metric = "MCLogLoss", verbose = FALSE)



submit <- predict(model2, cbind(test, testPred_xg, testPred_nn), type="prob")
submit <- cbind(id=1:nrow(testPred_xg), submit)
write.csv(submit, "submit.csv", row.names=FALSE)