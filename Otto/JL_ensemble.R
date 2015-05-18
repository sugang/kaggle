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

localH2O <- h2o.init(nthread = 8)

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

nloop = 1
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

#grid searching for parameters
grid_search <- function(n_set){
  
  #param is a list of parameters
  
  # Set necessary parameter
  param <- list("objective" = "multi:softprob",
                "max_depth"=6,
                "eta"=0.1,
                "subsample"=0.7,
                "colsample_bytree"= 1,
                #                "gamma"=2,
                #                "min_child_weight"=4,
                "eval_metric" = "mlogloss",
                "silent"=1,
                "num_class" = 9,
                "nthread" = 6)
  
  param_list <- list()
  para_idx = 1
  
  for (i in seq(n_set)){
    param$subsample <- seq(0.1,1, length=n_set)[i]
    for (j in seq(n_set)){
      param$colsample_bytree <- seq(0.1, 1, length=n_set)[j]
      
      param_list[[para_idx]] <- param
      para_idx = para_idx + 1
    }
  }
  return(param_list)
}
param2 <- grid_search(n_set=10)

# for test set

pre_test    = data.frame(matrix(0,nrow(test), 9))

bst.cv <- xgb.cv(param= param2[[75]], data = x[trind,], label = y, nfold = 3, nrounds=700)
pred_result = data.frame(matrix(0,length(teind), 9))
n_loop = 200
for(i in 1:n_loop){
  bst = xgboost(param=param2[[75]], data = x[trind,], label = y, nrounds= which.min(bst.cv[,test.mlogloss.mean]))
  pred = predict(bst,x[teind,])
  pred = matrix(pred,ncol = 9,byrow = T)
  pred_result = pred + pred_result
}

train2Pred_xg = pre_result / n_loop

####################################################
######### Ensemble by gbm                 ##########
####################################################

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
model2 <- train(x = cbind(train2Pred_xg, train2Pred_nn), y = target2, method = "gbm", trControl = fc, tuneGrid = tGrid, metric = "MCLogLoss", verbose = FALSE)
