require(xgboost)
require(methods)
require(caret)
library(ggplot2)

train = read.csv('./Data/train.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('./Data/test.csv',header=TRUE,stringsAsFactors = F)
train = train[,-1]
test = test[,-1]


y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)


pre_pro = preProcess(x, method = "BoxCox")
x = pre_pro(x)


# Set necessary parameter

#Randomly searching for parameters
random_search <- function(n_set){
  
  #param is a list of parameters
  
  # Set necessary parameter
  param <- list("objective" = "multi:softprob",
                "max_depth"=6,
#                "eta"=0.1,
                "subsample"=0.7,
                "colsample_bytree"= 1,
#                "gamma"=2,
#                "min_child_weight"=4,
                "eval_metric" = "mlogloss",
                "silent"=1,
                "num_class" = 9,
                "nthread" = 3)
  
  param_list <- list()
  
  for (i in seq(n_set)){
    
    ## n_par <- length(param)
    
    
#    param$max_depth <- sample(3:10,1, replace=T)
    
#    param$eta <- runif(1,0.01,0.6)
    param$subsample <- seq(0.1,1, length=n_set)[i]
    param$colsample_bytree <- seq(0.1,1,)
    param$subsample <- runif(1,0.1,1)
    param$colsample_bytree <- runif(1,0.1,1)
#    param$min_child_weight <- sample(1:17,1, replace=T)
    
#    param$gamma <- runif(1,0.1,10)
#    param$min_child_weight <- sample(1:15,1, replace=T)
    
    param_list[[i]] <- param
    
  }
  
  return(param_list)
  
}
param2 <- random_search(n_set=100)


# Run Cross Valication
cv.nround = 400
TrainRes <- matrix(, nrow=cv.nround, ncol=length(param2))
TestRes <- matrix(, nrow= cv.nround, ncol=length(param2))

for(i in 1:length(param2)){
  print(paste0("CV Round", i))
  bst.cv <- xgb.cv(param= param2[[i]], data = x[trind,], label = y, 
                   nfold = 3, nrounds=cv.nround)
  TrainRes[,i] <- as.numeric(bst.cv[,train.mlogloss.mean])
  TestRes[,i]  <- as.numeric(bst.cv[,test.mlogloss.mean])
  
}

#Saving CV results
save(TrainRes, TestRes, param2, file = "xgboost_benchmark.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark2.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark3.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark4.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark5.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark6.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark7.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark8.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark9.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark10.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark11.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark12.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark13.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark14.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark15.Rdata")
save(TrainRes, TestRes, param2, file = "xgboost_benchmark16.Rdata")

load("xgboost_benchmark10.Rdata")

#Converting to data.frame for plotting 
TestRes_df <- data.frame(TestRes)
#Plotting parameter curves
ggplot(data = TestRes_df, aes(x = 1:1000, y = TestRes_df$X9))+
  geom_line()

# Train the model
nround = 532
bst = xgboost(param=param2[[3]], data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submission5.csv', quote=FALSE,row.names=FALSE)