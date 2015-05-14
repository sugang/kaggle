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

pre_pro = preProcess(x[trind,], method = "YeoJohnson")
x = predict(pre_pro, x)


# Set necessary parameter

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


# Run Cross Valication
cv.nround = 1000
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
save(TrainRes, TestRes, param2, file = "./Data/xgboost_benchmark.Rdata")

load("./Data/xgboost_benchmark.Rdata")

# min in each column
library(ggplot2)
#data.frame(min_val = as.numeric(apply(TrainRes,2,min)), type = "Train")
AllRes <- rbind(data.frame(idx = 1:dim(TrainRes)[2], min_val = apply(TrainRes,2,min), type = "Train"), data.frame(idx = 1:dim(TrainRes)[2], min_val =apply(TestRes,2,min), type = "Test"))
ggplot(data = AllRes, aes(x = idx, y = min_val, color = type))+ geom_line() + annotate("text", x = subset(AllRes[AllRes$type == "Test",], min_val == min(min_val))$idx, y = subset(AllRes[AllRes$type == "Test",], min_val == min(min_val))$min_val, label = paste(subset(AllRes[AllRes$type == "Test",], min_val == min(min_val))$idx, subset(AllRes[AllRes$type == "Test",], min_val == min(min_val))$min_val, sep = ","))

library(reshape2)
TestRes_melt <- melt(TestRes)
ggplot(data = TestRes_melt, aes(x = Var1, y = value, color = as.factor(Var2))) + geom_line() + scale_x_log10() + scale_y_log10()

subset(AllRes[AllRes$type == "Test",], min_val == min(min_val))$min_val

plot(apply(TrainRes,2,min))
points(apply(TestRes,2,min))

#Converting to data.frame for plotting 
TestRes_df <- data.frame(TestRes)
#Plotting parameter curves
ggplot(data = TestRes_df, aes(x = 1:1000, y = TestRes_df$X9))+
  geom_line()


# for ensemble method
param_opt <- list()
param_idx <- 1
pred_result = data.frame(matrix(0,9,length(teind)/9))
for(i in 1:length(param2)){
  print(paste0("CV Round", i))
  bst.cv <- xgb.cv(param= param2[[i]], data = x[trind,], label = y, 
                   nfold = 3, nrounds=cv.nround)
  TrainRes[,i] <- as.numeric(bst.cv[,train.mlogloss.mean])
  TestRes[,i]  <- as.numeric(bst.cv[,test.mlogloss.mean])
  bst.nrounds <- which.min(bst.cv$test.mlogloss.mean)
  if(min(as.numeric(bst.cv[,test.mlogloss.mean])) < 0.47) {
    param_opt[param_idx] = param2[[i]]
    param_idx = param_idx + 1
    bst = xgboost(param=param2[[3]], data = x[trind,], label = y, nrounds=bst.nrounds)
    pred = predict(bst,x[teind,])
    pred = matrix(pred,ncol = 9,byrow = T)
    pred_result += pred
  }
}
pre_result = pre_result / length(param_opt)


bst.cv <- xgb.cv(param= param2[[75]], data = x[trind,], label = y, nfold = 3, nrounds=1000)

pred_result = data.frame(matrix(0,length(teind), 9))
loop_n = 100
for(i in 1:loop_n){
    bst = xgboost(param=param2[[75]], data = x[trind,], label = y, nrounds= 459)
    pred = predict(bst,x[teind,])
    pred = matrix(pred,ncol = 9,byrow = T)
    pred_result = pred + pred_result
}

# Output submission
pred_result = data.frame(1:nrow(pred_result),pred_result)
names(pred_result) = c('id', paste0('Class_',1:9))
write.csv(pred_result,file='submission5.csv', quote=FALSE,row.names=FALSE)

save(pred_result, param2, file = "./Data/xgboost_53.Rdata")
