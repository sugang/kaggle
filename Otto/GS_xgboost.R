#modified from Kaggle code.
#testing data range of 0.48 - 0.49 neighborhood. Will try feature selection + subsampling techniques.
#it seems that the class bias has not been addressed yet.
require(xgboost)
require(methods)


setwd('/Users/sugang/Desktop/Kaggle/otto')
train = read.csv('train.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('test.csv',header=TRUE,stringsAsFactors = F)
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y) #indices for training data
teind = (nrow(train)+1):nrow(x) #indicies for testing data

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8,
              "subsample" = 0.4,
              "colsample_bytree" = 0.6
              )

# Run Cross Valication
# the test logloss end up to be around 0.52
cv.nround = 300
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 3, nrounds=cv.nround)

# Train the model
nround = 133
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submission2.csv', quote=FALSE,row.names=FALSE)
