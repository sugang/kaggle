#modified from Kaggle code.
#testing data range of 0.48 - 0.49 neighborhood. Will try feature selection + subsampling techniques.
#it seems that the class bias has not been addressed yet.
require(xgboost)
require(methods)


# setwd('/Users/sugang/Desktop/Kaggle/otto')
setwd('C:/Projects/kaggle/otto')
train = read.csv('train.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('test.csv',header=TRUE,stringsAsFactors = F)

for(i in 2:94){
	train[,i] <- as.numeric(train[,i])
	train[,i] <- sqrt(train[,i]+(3/8))
}

for(i in 2:94){
	test[,i] <- as.numeric(test[,i])
	test[,i] <- sqrt(test[,i]+(3/8))
}

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
              "subsample" = 0.5,
              "colsample_bytree" = 0.5,
              "eta" = 0.1
              )

x.train <-  x[trind,]
y.train <- y

cv.nround <- 500

#gives around 0.475 cross validation error
bst.cv = xgb.cv(param=param, data = x.train, label = y.train, nfold = 3, nrounds=cv.nround) #this program is kinda buggy sometimes. xgboost still need some work.

for(i in 1:10){
	print(i)
	cv.nround <- 550 #550 seem to be optimal
	bst = xgboost(param=param, data = x.train, label = y.train, nrounds=cv.nround)
	# bst = xgboost(param=param, data = x.train, label = y.train, nrounds=nround)
	pred = predict(bst,x[teind,])
	pred = matrix(pred,9,length(pred)/9)
	pred = t(pred)
	# pred = format(pred, digits=2,scientific=F) # shrink the size of submission
	pred = data.frame(1:nrow(pred),pred)
	names(pred) = c('id', paste0('Class_',1:9))
	write.csv(pred,file=paste('20150511_xgboost_slow_learning_', i, '.csv', sep=''), quote=FALSE,row.names=FALSE)
}
