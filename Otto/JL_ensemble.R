#
#
#

dl = read.csv('./Data/20150514_deep_learning_final.csv',header=TRUE,stringsAsFactors = F)
gb = read.csv('./Data/20150511_xgboost_slow_learning_final.csv',header=TRUE,stringsAsFactors = F)
ensemble_d_g = (dl + 3*gb) / 4

pred_result = data.frame(id = test_data$id,ensemble_d_g[,-1])


write.csv(pred_result,file='submission5.csv', quote=FALSE,row.names=FALSE)
