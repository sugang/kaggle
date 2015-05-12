# predict result and analysis result



PredictWithModel <- function(model, test_data) {
  pre_result <- predict(model, test_data)
  
  write.csv(cbind(Id = test_data$Id, Prediction = pre_result), quote = F, row.names = F, file = "./predict_result.csv")
}
