# \section{Problem 4: Children and hotel reservations}
library(tidyverse)
library(ggplot2)
library(ModelMetrics)
library(caret)
library(lubridate)
library(plotROC)


# \subsection{Model building}
# We first load the data, and the head of data is as follows.
hotelsdev = read.csv("data/hotels_dev.csv",stringsAsFactors=TRUE)
summary(hotelsdev)
head(hotelsdev, 5)

# The mean f1-score from 5-fold cross-validation is applied to evaluate the performance of certain model.
evaluate_func = function(my_str,data){
  set.seed(100)
  indexs = createFolds(1:dim(data)[1], k = 5, list = TRUE, returnTrain = FALSE)
  err_result = c()
  for (i in 1:5){
    test_data = data[indexs[[i]],]
    train_data = data[-indexs[[i]],]
    model = glm(as.formula(my_str), data=train_data,family = "binomial")
    truth = test_data$children
    prediction = predict(model,newdata = test_data,type = "response")
    err_result = c(err_result,f1Score(truth,prediction))
  }
  return(mean(err_result))
}


# 1 Baseline 1
# The summary of Baseline 1(fitting on the whole data set):
baseline_1_str = "children ~ market_segment + adults + customer_type + is_repeated_guest"
model = glm(as.formula(baseline_1_str),data=hotelsdev,family = "binomial")
prediction = predict(model,type = "response")
truth = hotelsdev$children
# The f1-score of fitting
f1Score(truth,prediction)
# The 5-fold cross-validation f1-score of Baseline 1:
print(evaluate_func(baseline_1_str,hotelsdev))


# 2 Baseline 2
# The summary of Baseline 2(fitting on the whole data set):
baseline_2_str = "children ~ (. - arrival_date)"
model = glm(as.formula(baseline_2_str),data=hotelsdev,family = "binomial")
prediction = predict(model,type = "response")
truth = hotelsdev$children
# The f1-score of fitting
f1Score(truth,prediction)
# The 5-fold cross-validation f1-score of Baseline 2:
print(evaluate_func(baseline_2_str,hotelsdev))


# 3 Best linear model 3
# We generate the time-stamp of year, month, day, and day of week from arrival_date.
hotelsdev = mutate(hotelsdev,
                   arrival_date = ymd(arrival_date))
hotelsdev = mutate(hotelsdev, 
                   wday = wday(arrival_date) %>% factor(), 
                   day = day(arrival_date) %>% factor(),
                   month = month(arrival_date) %>% factor(),
                   year = year(arrival_date))
hotelsdev$arrival_date = as.character(hotelsdev$arrival_date)
hotelsdev <- subset(hotelsdev, select = -c(arrival_date)) 

# Then, we use greedy algorithm to find the best feature combination. 0.4685354
x_names = colnames(hotelsdev)[-6]
y_name = "children"
count = 0
rmse_record = 0
best_name = ""
best_test_str = ""
best_record_previous = 0
result_collect = c()

while(TRUE){
  for (name in x_names){
    if (count == 0){
      test_str = paste(y_name,"~",name,sep="")
    }
    else{
      test_str = paste(y_name,"+",name,sep="")
    }
    err = evaluate_func(test_str,hotelsdev)
    if (err>rmse_record){
      rmse_record = err
      best_name = name
      best_test_str = test_str
    }
  }
  if (rmse_record>best_record_previous){
    best_record_previous = rmse_record
  }
  else{
    break
  }
  y_name = best_test_str
  result_collect = c(result_collect,c(y_name,rmse_record))
  if (rmse_record>0.505){
    break
  }
  print(c(y_name,rmse_record))
  count = count + 1
  x_names = setdiff(x_names,best_name)
  if (length(x_names)==0){
    break
  }
}
print("children~reserved_room_type")                                             
print("0.364107552305935")                                                       
print("children~reserved_room_type+hotel")                                     
print("0.506343075649843")                                                         
print("children~reserved_room_type+hotel+previous_cancellations")                  
print("0.506437707712283")                                                         
print("children~reserved_room_type+hotel+previous_cancellations+booking_changes")  
print("0.506463838208211")  

# The summary of  best model(fitting on the whole data set):
best_str = "children~reserved_room_type+hotel+previous_cancellations+booking_changes"
model = glm(as.formula(best_str),data=hotelsdev,family = "binomial")
summary(model)
prediction = predict(model,type = "response")
truth = hotelsdev$children
# The f1-score of fitting
f1Score(truth,prediction)
# The 5-fold cross-validation f1-score of Baseline 1:
print(evaluate_func(best_str,hotelsdev))

# 4 Analysis
# The cross-validation f1-scores of 3 models are as follows:
# Baseline1 model 1:
#  0
# Baseline1 model 2:
#  0.4642258
# Best model:
#  0.5064638
# The best model is the best model with the highest f1-score.

# \subsection{Model validation: step 1}
# We first fit on the hotelsdev data 
best_str = "children~reserved_room_type+hotel+previous_cancellations+booking_changes"
model = glm(as.formula(best_str),data=hotelsdev,family = "binomial")
# Then we load hotels_val to conduct validation and draw ROC graph.
hotelsval = read.csv("data/hotels_val.csv")
prediction = predict(model,newdata = hotelsval,type = "response")
truth = hotelsval$children
roc_data = cbind(truth,prediction)
roc_data = data.frame(roc_data)
basicplot <- ggplot(roc_data, aes(d = truth,m = prediction))+ geom_roc() 

advanced_plot = basicplot + 
  style_roc(theme = theme_grey) +
  theme(axis.text = element_text(colour = "blue"),plot.title = element_text(hjust = 0.5))+ 
  annotate("text", x = .75, y = .25,label = paste("AUC =", round(calc_auc(basicplot)$AUC, 2))) +
  scale_x_continuous("1 - Specificity", breaks = seq(0, 1, by = .1)) ## x刻度
advanced_plot

# \subsection{Model validation: step 2}
# We first fit on the hotelsdev data 
best_str = "children~reserved_room_type+hotel+previous_cancellations+booking_changes"
model = glm(as.formula(best_str),data=hotelsdev,family = "binomial")
# Then we load hotels_val to conduct validation and draw ROC graph.
hotelsval = read.csv("data/hotels_val.csv")

# The hotels_val is divided into 20 folds.
set.seed(100)
indexs = createFolds(1:dim(hotelsval)[1], k = 20, list = TRUE, returnTrain = FALSE)
err_result = c()
for (i in 1:20){
  slice_data = hotelsval[indexs[[i]],]
  prediction = predict(model,newdata = slice_data,type = "response")
  predict_num = round(mean(prediction)*dim(slice_data)[1])
  ### expected number of bookings with children for that fold.
  actual_num = sum(slice_data$children)
  err_result = rbind(err_result,c(actual_num,predict_num))
} 

# The following is the summary of expected number of bookings with children for that 
# fold and actual number

err_result = data.frame(err_result)
colnames(err_result)=c("actual_num","predict_num") 
err_result$difference = err_result$predict_num-err_result$actual_num
print(err_result)

# The following figure demonstrates the distribution of difference beween actual number
# and expected number.
p0 = ggplot(data=err_result) + 
  geom_histogram(aes(x=difference)) 
p0




