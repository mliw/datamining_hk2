# \section{Problem 1: visualization}
# \subsection{line graphs}
### Load the data and package
library(ggplot2)
library(tidyverse)
capmetro = read.csv("data/capmetro.csv")


### Extract the month and year
tem = capmetro$timestamp[1]
month_func = function(my_str){
  return(as.numeric(strsplit(my_str, "-", fixed=TRUE)[[1]][2]))
}
hour_func = function(my_str){
detailed_time = strsplit(my_str, " ", fixed=TRUE)[[1]][2]
hour = as.numeric(strsplit(detailed_time, ":", fixed=TRUE)[[1]][1])
return(hour)
}
capmetro$month = apply(array(capmetro$timestamp),1,month_func)
capmetro$hour = apply(array(capmetro$timestamp),1,hour_func)

### Select data from certain periods
logi = capmetro$month>=9 &  capmetro$month<=11
data = capmetro[logi,]


# We summarize the data and calculate average boardings
### Calculate average boardings
data_summary = data %>%
  group_by(hour,month,day) %>%
  summarize(mean_boarding=mean(boarding))
print(data_summary[1:18,])


# The faceted line plot is as follows.
data_summary$month = factor(data_summary$month)
p0 = ggplot(data=data_summary,aes(x=hour,y=mean_boarding,group=month,color=month))+ geom_line()+facet_wrap(~day, nrow=4)+ggtitle("Mean Boarding in different time intervals")+theme(plot.title = element_text(hjust = 0.5))
p0


# Q1_1_1 Does the hour of peak boardings change from day to day, or is it broadly similar across days?}
# Based on the figure, It is broadly similar across days. At about hour 15.
# Q1_1_2 Why do you think average boardings on Mondays in September look lower, compared to other days and months?
# September is the beginning of Fall semester, which means there is less people on campus. On Monday, perhaps there is less courses than other days.
# Q1_1_3 Similarly, why do you think average boardings on Weds/Thurs/Fri in November look lower?}
# There are many midterm exams in November, which means students stay in the dorm to study for the exams without having to go outside. 

# \subsection{scatter plots}
# We first desigin indicator for weekdays/weekend.(not show)
### is_weekend 1 if it's a weekend.
data$is_weekend = as.numeric(data$day=="Saturday" | data$day=="Sunday")
transform_func = function(num){
  if (num==1){
    return("weekend")
  }
  else{
    return("weekday")
  }
}
data$is_weekend_indicator = apply(array(data$is_weekend),1,transform_func)


# The figure is as follows.
p0 = ggplot(data=data,aes(x=temperature,y=boarding,color=is_weekend_indicator))+
geom_point(alpha=0.3)+facet_wrap(~hour, nrow=4)+
ggtitle("The Relationship between Boarding and Temperature in different Hours")+
theme(plot.title = element_text(hjust = 0.5))
p0

# Q1_2_1 When we hold hour of day and weekend status constant, does temperature seem to have a noticeable effect on the number of UT students riding the bus?}
# Just from the plot above, temperature doesn't have a noticeable effect on the number of UT students riding the bus.

#\section{Problem 2: Saratoga house prices}
#\subsection{The Best Linear Model}

### We first load data
library(tidyverse)
library(ggplot2)
library(rsample)
library(mosaic)
library(ModelMetrics)
data(SaratogaHouses)

# The average of 5-fold corss-validation Rmse is used to evaluate a certain model.
### Metric definition(Use a number to evaluate the performance of a certain model),rmse is adopted.(not show)
library(caret)
evaluate_func = function(my_str,data){
  set.seed(100)
  indexs = createFolds(1:dim(data)[1], k = 5, list = TRUE, returnTrain = FALSE) 
  err_result = c()
  for (i in 1:5){
    test_data = data[indexs[[i]],]
    train_data = data[-indexs[[i]],]  
    lm_model = lm(as.formula(my_str), data=train_data)
    err = rmse(lm_model, test_data)
    err_result = c(err_result,err)
  }
  return(mean(err_result))
}

middle_str = "price ~ lotSize + age + livingArea + bedrooms + 
fireplaces + bathrooms + rooms + heating + fuel + centralAir"
#print(evaluate_func(middle_str,SaratogaHouses))


# The cross-validation Rmse of middle model is 65989.29. Our target is very simple, to find a model with cross-validation rmse lower than 65989.29. A greedy algorithm is used for feature selection, and the results are as follows.
x_names = colnames(SaratogaHouses)[2:length(colnames(SaratogaHouses))]
y_name = "price"
count = 0
rmse_record = Inf
best_name = ""
best_test_str = ""
best_record_previous = Inf
result_collect = c()

while(TRUE){
  for (name in x_names){
    if (count == 0){
     test_str = paste(y_name,"~",name,sep="")
    }
    else{
     test_str = paste(y_name,"+",name,sep="")
    }
    err = evaluate_func(test_str,SaratogaHouses)
    if (err<rmse_record){
      rmse_record = err
      best_name = name
      best_test_str = test_str
    }
  }
  if (rmse_record<best_record_previous){
    best_record_previous = rmse_record
  }
  else{
    break
  }
  y_name = best_test_str
  result_collect = c(result_collect,c(y_name,rmse_record))
  count = count + 1
  x_names = setdiff(x_names,best_name)
  if (length(x_names)==0){
    break
  }
}
print(result_collect)
best_str = "price~livingArea+landValue+bathrooms+waterfront+newConstruction+
heating+lotSize+centralAir+age+rooms+bedrooms+fuel+pctCollege+sewer+fireplaces"
print(evaluate_func(best_str,SaratogaHouses))

# The best variables are:
# "price~livingArea+landValue+bathrooms+waterfront+newConstruction+
# heating+lotSize+centralAir+age+rooms+bedrooms+fuel+pctCollege+sewer+fireplaces"
# Error:57828.05
# Corresponding cross-validation error is 57828.05, lower than 65989.29 from the medium model.
# In all, we successfully overperform the medium model!.
# \subsection{The Best KNN}
# Package kknn is used, and we slightly modify the evaluation function.

library(kknn)
knn_evaluate_func = function(my_str,data){
  set.seed(100)
  indexs = createFolds(1:dim(data)[1], k = 5, list = TRUE, returnTrain = FALSE) 
  err_result = c()
  for (i in 1:5){
    test_data = data[indexs[[i]],]
    train_data = data[-indexs[[i]],]  
    model = train.kknn(as.formula(my_str),train_data,scale = TRUE)
    test_prediction = predict(model,test_data)
    test_actual = test_data$price
    err = rmse(actual=test_actual, predicted=test_prediction)
    err_result = c(err_result,err)
  }
  return(mean(err_result))
}

# Greedy algorithm is adopted again to select the best feature combination, the results are as follows.
x_names = colnames(SaratogaHouses)[2:length(colnames(SaratogaHouses))]
y_name = "price"
count = 0
rmse_record = Inf
best_name = ""
best_test_str = ""
best_record_previous = Inf
result_collect = c()

while(TRUE){
  for (name in x_names){
    if (count == 0){
     test_str = paste(y_name,"~",name,sep="")
    }
    else{
     test_str = paste(y_name,"+",name,sep="")
    }
    err = knn_evaluate_func(test_str,SaratogaHouses)
    if (err<rmse_record){
      rmse_record = err
      best_name = name
      best_test_str = test_str
    }
  }
  if (rmse_record<best_record_previous){
    best_record_previous = rmse_record
  }
  else{
    break
  }
  y_name = best_test_str
  result_collect = c(result_collect,c(y_name,rmse_record))
  count = count + 1
  x_names = setdiff(x_names,best_name)
  if (length(x_names)==0){
    break
  }
}
print(result_collect)

# \subsection{Analysis}

# The best variables and cross-validation error for KNN is
# "price~livingArea+landValue+age+pctCollege+waterfront+newConstruction"
# 58061.7316651973

# The best variables and cross-validation error for linear model is
# "price~livingArea+landValue+bathrooms+waterfront+newConstruction+heating+
# lotSize+centralAir+age+rooms+bedrooms+fuel+pctCollege+sewer+fireplaces"
# 57828.05

# Although the cross-validation error is lower for linear model, I still believe knn
# is better, as it uses only 6 variables to achieve its lowest error.
# 
# Moreover,$\frac{58061.7-57828.05}{57828.05}=0.00404$, not very much.




