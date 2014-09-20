---
title: "PracticalMachineLearning - Prediction Assignment Writeup"
author: "Sivarajan Napoleon"
date: "Sunday, September 21, 2014"
output: pdf_document
---

This is an R Markdown document for the Practical Machine Learning Assignment.
This document describe the analysis done for the prediction assignment of the practical machine learning course.

The first part is about the packages which will be loaded. In addition to caret & randomForest already used on the course, the code uses Hmisc to help me on the data analysis phases & foreach & doParallel to decrease the random forrest processing time by parallelising the operation. To make it better reproductible, the seed valueis set to 2048.

```{r}
install.packages("Hmisc")
install.packages("doParallel")

library(Hmisc)
library(caret)
library(randomForest)
library(foreach)
library(doParallel)
set.seed(2048)
options(warn=-1)
```

The data both from the provided training and test data provided by the course instructions is loaded. Some values that contained a "#DIV/0!" is replaced with an NA value.

```{r}
training_data <- read.csv("pml-training.csv", na.strings=c("#DIV/0!") )
evaluation_data <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!") )

```

Next, cast all columns 8 to the end to be numeric.

```{r}
for(i in c(8:ncol(training_data)-1)) {training_data[,i] = as.numeric(as.character(training_data[,i]))}
for(i in c(8:ncol(evaluation_data)-1)) {evaluation_data[,i] = as.numeric(as.character(evaluation_data[,i]))}
```

Leave the blank columns since these did not contribute well to the prediction. Choose a feature set that only included complete columns. Then remove the user name, timestamps and windows as they are not needed.

Determine the feature set and and display it out my calling feature_set.

```{r}
feature_set <- colnames(training_data[colSums(is.na(training_data)) == 0])[-(1:7)]
model_data <- training_data[feature_set]
feature_set
```

Now the model data built from the feature set is ready. Lets review them.


```{r}
idx <- createDataPartition(y=model_data$classe, p=0.75, list=FALSE )
training <- model_data[idx,]
testing <- model_data[-idx,]
```

In order to analyse the expected output from sample error and estimate the error appropriately with cross-validation, we build 5 random forests with 150 trees each. We make use of parallel processing to build this model. The method uses perform parallel processing with random forests in R, thus providing a great speedup.


```{r}
registerDoParallel()
x <- training[-ncol(training)]
y <- training$classe

rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
randomForest(x, y, ntree=ntree) 
}
```

Now, get the error reports for both training and test data. This will help analyse the expected output from sample error and estimate the error appropriately with cross-validation.


```{r}
predictions1 <- predict(rf, newdata=training)
confusionMatrix(predictions1,training$classe)
predictions2 <- predict(rf, newdata=testing)
confusionMatrix(predictions2,testing$classe)
```

Now prepare the Project Submission using Course provided code.


```{r}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

```

Evaluate the Data, create a feature set, predict the answers, and print the answer.

```{r}
x <- evaluation_data
x <- x[feature_set[feature_set!='classe']]
answers <- predict(rf, newdata=x)

answers

pml_write_files(answers)

```

As it is clearly visible in the result of the confusionmatrix, the model is good and efficient because it has an accuracy of 0.9997 and very good sensitivity & specificity values on the testing dataset. (the lowest value is 0.992 for the sensitivity of the class C)

The course project also gets 100% - 20 out of 20.  The experiments with PCA and other models did not get as good of accuracy.