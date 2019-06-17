##Safety Driver Detection Based on Statistical Feature Extraction with Machine Learning Algorithm

The use of Statistical Feature Extraction in this case is because the data condition is a data stream. The indication of the data stream can be seen in variable seconds. Seconds It refers to the time that the record was created. It starts from the beginning of the trip (beginning from 0).

Generally Statistical Feature Extraction is mostly applied in the case of Signal Processing and Image Processing. But in this case, I tried using Statistical Feature Extraction to create a Machine Learning model.

The software that I use is R. Some of the packages needed are as follows
``` r
library(dplyr) #For data wrangling
library(MLmetrics) #evaluation metrics in machine learning
library(fBasics) #statistical feature extraction
library(caret) 
library(tree)
library(randomForest)
library(e1071)
```


``` r
#combining datasets
dataset <- data.frame()
for (i in 0:9){
  directory <- paste ("D:/GRAB FOR SEA/safety/features/part-0000",i,"-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv",sep="")
  data_temp <- read.csv(directory)
  dataset <- rbind(dataset,data_temp)
}
head(dataset)
```

    ##      bookingID Accuracy Bearing acceleration_x acceleration_y
    ## 1 1.202591e+12    3.000     353     1.22886700       8.900100
    ## 2 2.748779e+11    9.293      17     0.03277481       8.659933
    ## 3 8.847633e+11    3.000     189     1.13967480       9.545974
    ## 4 1.073742e+12    3.900     126     3.87154250      10.386364
    ## 5 1.056562e+12    3.900      50    -0.11288182      10.550960
    ## 6 1.185411e+12    3.900     178     0.80564886       9.206902
    ##   acceleration_z       gyro_x       gyro_y       gyro_z second      Speed
    ## 1      3.9869683  0.008220500  0.002268928 -0.009965830   1362  0.0000000
    ## 2      4.7373000  0.024629302  0.004027889 -0.010858121    257  0.1900000
    ## 3      1.9513340 -0.006899112 -0.015080430  0.001121549    973  0.6670589
    ## 4     -0.1364737  0.001343904 -0.339601370 -0.017956385    902  7.9132853
    ## 5     -1.5601097  0.130568090 -0.061697390  0.161530230    820 20.4194090
    ## 6      2.9544450 -0.057103686 -0.043554686  0.002333503    533 19.2500000

The feature extraction used in this case is:

1.  Mean 
2.  Median 
3.  Standard Deviation 
4.  Skewness 
5.  Kurtosis 

All of feature based on concept "Pearson System" which means :

1.  Central value about which the measurements scatter (mean and median)
2.  How far most of the measurements scatter about the mean (standard deviation)
3.  The degree to which the measurements pile up on only one side of the mean (skewness)
4.  How far rare measurements scatter from the mean (Kurtosis)

``` r
#feature engineering with statistical feature extraction

dataset_clean <- dataset %>%
  select(bookingID, Accuracy, Bearing, acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z, second, Speed) %>%
  group_by(bookingID) %>%
  summarise(
    Accuracy_mean = mean(Accuracy), Accuracy_median = median(Accuracy), Accuracy_sd = sd(Accuracy), 
    Accuracy_skewness = skewness(Accuracy),Accuracy_kurtosis = kurtosis(Accuracy),  
    
    Bearing_mean = mean(Bearing), Bearing_median = median(Bearing), Bearing_sd = sd(Bearing), 
    Bearing_skewness =skewness(Bearing),Bearing_kurtosis = kurtosis(Bearing),  
    
    acceleration_x_mean = mean(acceleration_x), acceleration_x_median = median(acceleration_x),
    acceleration_x_sd= sd(acceleration_x), acceleration_x_skewness = skewness(acceleration_x),
    acceleration_x_kurtosis = kurtosis(acceleration_x),  
    
    acceleration_y_mean = mean(acceleration_y), acceleration_y_median = median(acceleration_y), 
    acceleration_y_sd = sd(acceleration_y), acceleration_y_skewness = skewness(acceleration_y),
    acceleration_y_kurtosis = kurtosis(acceleration_y),  
    
    acceleration_z_mean = mean(acceleration_z), acceleration_z_median = median(acceleration_z), 
    acceleration_z_sd = sd(acceleration_z), acceleration_z_skewness = skewness(acceleration_z),
    acceleration_z_kurtosis = kurtosis(acceleration_z),  
    
    gyro_x_mean = mean(gyro_x), gyro_x_median = median(gyro_x), gyro_x_sd = sd(gyro_x), 
    gyro_x_skewness = skewness(gyro_x),gyro_x_kurtosis = kurtosis(gyro_x),  
    
    gyro_y_mean = mean(gyro_y), gyro_y_median = median(gyro_y), gyro_y_sd = sd(gyro_y), 
    gyro_y_skewness = skewness(gyro_y),gyro_y_kurtosis = kurtosis(gyro_y),  
    
    gyro_z_mean = mean(gyro_z),gyro_z_median = median(gyro_z), gyro_z_sd = sd(gyro_z), 
    gyro_z_skewness = skewness(gyro_z),gyro_z_kurtosis = kurtosis(gyro_z),  
    
    second_mean=mean(second), second_median = median(second), second_sd = sd(second), 
    second_skewness = skewness(second),second_kurtosis = kurtosis(second),  
    
    Speed_mean=mean(Speed), Speed_median = median(Speed), Speed_sd = sd(Speed), 
    Speed_skewness = skewness(Speed),Speed_kurtosis = kurtosis(Speed))

label <- read.csv("D:/GRAB FOR SEA/safety/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv")
dataset_clean <- left_join(dataset_clean,label,by="bookingID")
dataset_clean <- na.omit(dataset_clean)
head(dataset_clean)
```

    ## # A tibble: 6 x 52
    ##   bookingID Accuracy_mean Accuracy_median Accuracy_sd Accuracy_skewne~
    ##       <dbl>         <dbl>           <dbl>       <dbl>            <dbl>
    ## 1         0         10.2             8          3.86              1.73
    ## 2         1          3.72            3.9        0.598             1.26
    ## 3         2          3.93            3.63       1.12              1.32
    ## 4         6          4.59            4.00       1.33              1.70
    ## 5         7          3.68            3.9        0.378            -1.20
    ## 6         8          7.01            6.07       3.15              1.07
    ## # ... with 47 more variables: Accuracy_kurtosis <dbl>, Bearing_mean <dbl>,
    ## #   Bearing_median <dbl>, Bearing_sd <dbl>, Bearing_skewness <dbl>,
    ## #   Bearing_kurtosis <dbl>, acceleration_x_mean <dbl>,
    ## #   acceleration_x_median <dbl>, acceleration_x_sd <dbl>,
    ## #   acceleration_x_skewness <dbl>, acceleration_x_kurtosis <dbl>,
    ## #   acceleration_y_mean <dbl>, acceleration_y_median <dbl>,
    ## #   acceleration_y_sd <dbl>, acceleration_y_skewness <dbl>,
    ## #   acceleration_y_kurtosis <dbl>, acceleration_z_mean <dbl>,
    ## #   acceleration_z_median <dbl>, acceleration_z_sd <dbl>,
    ## #   acceleration_z_skewness <dbl>, acceleration_z_kurtosis <dbl>,
    ## #   gyro_x_mean <dbl>, gyro_x_median <dbl>, gyro_x_sd <dbl>,
    ## #   gyro_x_skewness <dbl>, gyro_x_kurtosis <dbl>, gyro_y_mean <dbl>,
    ## #   gyro_y_median <dbl>, gyro_y_sd <dbl>, gyro_y_skewness <dbl>,
    ## #   gyro_y_kurtosis <dbl>, gyro_z_mean <dbl>, gyro_z_median <dbl>,
    ## #   gyro_z_sd <dbl>, gyro_z_skewness <dbl>, gyro_z_kurtosis <dbl>,
    ## #   second_mean <dbl>, second_median <dbl>, second_sd <dbl>,
    ## #   second_skewness <dbl>, second_kurtosis <dbl>, Speed_mean <dbl>,
    ## #   Speed_median <dbl>, Speed_sd <dbl>, Speed_skewness <dbl>,
    ## #   Speed_kurtosis <dbl>, label <int>


In this case there are 4 machine learning algorithms that are tested, namely

1. Naive Bayes Classifier 
2. Logistic Regression 
3. Random Forest 
4. Decission Tree 


Evaluation of machine learning models using AUC with the scheme training and testing data using k-fold cross validation. The number of folds used is 5.

``` r
#this is stratified k fold cross validation with k = 5
k_folds_0 <- createFolds(data.frame(subset(dataset_clean,label==0))[,1], k=5, list=T)
k_folds_1 <- createFolds(data.frame(subset(dataset_clean,label==1))[,1], k=5, list=T)
```

``` r
auc_nbc<-matrix(ncol=1,nrow=5)
auc_glm<-matrix(ncol=1,nrow=5)
auc_rf<-matrix(ncol=1,nrow=5)
auc_tree<-matrix(ncol=1,nrow=5) 
dataset_clean <- dataset_clean[,-1] #exclude booking_id variable
for (i in 1:5){
  #create data training and data testing for each fold
  
  train <- data.frame(rbind(dataset_clean[-k_folds_0[[i]],],dataset_clean[-k_folds_1[[i]],])) #data training
  test <- data.frame(rbind(dataset_clean[k_folds_0[[i]],],dataset_clean[k_folds_1[[i]],])) #data testing
  
  #modelling using naive bayes classifier
  model_nbc <- naiveBayes(x=train[,1:50], y=as.factor(train[,51]))
  prob_nbc <- predict(model_nbc, test[,1:50], type="raw")
  auc_nbc[i,] <- AUC(prob_nbc[,2],test[,51])
  
  #modelling using logistic regression
  model_glm <- glm(label~.,data=train,binomial(link = "logit"))
  prob_glm <- predict(model_glm, test)
  auc_glm[i,] <- AUC(prob_glm,test[,51])
  
  #modelling using logistic regression
  model_rf <- randomForest(train[,1:50], as.factor(train[,51]))
  prob_rf <- (predict(model_rf, test[,1:50],type="prob"))[,2]
  auc_rf[i,] <- AUC(prob_rf,test[,51])
  
  #modelling using decision tree
  model_tree <- tree(label~.,data=train)
  prob_tree <- predict(model_tree, test[,1:50])
  auc_tree[i,] <- AUC(prob_tree,test[,51])
 
}
```


``` r
Eval_full <- data.frame(auc_nbc,auc_glm,auc_rf,auc_tree) #evaluation result for data testing each fold
Mean_auc_full <- matrix(cbind(colMeans(Eval_full)),nrow=1,ncol=4)
colnames(Mean_auc_full) <- c("Naive_Bayes_Classifier","Logistic_Regression","Random_Forest","Decision_Tree")
```

The results of the evaluation model of some machine learning algorithms above can be seen in the results below

``` r
Eval_full #auc for each fold
```

    ##     auc_nbc   auc_glm    auc_rf  auc_tree
    ## 1 0.6418020 0.7174615 0.9949187 0.6712740
    ## 2 0.6404941 0.7209973 0.9923155 0.6722523
    ## 3 0.6377376 0.7228167 0.9944071 0.6659600
    ## 4 0.6482296 0.7247590 0.9925348 0.6802225
    ## 5 0.6383961 0.7304030 0.9946284 0.6849312

``` r
Mean_auc_full 
```

    ##      Naive_Bayes_Classifier Logistic_Regression Random_Forest
    ## [1,]              0.6413319           0.7232875     0.9937609
    ##      Decision_Tree
    ## [1,]      0.674928

The best model is random forest. So we can use 'model_rf' for classify safety driver data.
