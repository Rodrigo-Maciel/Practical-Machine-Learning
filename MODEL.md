################################################ LOADING PACKAGES ################################################
library(dplyr)
library(caret)
library(rattle)



################################################ LOADING THE DATA ################################################
# SET WORKING DIRECTORY
setwd("~/Coursera/Practical Machine Learning (ML en R)/Proyecto Final")

database = read.csv(file = "pml-training.csv")



################################################ PREPARING THE DATA ################################################
# Drop user_name, new_window cvtd_timestamp, to predict "CLASSE" in base to continuos and measurements data (not in base of user or time of day)
database = subset(database, select = -c(user_name, cvtd_timestamp, new_window))

# Counting missing data
sapply(database, function(x) sum(is.na(x)))

# Drop variables with +5% of missing data, filling this variables would change the model's outcome
database = subset(database, select = -c(X, max_roll_belt, max_picth_belt, min_roll_belt, min_pitch_belt, amplitude_roll_belt, amplitude_pitch_belt,
                                        var_total_accel_belt, avg_roll_belt, stddev_roll_belt, var_roll_belt, avg_pitch_belt, stddev_pitch_belt,
                                        var_pitch_belt, avg_yaw_belt, stddev_yaw_belt, var_yaw_belt, var_accel_arm, avg_roll_arm, stddev_roll_arm,
                                        var_roll_arm, avg_pitch_arm, stddev_pitch_arm, var_pitch_arm, avg_yaw_arm, stddev_yaw_arm, var_yaw_arm,
                                        max_roll_arm, max_picth_arm, max_yaw_arm, min_roll_arm, min_pitch_arm, min_yaw_arm, amplitude_roll_arm,
                                        amplitude_pitch_arm, amplitude_yaw_arm, max_roll_dumbbell, max_picth_dumbbell, min_roll_dumbbell, min_pitch_dumbbell,
                                        amplitude_roll_dumbbell, amplitude_pitch_dumbbell, var_accel_dumbbell, avg_roll_dumbbell, stddev_roll_dumbbell,
                                        var_roll_dumbbell, avg_pitch_dumbbell, stddev_pitch_dumbbell, var_pitch_dumbbell, avg_yaw_dumbbell,
                                        stddev_yaw_dumbbell,  max_roll_forearm, max_picth_forearm, min_roll_forearm, min_pitch_forearm,
                                        amplitude_roll_forearm, amplitude_roll_forearm, amplitude_pitch_forearm, var_accel_forearm, avg_roll_forearm,
                                        stddev_roll_forearm, var_roll_forearm, avg_pitch_forearm, stddev_pitch_forearm, var_pitch_forearm,
                                        avg_yaw_forearm, stddev_yaw_forearm, var_yaw_forearm, kurtosis_roll_belt, kurtosis_picth_belt, kurtosis_yaw_belt, 
                                        skewness_roll_belt, skewness_roll_belt.1, skewness_yaw_belt, kurtosis_roll_arm, kurtosis_picth_arm, 
                                        kurtosis_yaw_arm, skewness_roll_arm, skewness_pitch_arm, skewness_yaw_arm, kurtosis_roll_dumbbell, 
                                        kurtosis_picth_dumbbell, kurtosis_yaw_dumbbell, skewness_roll_dumbbell, skewness_pitch_dumbbell,
                                        skewness_yaw_dumbbell, kurtosis_roll_forearm, kurtosis_picth_forearm, kurtosis_yaw_forearm, skewness_roll_forearm,
                                        skewness_pitch_forearm, skewness_yaw_forearm, max_yaw_belt, min_yaw_belt, amplitude_yaw_belt, max_yaw_dumbbell,
                                        min_yaw_dumbbell, amplitude_yaw_dumbbell, max_yaw_forearm, min_yaw_forearm, amplitude_yaw_forearm,
                                        var_yaw_dumbbell))


# FEATURE SELECTION (FILTER METHOD)
# CORRELATION (PCA, REDUCE DIMENSIONALITY KEEPING MAYOR PART OF THE INFORMATION)
database_correlation = subset(database, select = -c(classe))
correlation = cor(database_correlation)

# VARIABLES WITH ABSOLUTE VALUE OF CORRELATION EQUAL OR BIGGER TO 0.8
# roll_belt, yaw_belt, total_accel_belt, accel_belt_y, accel_belt_z
v1 = prcomp(database[c(4, 6, 7, 12, 13)]) 
summary(v1) #Component 1-90%
v1[2]
database[,57] = database[,4]*0.42079160 + database[,6]*0.59154028 + database[,7]*0.05024761 + database[,12]*0.16805308 + database[,13]*-0.66501720

# pitch_belt, accel_belt_x, magnet_belt_x
v2 = prcomp(database[c(5, 11, 14)]) 
summary(v2) #Component 1-72%, 2-14%
v2[2]
database[,58] = database[,5]*-0.2851349 + database[,11]*0.3816465 + database[,14]*0.8792292
database[,59] = database[,5]*-0.5242966 + database[,11]*0.7058034 + database[,14]*-0.4763975

# gyros_arm_x, gyros_arm_y
v3 = prcomp(database[c(21, 22)])
summary(v3) #Component 1-97%
v3[2]
database[,60] = database[,21]*0.9278154 + database[,22]*-0.3730396

# accel_arm_x, magnet_arm_x
v4 = prcomp(database[c(24, 27)])
summary(v4) #Component 1-95%
v4[2]
database[,61] = database[,24]*0.3320298 + database[,27]*0.9432689

# magnet_arm_y, magnet_arm_z
v5 = prcomp(database[c(28, 29)])
summary(v5) #Component 1-92%
v5[2]
database[,62] = database[,28]*-0.488373 + database[,29]*-0.872635

# yaw_dumbbell, accel_dumbbell_z
v6 = prcomp(database[c(32, 39)])
summary(v6) #Component 1-93%
v6[2]
database[,63] = database[,32]*0.5832650 + database[,39]*0.8122819

# gyros_dumbbell_x, gyros_dumbbell_z, gyros_forearm_z
v7 = prcomp(database[c(34, 36, 49)])
summary(v7) #Component 1-96%
v7[2]
database[,64] = database[,34]*0.4633751 + database[,36]*-0.7105826 + database[,49]*-0.5294864

# Drop variables that were transformed via PCA
database = subset(database, select = -c(roll_belt, yaw_belt, total_accel_belt, accel_belt_y, accel_belt_z, pitch_belt, accel_belt_x, magnet_belt_x,
                                        gyros_arm_x, gyros_arm_y, accel_arm_x, magnet_arm_x, magnet_arm_y, magnet_arm_z, yaw_dumbbell, accel_dumbbell_z,
                                        gyros_dumbbell_x, gyros_dumbbell_z, gyros_forearm_z))



# REORDER DATABASE
database = database[,c(1,2,3,4,5,6,7,8,9,10,
                       11,12,13,14,15,16,17,18,19,20,
                       21,22,23,24,25,26,27,28,29,30,
                       31,32,33,34,35,36,38,39,40,
                       41,42,43,44,45,
                       37)]



# Estandarizacion de datos
estandarized_matrix = matrix(data = NA, nrow = 44, ncol = 2)
colnames(estandarized_matrix) = c("MEAN", "SD")

for(i in 1:44){
  estandarized_matrix[i,1] = mean(database[,i])
  estandarized_matrix[i,2] = sd(database[,i])
}

estandarized_database = database[,45]
estandarized_database = as.data.frame(estandarized_database)

for(i in 1:44){
  estandarized_database[,i+1] = (database[,i]-estandarized_matrix[i,1])/estandarized_matrix[i,2]
}

colnames(estandarized_database) = c("classe", "raw_timestamp_part_1", "raw_timestamp_part_2", "num_window", "gyros_belt_x", 
                                    "gyros_belt_y", "gyros_belt_z", "magnet_belt_y", "magnet_belt_z", "roll_arm", "pitch_arm", "yaw_arm",
                                    "total_accel_arm", "gyros_arm_z", "accel_arm_y", "accel_arm_z", "roll_dumbbell", "pitch_dumbbell",
                                    "total_accel_dumbbell", "gyros_dumbbell_y", "accel_dumbbell_x", "accel_dumbbell_y", "magnet_dumbbell_x",
                                    "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", "yaw_forearm",
                                    "total_accel_forearm", "gyros_forearm_x", "gyros_forearm_y", "accel_forearm_x",
                                    "accel_forearm_y", "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z",
                                    "V57", "V58", "V59", "V60", "V61", "V62", "V63", "V64")


# RANDOMLY SHUFFLE DATABASE (BECAUSE CLASSE IS IN ORDER)
rows <- sample(nrow(estandarized_database))
estandarized_database = estandarized_database[rows,]



# DATA SLICING (5 FOLD CROSS VALDATION)
train1 = estandarized_database[-(1:3924),]
train2 = estandarized_database[-(3925:7848),]
train3 = estandarized_database[-(7849:11772),]
train4 = estandarized_database[-(11773:15696),]
train5 = estandarized_database[-(15697:19622),]

test1 = estandarized_database[1:3924,]
test2 = estandarized_database[3925:7848,]
test3 = estandarized_database[7849:11772,]
test4 = estandarized_database[11773:15696,]
test5 = estandarized_database[15697:19622,]



# SELECT VARIABLES TU USE
# WE TRAIN 5 DECISION TREES, EACH WITH A DIFFERENT TRAINING SET AND WID THIS WE DECIDE WICH VARIABLES CAN HEL US PREDICT CLASSE
# THIS WILL MAKE A SIMPLER MODEL, EASIER TO INTERPRET, FASTER TO TRAIN, AND LESS SUSCEPTIBLE TO OVERFITTING
# WE KEEP ANY VARIABLE THAT APPEARS IMPORTANT IN ALL  MODELS
model1 = train(classe ~., method="rpart", data=train1)
model2 = train(classe ~., method="rpart", data=train2)
model3 = train(classe ~., method="rpart", data=train3)
model4 = train(classe ~., method="rpart", data=train4)
model5 = train(classe ~., method="rpart", data=train5)

importance1 <- varImp(model1)
importance2 <- varImp(model2)
importance3 <- varImp(model3)
importance4 <- varImp(model4)
importance5 <- varImp(model5)

print(importance1)
print(importance2)
print(importance3)
print(importance4)
print(importance5)



# DATA SLICING (RELEVANT VARIABLES)
train1 = train1[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]
train2 = train2[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]
train3 = train3[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]
train4 = train4[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]
train5 = train5[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]

test1 = test1[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]
test2 = test2[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]
test3 = test3[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]
test4 = test4[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]
test5 = test5[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]



# DECISION TREE MODEL
model1 = train(classe ~., method="rpart", data=train1)
model2 = train(classe ~., method="rpart", data=train2)
model3 = train(classe ~., method="rpart", data=train3)
model4 = train(classe ~., method="rpart", data=train4)
model5 = train(classe ~., method="rpart", data=train5)



# Prediction
test1[,16] = predict(model1,test1)
test2[,16] = predict(model2,test2)
test3[,16] = predict(model3,test3)
test4[,16] = predict(model4,test4)
test5[,16] = predict(model5,test5)



# BALANCED DATASET (A:29%, B:20%, C:17%, D:17%, E:18% OF THE TRAINING DATASET) WE WILL USE ACCURACY AS METRIC
count1=0
count2=0
count3=0
count4=0
count5=0

for(i in 1:3924){
  if(test1[i,1] == test1[i,16]){
    count1 = count1+1
  }
  
  if(test2[i,1] == test2[i,16]){
    count2 = count2+1
  }
  
  if(test3[i,1] == test3[i,16]){
    count3 = count3+1
  }
  
  if(test4[i,1] == test4[i,16]){
    count4 = count4+1
  }
}

for(i in 1:3926){
  if(test5[i,1] == test5[i,16]){
    count5 = count5+1
  }
}



# EVALUATION METRICS
metrics = matrix(NA, nrow = 5, ncol = 1)

metrics[1,1] = count1/3924
metrics[2,1] = count2/3924
metrics[3,1] = count3/3924
metrics[4,1] = count4/3924
metrics[5,1] = count5/3926

### MEAN = 0.5, SD = 0.015
mean(metrics[,1])
sd(metrics[,1])



### ANALISIS DE ARBOLES RESULTANTES
fancyRpartPlot(model1$finalModel)
fancyRpartPlot(model2$finalModel)
fancyRpartPlot(model3$finalModel)
fancyRpartPlot(model4$finalModel)
fancyRpartPlot(model5$finalModel)



##### PITCH_FOREARM UNDER -1.6, 99% OF THE DATA BELONGS TO CLASSE A, SO WE PREDICT THIS BEFORE RUNNING DECISION TREE
train1a= train1 %>% filter(pitch_forearm >= -1.6)
train1b= train1 %>% filter(pitch_forearm < -1.6)
train2a= train2 %>% filter(pitch_forearm >= -1.6)
train2b= train2 %>% filter(pitch_forearm < -1.6)
train3a= train3 %>% filter(pitch_forearm >= -1.6)
train3b= train3 %>% filter(pitch_forearm < -1.6)
train4a= train4 %>% filter(pitch_forearm >= -1.6)
train4b= train4 %>% filter(pitch_forearm < -1.6)
train5a= train5 %>% filter(pitch_forearm >= -1.6)
train5b= train5 %>% filter(pitch_forearm < -1.6)

test1a= test1 %>% filter(pitch_forearm >= -1.6)
test1b= test1 %>% filter(pitch_forearm < -1.6)
test2a= test2 %>% filter(pitch_forearm >= -1.6)
test2b= test2 %>% filter(pitch_forearm < -1.6)
test3a= test3 %>% filter(pitch_forearm >= -1.6)
test3b= test3 %>% filter(pitch_forearm < -1.6)
test4a= test4 %>% filter(pitch_forearm >= -1.6)
test4b= test4 %>% filter(pitch_forearm < -1.6)
test5a= test5 %>% filter(pitch_forearm >= -1.6)
test5b= test5 %>% filter(pitch_forearm < -1.6)

test1b[,16]="A"
test2b[,16]="A"
test3b[,16]="A"
test4b[,16]="A"
test5b[,16]="A"

train1a1= train1a %>% filter(magnet_belt_y >= -1.1)
train1a2= train1a %>% filter(magnet_belt_y < -1.1)
train2a1= train2a %>% filter(magnet_belt_y >= -1.1)
train2a2= train2a %>% filter(magnet_belt_y < -1.1)
train3a1= train3a %>% filter(magnet_belt_y >= -1.1)
train3a2= train3a %>% filter(magnet_belt_y < -1.1)
train4a1= train4a %>% filter(magnet_belt_y >= -1.1)
train4a2= train4a %>% filter(magnet_belt_y < -1.1)
train5a1= train5a %>% filter(magnet_belt_y >= -1.1)
train5a2= train5a %>% filter(magnet_belt_y < -1.1)

test1a1= test1a %>% filter(magnet_belt_y >= -1.1)
test1a2= test1a %>% filter(magnet_belt_y < -1.1)
test2a1= test2a %>% filter(magnet_belt_y >= -1.1)
test2a2= test2a %>% filter(magnet_belt_y < -1.1)
test3a1= test3a %>% filter(magnet_belt_y >= -1.1)
test3a2= test3a %>% filter(magnet_belt_y < -1.1)
test4a1= test4a %>% filter(magnet_belt_y >= -1.1)
test4a2= test4a %>% filter(magnet_belt_y < -1.1)
test5a1= test5a %>% filter(magnet_belt_y >= -1.1)
test5a2= test5a %>% filter(magnet_belt_y < -1.1)

model1a2 = train(classe ~., method="rpart", data=train1a2)
model2a2 = train(classe ~., method="rpart", data=train2a2)
model3a2 = train(classe ~., method="rpart", data=train3a2)
model4a2 = train(classe ~., method="rpart", data=train4a2)
model5a2 = train(classe ~., method="rpart", data=train5a2)
test1a2[,16] = predict(model1a2,test1a2)
test2a2[,16] = predict(model2a2,test2a2)
test3a2[,16] = predict(model3a2,test3a2)
test4a2[,16] = predict(model4a2,test4a2)
test5a2[,16] = predict(model5a2,test5a2)

train1a1a= train1a1 %>% filter(magnet_dumbbell_y >= 0.66)
train1a1b= train1a1 %>% filter(magnet_dumbbell_y < 0.66)
train2a1a= train2a1 %>% filter(magnet_dumbbell_y >= 0.66)
train2a1b= train2a1 %>% filter(magnet_dumbbell_y < 0.66)
train3a1a= train3a1 %>% filter(magnet_dumbbell_y >= 0.66)
train3a1b= train3a1 %>% filter(magnet_dumbbell_y < 0.66)
train4a1a= train4a1 %>% filter(magnet_dumbbell_y >= 0.66)
train4a1b= train4a1 %>% filter(magnet_dumbbell_y < 0.66)
train5a1a= train5a1 %>% filter(magnet_dumbbell_y >= 0.66)
train5a1b= train5a1 %>% filter(magnet_dumbbell_y < 0.66)

test1a1a= test1a1 %>% filter(magnet_dumbbell_y >= 0.66)
test1a1b= test1a1 %>% filter(magnet_dumbbell_y < 0.66)
test2a1a= test2a1 %>% filter(magnet_dumbbell_y >= 0.66)
test2a1b= test2a1 %>% filter(magnet_dumbbell_y < 0.66)
test3a1a= test3a1 %>% filter(magnet_dumbbell_y >= 0.66)
test3a1b= test3a1 %>% filter(magnet_dumbbell_y < 0.66)
test4a1a= test4a1 %>% filter(magnet_dumbbell_y >= 0.66)
test4a1b= test4a1 %>% filter(magnet_dumbbell_y < 0.66)
test5a1a= test5a1 %>% filter(magnet_dumbbell_y >= 0.66)
test5a1b= test5a1 %>% filter(magnet_dumbbell_y < 0.66)

model1a1a = train(classe ~., method="rpart", data=train1a1a)
model2a1a = train(classe ~., method="rpart", data=train2a1a)
model3a1a = train(classe ~., method="rpart", data=train3a1a)
model4a1a = train(classe ~., method="rpart", data=train4a1a)
model5a1a = train(classe ~., method="rpart", data=train5a1a)
test1a1a[,16] = predict(model1a1a,test1a1a)
test2a1a[,16] = predict(model2a1a,test2a1a)
test3a1a[,16] = predict(model3a1a,test3a1a)
test4a1a[,16] = predict(model4a1a,test4a1a)
test5a1a[,16] = predict(model5a1a,test5a1a)

train1a1b1 = train1a1b %>% filter(roll_forearm >= 0.82)
train1a1b2 = train1a1b %>% filter(roll_forearm < 0.82)
train2a1b1 = train2a1b %>% filter(roll_forearm >= 0.82)
train2a1b2 = train2a1b %>% filter(roll_forearm < 0.82)
train3a1b1 = train3a1b %>% filter(roll_forearm >= 0.82)
train3a1b2 = train3a1b %>% filter(roll_forearm < 0.82)
train4a1b1 = train4a1b %>% filter(roll_forearm >= 0.82)
train4a1b2 = train4a1b %>% filter(roll_forearm < 0.82)
train5a1b1 = train5a1b %>% filter(roll_forearm >= 0.82)
train5a1b2 = train5a1b %>% filter(roll_forearm < 0.82)

test1a1b1 = test1a1b %>% filter(roll_forearm >= 0.82)
test1a1b2 = test1a1b %>% filter(roll_forearm < 0.82)
test2a1b1 = test2a1b %>% filter(roll_forearm >= 0.82)
test2a1b2 = test2a1b %>% filter(roll_forearm < 0.82)
test3a1b1 = test3a1b %>% filter(roll_forearm >= 0.82)
test3a1b2 = test3a1b %>% filter(roll_forearm < 0.82)
test4a1b1 = test4a1b %>% filter(roll_forearm >= 0.82)
test4a1b2 = test4a1b %>% filter(roll_forearm < 0.82)
test5a1b1 = test5a1b %>% filter(roll_forearm >= 0.82)
test5a1b2 = test5a1b %>% filter(roll_forearm < 0.82)

train1a1b1a = train1a1b1 %>% filter(raw_timestamp_part_1 >= -1.6)
train1a1b1b = train1a1b1 %>% filter(raw_timestamp_part_1 < -1.6)
train2a1b1a = train2a1b1 %>% filter(raw_timestamp_part_1 >= -1.6)
train2a1b1b = train2a1b1 %>% filter(raw_timestamp_part_1 < -1.6)
train3a1b1a = train3a1b1 %>% filter(raw_timestamp_part_1 >= -1.6)
train3a1b1b = train3a1b1 %>% filter(raw_timestamp_part_1 < -1.6)
train4a1b1a = train4a1b1 %>% filter(raw_timestamp_part_1 >= -1.6)
train4a1b1b = train4a1b1 %>% filter(raw_timestamp_part_1 < -1.6)
train5a1b1a = train5a1b1 %>% filter(raw_timestamp_part_1 >= -1.6)
train5a1b1b = train5a1b1 %>% filter(raw_timestamp_part_1 < -1.6)

test1a1b1a = test1a1b1 %>% filter(raw_timestamp_part_1 >= -1.6)
test1a1b1b = test1a1b1 %>% filter(raw_timestamp_part_1 < -1.6)
test2a1b1a = test2a1b1 %>% filter(raw_timestamp_part_1 >= -1.6)
test2a1b1b = test2a1b1 %>% filter(raw_timestamp_part_1 < -1.6)
test3a1b1a = test3a1b1 %>% filter(raw_timestamp_part_1 >= -1.6)
test3a1b1b = test3a1b1 %>% filter(raw_timestamp_part_1 < -1.6)
test4a1b1a = test4a1b1 %>% filter(raw_timestamp_part_1 >= -1.6)
test4a1b1b = test4a1b1 %>% filter(raw_timestamp_part_1 < -1.6)
test5a1b1a = test5a1b1 %>% filter(raw_timestamp_part_1 >= -1.6)
test5a1b1b = test5a1b1 %>% filter(raw_timestamp_part_1 < -1.6)

model1a1b1a = train(classe ~., method="rpart", data=train1a1b1a)
model1a1b1b = train(classe ~., method="rpart", data=train1a1b1b)
model2a1b1a = train(classe ~., method="rpart", data=train2a1b1a)
model2a1b1b = train(classe ~., method="rpart", data=train2a1b1b)
model3a1b1a = train(classe ~., method="rpart", data=train3a1b1a)
model3a1b1b = train(classe ~., method="rpart", data=train3a1b1b)
model4a1b1a = train(classe ~., method="rpart", data=train4a1b1a)
model4a1b1b = train(classe ~., method="rpart", data=train4a1b1b)
model5a1b1a = train(classe ~., method="rpart", data=train5a1b1a)
model5a1b1b = train(classe ~., method="rpart", data=train5a1b1b)

test1a1b1a[,16] = predict(model1a1b1a,test1a1b1a)
test1a1b1b[,16] = predict(model1a1b1b,test1a1b1b)
test2a1b1a[,16] = predict(model2a1b1a,test2a1b1a)
test2a1b1b[,16] = predict(model2a1b1b,test2a1b1b)
test3a1b1a[,16] = predict(model3a1b1a,test3a1b1a)
test3a1b1b[,16] = predict(model3a1b1b,test3a1b1b)
test4a1b1a[,16] = predict(model4a1b1a,test4a1b1a)
test4a1b1b[,16] = predict(model4a1b1b,test4a1b1b)
test5a1b1a[,16] = predict(model5a1b1a,test5a1b1a)
test5a1b1b[,16] = predict(model5a1b1b,test5a1b1b)

train1a1b2a = train1a1b2 %>% filter(magnet_dumbbell_z >= -0.5)
train1a1b2b = train1a1b2 %>% filter(magnet_dumbbell_z < -0.5)
train2a1b2a = train2a1b2 %>% filter(magnet_dumbbell_z >= -0.5)
train2a1b2b = train2a1b2 %>% filter(magnet_dumbbell_z < -0.5)
train3a1b2a = train3a1b2 %>% filter(magnet_dumbbell_z >= -0.5)
train3a1b2b = train3a1b2 %>% filter(magnet_dumbbell_z < -0.5)
train4a1b2a = train4a1b2 %>% filter(magnet_dumbbell_z >= -0.5)
train4a1b2b = train4a1b2 %>% filter(magnet_dumbbell_z < -0.5)
train5a1b2a = train5a1b2 %>% filter(magnet_dumbbell_z >= -0.5)
train5a1b2b = train5a1b2 %>% filter(magnet_dumbbell_z < -0.5)

test1a1b2a = test1a1b2 %>% filter(magnet_dumbbell_z >= -0.5)
test1a1b2b = test1a1b2 %>% filter(magnet_dumbbell_z < -0.5)
test2a1b2a = test2a1b2 %>% filter(magnet_dumbbell_z >= -0.5)
test2a1b2b = test2a1b2 %>% filter(magnet_dumbbell_z < -0.5)
test3a1b2a = test3a1b2 %>% filter(magnet_dumbbell_z >= -0.5)
test3a1b2b = test3a1b2 %>% filter(magnet_dumbbell_z < -0.5)
test4a1b2a = test4a1b2 %>% filter(magnet_dumbbell_z >= -0.5)
test4a1b2b = test4a1b2 %>% filter(magnet_dumbbell_z < -0.5)
test5a1b2a = test5a1b2 %>% filter(magnet_dumbbell_z >= -0.5)
test5a1b2b = test5a1b2 %>% filter(magnet_dumbbell_z < -0.5)

train1a1b2b1 = train1a1b2b %>% filter(raw_timestamp_part_1 >= 1.3)
train1a1b2b2 = train1a1b2b %>% filter(raw_timestamp_part_1 < 1.3)
train2a1b2b1 = train2a1b2b %>% filter(raw_timestamp_part_1 >= 1.3)
train2a1b2b2 = train2a1b2b %>% filter(raw_timestamp_part_1 < 1.3)
train3a1b2b1 = train3a1b2b %>% filter(raw_timestamp_part_1 >= 1.3)
train3a1b2b2 = train3a1b2b %>% filter(raw_timestamp_part_1 < 1.3)
train4a1b2b1 = train4a1b2b %>% filter(raw_timestamp_part_1 >= 1.3)
train4a1b2b2 = train4a1b2b %>% filter(raw_timestamp_part_1 < 1.3)
train5a1b2b1 = train5a1b2b %>% filter(raw_timestamp_part_1 >= 1.3)
train5a1b2b2 = train5a1b2b %>% filter(raw_timestamp_part_1 < 1.3)

test1a1b2b1 = test1a1b2b %>% filter(raw_timestamp_part_1 >= 1.3)
test1a1b2b2 = test1a1b2b %>% filter(raw_timestamp_part_1 < 1.3)
test2a1b2b1 = test2a1b2b %>% filter(raw_timestamp_part_1 >= 1.3)
test2a1b2b2 = test2a1b2b %>% filter(raw_timestamp_part_1 < 1.3)
test3a1b2b1 = test3a1b2b %>% filter(raw_timestamp_part_1 >= 1.3)
test3a1b2b2 = test3a1b2b %>% filter(raw_timestamp_part_1 < 1.3)
test4a1b2b1 = test4a1b2b %>% filter(raw_timestamp_part_1 >= 1.3)
test4a1b2b2 = test4a1b2b %>% filter(raw_timestamp_part_1 < 1.3)
test5a1b2b1 = test5a1b2b %>% filter(raw_timestamp_part_1 >= 1.3)
test5a1b2b2 = test5a1b2b %>% filter(raw_timestamp_part_1 < 1.3)

model1a1b2b1 = train(classe ~., method="rpart", data=train1a1b2b1)
model1a1b2b2 = train(classe ~., method="rpart", data=train1a1b2b2)
model2a1b2b1 = train(classe ~., method="rpart", data=train2a1b2b1)
model2a1b2b2 = train(classe ~., method="rpart", data=train2a1b2b2)
model3a1b2b1 = train(classe ~., method="rpart", data=train3a1b2b1)
model3a1b2b2 = train(classe ~., method="rpart", data=train3a1b2b2)
model4a1b2b1 = train(classe ~., method="rpart", data=train4a1b2b1)
model4a1b2b2 = train(classe ~., method="rpart", data=train4a1b2b2)
model5a1b2b1 = train(classe ~., method="rpart", data=train5a1b2b1)
model5a1b2b2 = train(classe ~., method="rpart", data=train5a1b2b2)

test1a1b2b1[,16] = predict(model1a1b2b1,test1a1b2b1)
test1a1b2b2[,16] = predict(model1a1b2b2,test1a1b2b2)
test2a1b2b1[,16] = predict(model2a1b2b1,test2a1b2b1)
test2a1b2b2[,16] = predict(model2a1b2b2,test2a1b2b2)
test3a1b2b1[,16] = predict(model3a1b2b1,test3a1b2b1)
test3a1b2b2[,16] = predict(model3a1b2b2,test3a1b2b2)
test4a1b2b1[,16] = predict(model4a1b2b1,test4a1b2b1)
test4a1b2b2[,16] = predict(model4a1b2b2,test4a1b2b2)
test5a1b2b1[,16] = predict(model5a1b2b1,test5a1b2b1)
test5a1b2b2[,16] = predict(model5a1b2b2,test5a1b2b2)

train1a1b2a1 = train1a1b2a %>% filter(num_window >= -0.76)
train1a1b2a2 = train1a1b2a %>% filter(num_window < -0.76)
train2a1b2a1 = train2a1b2a %>% filter(num_window >= -0.76)
train2a1b2a2 = train2a1b2a %>% filter(num_window < -0.76)
train3a1b2a1 = train3a1b2a %>% filter(num_window >= -0.76)
train3a1b2a2 = train3a1b2a %>% filter(num_window < -0.76)
train4a1b2a1 = train4a1b2a %>% filter(num_window >= -0.76)
train4a1b2a2 = train4a1b2a %>% filter(num_window < -0.76)
train5a1b2a1 = train5a1b2a %>% filter(num_window >= -0.76)
train5a1b2a2 = train5a1b2a %>% filter(num_window < -0.76)

test1a1b2a1 = test1a1b2a %>% filter(num_window >= -0.76)
test1a1b2a2 = test1a1b2a %>% filter(num_window < -0.76)
test2a1b2a1 = test2a1b2a %>% filter(num_window >= -0.76)
test2a1b2a2 = test2a1b2a %>% filter(num_window < -0.76)
test3a1b2a1 = test3a1b2a %>% filter(num_window >= -0.76)
test3a1b2a2 = test3a1b2a %>% filter(num_window < -0.76)
test4a1b2a1 = test4a1b2a %>% filter(num_window >= -0.76)
test4a1b2a2 = test4a1b2a %>% filter(num_window < -0.76)
test5a1b2a1 = test5a1b2a %>% filter(num_window >= -0.76)
test5a1b2a2 = test5a1b2a %>% filter(num_window < -0.76)

model1a1b2a2 = train(classe ~., method="rpart", data=train1a1b2a2)
model2a1b2a2 = train(classe ~., method="rpart", data=train2a1b2a2)
model3a1b2a2 = train(classe ~., method="rpart", data=train3a1b2a2)
model4a1b2a2 = train(classe ~., method="rpart", data=train4a1b2a2)
model5a1b2a2 = train(classe ~., method="rpart", data=train5a1b2a2)

test1a1b2a2[,16] = predict(model1a1b2a2,test1a1b2a2)
test2a1b2a2[,16] = predict(model2a1b2a2,test2a1b2a2)
test3a1b2a2[,16] = predict(model3a1b2a2,test3a1b2a2)
test4a1b2a2[,16] = predict(model4a1b2a2,test4a1b2a2)
test5a1b2a2[,16] = predict(model5a1b2a2,test5a1b2a2)

train1a1b2a1a = train1a1b2a1 %>% filter(raw_timestamp_part_1 >= 0.028)
train1a1b2a1b = train1a1b2a1 %>% filter(raw_timestamp_part_1 < 0.028)
train2a1b2a1a = train2a1b2a1 %>% filter(raw_timestamp_part_1 >= 0.028)
train2a1b2a1b = train2a1b2a1 %>% filter(raw_timestamp_part_1 < 0.028)
train3a1b2a1a = train3a1b2a1 %>% filter(raw_timestamp_part_1 >= 0.028)
train3a1b2a1b = train3a1b2a1 %>% filter(raw_timestamp_part_1 < 0.028)
train4a1b2a1a = train4a1b2a1 %>% filter(raw_timestamp_part_1 >= 0.028)
train4a1b2a1b = train4a1b2a1 %>% filter(raw_timestamp_part_1 < 0.028)
train5a1b2a1a = train5a1b2a1 %>% filter(raw_timestamp_part_1 >= 0.028)
train5a1b2a1b = train5a1b2a1 %>% filter(raw_timestamp_part_1 < 0.028)

test1a1b2a1a = test1a1b2a1 %>% filter(raw_timestamp_part_1 >= 0.028)
test1a1b2a1b = test1a1b2a1 %>% filter(raw_timestamp_part_1 < 0.028)
test2a1b2a1a = test2a1b2a1 %>% filter(raw_timestamp_part_1 >= 0.028)
test2a1b2a1b = test2a1b2a1 %>% filter(raw_timestamp_part_1 < 0.028)
test3a1b2a1a = test3a1b2a1 %>% filter(raw_timestamp_part_1 >= 0.028)
test3a1b2a1b = test3a1b2a1 %>% filter(raw_timestamp_part_1 < 0.028)
test4a1b2a1a = test4a1b2a1 %>% filter(raw_timestamp_part_1 >= 0.028)
test4a1b2a1b = test4a1b2a1 %>% filter(raw_timestamp_part_1 < 0.028)
test5a1b2a1a = test5a1b2a1 %>% filter(raw_timestamp_part_1 >= 0.028)
test5a1b2a1b = test5a1b2a1 %>% filter(raw_timestamp_part_1 < 0.028)

model1a1b2a1a = train(classe ~., method="rpart", data=train1a1b2a1a)
model1a1b2a1b = train(classe ~., method="rpart", data=train1a1b2a1b)
model2a1b2a1a = train(classe ~., method="rpart", data=train2a1b2a1a)
model2a1b2a1b = train(classe ~., method="rpart", data=train2a1b2a1b)
model3a1b2a1a = train(classe ~., method="rpart", data=train3a1b2a1a)
model3a1b2a1b = train(classe ~., method="rpart", data=train3a1b2a1b)
model4a1b2a1a = train(classe ~., method="rpart", data=train4a1b2a1a)
model4a1b2a1b = train(classe ~., method="rpart", data=train4a1b2a1b)
model5a1b2a1a = train(classe ~., method="rpart", data=train5a1b2a1a)
model5a1b2a1b = train(classe ~., method="rpart", data=train5a1b2a1b)

test1a1b2a1a[,16] = predict(model1a1b2a1a,test1a1b2a1a)
test1a1b2a1b[,16] = predict(model1a1b2a1b,test1a1b2a1b)
test2a1b2a1a[,16] = predict(model2a1b2a1a,test2a1b2a1a)
test2a1b2a1b[,16] = predict(model2a1b2a1b,test2a1b2a1b)
test3a1b2a1a[,16] = predict(model3a1b2a1a,test3a1b2a1a)
test3a1b2a1b[,16] = predict(model3a1b2a1b,test3a1b2a1b)
test4a1b2a1a[,16] = predict(model4a1b2a1a,test4a1b2a1a)
test4a1b2a1b[,16] = predict(model4a1b2a1b,test4a1b2a1b)
test5a1b2a1a[,16] = predict(model5a1b2a1a,test5a1b2a1a)
test5a1b2a1b[,16] = predict(model5a1b2a1b,test5a1b2a1b)

predictions = rbind(test1b,test2b,test3b,test4b,test5b,
                    test1a2,test2a2,test3a2,test4a2,test5a2,
                    test1a1a,test2a1a,test3a1a,test4a1a,test5a1a,
                    test1a1b1a,test1a1b1b,test2a1b1a,test2a1b1b,test3a1b1a,test3a1b1b,test4a1b1a,test4a1b1b,test5a1b1a,test5a1b1b,
                    test1a1b2b1,test1a1b2b2,test2a1b2b1,test2a1b2b2,test3a1b2b1,test3a1b2b2,test4a1b2b1,test4a1b2b2,test5a1b2b1,test5a1b2b2,
                    test1a1b2a2,test2a1b2a2,test3a1b2a2,test4a1b2a2,test5a1b2a2,
                    test1a1b2a1a,test1a1b2a1b,test2a1b2a1a,test2a1b2a1b,test3a1b2a1a,test3a1b2a1b,test4a1b2a1a,test4a1b2a1b,test5a1b2a1a,test5a1b2a1b)


countp=0
for(i in 1:19622){
  if(predictions[i,1] == predictions[i,16]){
    countp = countp+1
  }
}
accurcy=countp/19622