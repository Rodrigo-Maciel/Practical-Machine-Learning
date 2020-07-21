################################################ LOADING PACKAGES ################################################
library(dplyr)
library(caret)
library(rattle)



################################################ LOADING THE DATA ################################################
# SET WORKING DIRECTORY
setwd("~/Coursera/Practical Machine Learning (ML en R)/Proyecto Final")

train = read.csv(file = "pml-training.csv")
test = read.csv(file = "pml-testing.csv")



################################################ PREPARING THE DATA ################################################
# Drop user_name, new_window cvtd_timestamp, to predict "CLASSE" in base to continuos and measurements data (not in base of user or time of day)
train = subset(train, select = -c(user_name, cvtd_timestamp, new_window))
test = subset(test, select = -c(user_name, cvtd_timestamp, new_window))

# Drop variables with +5% of missing data, filling this variables would change the model's outcome
train = subset(train, select = -c(X, max_roll_belt, max_picth_belt, min_roll_belt, min_pitch_belt, amplitude_roll_belt, amplitude_pitch_belt,
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

test = subset(test, select = -c(X, max_roll_belt, max_picth_belt, min_roll_belt, min_pitch_belt, amplitude_roll_belt, amplitude_pitch_belt,
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
# KEEP 90%, 72%-14%, 97%, 95%, 92%, 93%, 96%
train[,57] = train[,4]*0.42079160 + train[,6]*0.59154028 + train[,7]*0.05024761 + train[,12]*0.16805308 + train[,13]*-0.66501720
test[,57] = test[,4]*0.42079160 + test[,6]*0.59154028 + test[,7]*0.05024761 + test[,12]*0.16805308 + test[,13]*-0.66501720
train[,58] = train[,5]*-0.2851349 + train[,11]*0.3816465 + train[,14]*0.8792292
train[,59] = train[,5]*-0.5242966 + train[,11]*0.7058034 + train[,14]*-0.4763975
test[,58] = test[,5]*-0.2851349 + test[,11]*0.3816465 + test[,14]*0.8792292
test[,59] = test[,5]*-0.5242966 + test[,11]*0.7058034 + test[,14]*-0.4763975
train[,60] = train[,21]*0.9278154 + train[,22]*-0.3730396
test[,60] = test[,21]*0.9278154 + test[,22]*-0.3730396
train[,61] = train[,24]*0.3320298 + train[,27]*0.9432689
test[,61] = test[,24]*0.3320298 + test[,27]*0.9432689
train[,62] = train[,28]*-0.488373 + train[,29]*-0.872635
test[,62] = test[,28]*-0.488373 + test[,29]*-0.872635
train[,63] = train[,32]*0.5832650 + train[,39]*0.8122819
test[,63] = test[,32]*0.5832650 + test[,39]*0.8122819
train[,64] = train[,34]*0.4633751 + train[,36]*-0.7105826 + train[,49]*-0.5294864
test[,64] = test[,34]*0.4633751 + test[,36]*-0.7105826 + test[,49]*-0.5294864

# Drop variables that were transformed via PCA
train = subset(train, select = -c(roll_belt, yaw_belt, total_accel_belt, accel_belt_y, accel_belt_z, pitch_belt, accel_belt_x, magnet_belt_x,
                                        gyros_arm_x, gyros_arm_y, accel_arm_x, magnet_arm_x, magnet_arm_y, magnet_arm_z, yaw_dumbbell, accel_dumbbell_z,
                                        gyros_dumbbell_x, gyros_dumbbell_z, gyros_forearm_z))

test = subset(test, select = -c(roll_belt, yaw_belt, total_accel_belt, accel_belt_y, accel_belt_z, pitch_belt, accel_belt_x, magnet_belt_x,
                                        gyros_arm_x, gyros_arm_y, accel_arm_x, magnet_arm_x, magnet_arm_y, magnet_arm_z, yaw_dumbbell, accel_dumbbell_z,
                                        gyros_dumbbell_x, gyros_dumbbell_z, gyros_forearm_z))

# REORDER DATABASE
train = train[,c(1,2,3,4,5,6,7,8,9,10,
                       11,12,13,14,15,16,17,18,19,20,
                       21,22,23,24,25,26,27,28,29,30,
                       31,32,33,34,35,36,38,39,40,
                       41,42,43,44,45,
                       37)]

test = test[,c(1,2,3,4,5,6,7,8,9,10,
                       11,12,13,14,15,16,17,18,19,20,
                       21,22,23,24,25,26,27,28,29,30,
                       31,32,33,34,35,36,38,39,40,
                       41,42,43,44,45,
                       37)]

# DATA ESTANDARIZATION
estandarized_matrix = matrix(data = NA, nrow = 44, ncol = 2)
colnames(estandarized_matrix) = c("MEAN", "SD")

for(i in 1:44){
  estandarized_matrix[i,1] = mean(train[,i])
  estandarized_matrix[i,2] = sd(train[,i])
}

etrain = train[,45]
etrain = as.data.frame(etrain)
etest = test[,45]
etest = as.data.frame(etest)

for(i in 1:44){
  etrain[,i+1] = (train[,i]-estandarized_matrix[i,1])/estandarized_matrix[i,2]
  etest[,i+1] = (test[,i]-estandarized_matrix[i,1])/estandarized_matrix[i,2]
}

colnames(etrain) = c("classe", "raw_timestamp_part_1", "raw_timestamp_part_2", "num_window", "gyros_belt_x", 
                                    "gyros_belt_y", "gyros_belt_z", "magnet_belt_y", "magnet_belt_z", "roll_arm", "pitch_arm", "yaw_arm",
                                    "total_accel_arm", "gyros_arm_z", "accel_arm_y", "accel_arm_z", "roll_dumbbell", "pitch_dumbbell",
                                    "total_accel_dumbbell", "gyros_dumbbell_y", "accel_dumbbell_x", "accel_dumbbell_y", "magnet_dumbbell_x",
                                    "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", "yaw_forearm",
                                    "total_accel_forearm", "gyros_forearm_x", "gyros_forearm_y", "accel_forearm_x",
                                    "accel_forearm_y", "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z",
                                    "V57", "V58", "V59", "V60", "V61", "V62", "V63", "V64")

colnames(etest) = c("classe", "raw_timestamp_part_1", "raw_timestamp_part_2", "num_window", "gyros_belt_x", 
                                    "gyros_belt_y", "gyros_belt_z", "magnet_belt_y", "magnet_belt_z", "roll_arm", "pitch_arm", "yaw_arm",
                                    "total_accel_arm", "gyros_arm_z", "accel_arm_y", "accel_arm_z", "roll_dumbbell", "pitch_dumbbell",
                                    "total_accel_dumbbell", "gyros_dumbbell_y", "accel_dumbbell_x", "accel_dumbbell_y", "magnet_dumbbell_x",
                                    "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", "yaw_forearm",
                                    "total_accel_forearm", "gyros_forearm_x", "gyros_forearm_y", "accel_forearm_x",
                                    "accel_forearm_y", "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z",
                                    "V57", "V58", "V59", "V60", "V61", "V62", "V63", "V64")

# DATA SLICING (5 FOLD CROSS VALDATION)
# SELECT VARIABLES TU USE
# WE TRAIN 5 DECISION TREES, EACH WITH A DIFFERENT TRAINING SET AND WID THIS WE DECIDE WICH VARIABLES CAN HEL US PREDICT CLASSE
# THIS WILL MAKE A SIMPLER MODEL, EASIER TO INTERPRET, FASTER TO TRAIN, AND LESS SUSCEPTIBLE TO OVERFITTING
# WE KEEP ANY VARIABLE THAT APPEARS IMPORTANT IN ALL  MODELS

# DATA SLICING (RELEVANT VARIABLES)
etrain = etrain[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]
etest = etest[,c(1, 2, 4, 7, 8, 9, 12, 17, 22, 23, 24, 25, 26, 27, 42)]

# BALANCED TRAIN DATASET (A:29%, B:20%, C:17%, D:17%, E:18% OF THE TRAINING DATASET) WE WILL USE ACCURACY AS METRIC

# A 5 FOLD CROSS VALIDATION WITH DECISION TREE CLASSIFIER WAS MADE IN THE TRAIN DATASET WITH ALL PRIOR STEPS
# MEAN ACCURACY = 0.85
# SD ACCURACY = 0.015
# RANDOM FOREST ISN'T NECESSARY DUE TO THE SMALL VARIANCE (RESULT WOULDN'T CHANGE MUCH)
# DECISON TREE WAS THE ALGORITHM USED DUE TO IT'S SPEED, PREDICTION CAPACITY, AND RULES INTERPRETATION

# IN THIS PART WE WILL TRAIN THE SAME MODEL WITH THE ENTIRE TRAIN DATASET AND TEST IT WITH THE TEST DATASET
# WE ALREADY HAVE A VERY GOOD IDEA OF HOW THE MODEL WILL PREFORM, BUT WITH THIS WE WILL VERIFY IT

# MODEL
train1= etrain %>% filter(pitch_forearm < -1.6)
train2= etrain %>% filter(pitch_forearm >= -1.6)
test1= etest %>% filter(pitch_forearm < -1.6)
test2= etest %>% filter(pitch_forearm >= -1.6)

test1[,16]="A"

train3= train2 %>% filter(magnet_belt_y >= -1.1)
train4= train2 %>% filter(magnet_belt_y < -1.1)
test3= test2 %>% filter(magnet_belt_y >= -1.1)
test4= test2 %>% filter(magnet_belt_y < -1.1)

model4 = train(classe ~., method="rpart", data=train4)
test4[,16] = predict(model4,test4)

train5= train3 %>% filter(magnet_dumbbell_y >= 0.66)
train6= train3 %>% filter(magnet_dumbbell_y < 0.66)
test5= test3 %>% filter(magnet_dumbbell_y >= 0.66)
test6= test3 %>% filter(magnet_dumbbell_y < 0.66)

model5 = train(classe ~., method="rpart", data=train5)
test5[,16] = predict(model5,test5)

train7 = train6 %>% filter(roll_forearm >= 0.82)
train8 = train6 %>% filter(roll_forearm < 0.82)
test7 = test6 %>% filter(roll_forearm >= 0.82)
test8 = test6 %>% filter(roll_forearm < 0.82)

train9 = train7 %>% filter(raw_timestamp_part_1 >= -1.6)
train10 = train7 %>% filter(raw_timestamp_part_1 < -1.6)
test9 = test7 %>% filter(raw_timestamp_part_1 >= -1.6)
test10 = test7 %>% filter(raw_timestamp_part_1 < -1.6)

model9 = train(classe ~., method="rpart", data=train9)
model10 = train(classe ~., method="rpart", data=train10)
test9[,16] = predict(model9,test9)
test10[,16] = predict(model10,test10)

train11 = train8 %>% filter(magnet_dumbbell_z >= -0.5)
train12 = train8 %>% filter(magnet_dumbbell_z < -0.5)
test11 = test8 %>% filter(magnet_dumbbell_z >= -0.5)
test12 = test8 %>% filter(magnet_dumbbell_z < -0.5)

train13 = train12 %>% filter(raw_timestamp_part_1 >= 1.3)
train14 = train12 %>% filter(raw_timestamp_part_1 < 1.3)
test13 = test12 %>% filter(raw_timestamp_part_1 >= 1.3)
test14 = test12 %>% filter(raw_timestamp_part_1 < 1.3)

model13 = train(classe ~., method="rpart", data=train13)
model14 = train(classe ~., method="rpart", data=train14)
test13[,16] = predict(model13,test13)
test14[,16] = predict(model14,test14)

train15 = train11 %>% filter(num_window >= -0.76)
train16 = train11 %>% filter(num_window < -0.76)
test15 = test11 %>% filter(num_window >= -0.76)
test16 = test11 %>% filter(num_window < -0.76)

model16 = train(classe ~., method="rpart", data=train16)
test16[,16] = predict(model16,test16)

train17 = train15 %>% filter(raw_timestamp_part_1 >= 0.028)
train18 = train15 %>% filter(raw_timestamp_part_1 < 0.028)
test17 = test15 %>% filter(raw_timestamp_part_1 >= 0.028)
test18 = test15 %>% filter(raw_timestamp_part_1 < 0.028)

model17 = train(classe ~., method="rpart", data=train17)
model18 = train(classe ~., method="rpart", data=train18)
test17[,16] = predict(model17,test17)
test18[,16] = predict(model18,test18)

predictions = rbind(test1,test4,test5,test9,test10,test13,test14,test16,test17,test18)