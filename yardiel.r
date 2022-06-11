#install.packages("keras", dependencies = TRUE)
library(keras)

rawData <- read.csv("/Users/natorro/Desktop/programming/yardiel_projects/hawc-events/datasets/hawc_data/hawc_crudos200k.csv")

rawData<- subset(rawData, select = -c(evento))
indexes <- sample(1:200000, 40000)
randomSampleData <- rawData[indexes, ]
leftData <- rawData[-indexes, ]

training_events_df <- subset(randomSampleData, select = -c(class))
training_events_df <- as.matrix(training_events_df)
dimnames(training_events_df) <- NULL 

training_class_df <- subset(randomSampleData, select = c(class))
training_class_df <- as.matrix(training_class_df)
dimnames(training_class_df) <- NULL

test_events_df <- subset(leftData, select = -c(class))
test_events_df <- as.matrix(test_events_df)
dimnames(test_events_df) <- NULL

test_class_df <- subset(leftData, select = c(class))
test_class_df <- as.matrix(test_class_df)
dimnames(test_class_df) <- NULL


# DO I NEED TO RESHAPE HERE?
train_evs <- array_reshape(training_events_df, c(40000, 300))
train_class <- array_reshape(training_class_df, c(40000, 1))

rm("rawData", "indexes", "randomSampleData", "leftData")


network <- keras_model_sequential() %>%
      layer_dense(units = 300, activation = "sigmoid", input_shape = c(300)) %>% 
      layer_dense(units = 300, activation = "sigmoid") %>% 
      layer_dense(units = 300, activation = "sigmoid") %>% 
      layer_dense(units = 100, activation = "sigmoid") %>% 
      layer_dense(units = 1, activation = "sigmoid")

network %>% compile(
      optimizer = optimizer_rmsprop(learning_rate = 0.001),
      loss = "binary_crossentropy", 
      metrics = c("binary_accuracy")
)

network %>% fit(train_evs, train_class, epochs = 200)

save_model_hdf5(network, 
                filepath = "/Users/natorro/Desktop/programming/yardiel_projects/hawc-events/fivelayers2.hf5")
      