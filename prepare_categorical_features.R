#prepa 

# Load libraries
library(data.table)
library(lightgbm)


data(bank, package = "lightgbm")
str(bank)

# We must now transform the data to fit in LightGBM
# For this task, we use lgb.prepare
# The function transforms the data into a fittable data
bank <- lgb.prepare(data = bank)
str(bank)

# Remove 1 to label because it must be between 0 and 1
bank$y <- bank$y - 1

# Data input to LightGBM must be a matrix, without the label
my_data <- as.matrix(bank[, 1:16, with = FALSE])

# Creating the LightGBM dataset with categorical features
# The categorical features must be indexed like in R (1-indexed, not 0-indexed)
lgb_data <- lgb.Dataset(data = my_data,
                        label = bank$y,
                        categorical_feature = c(2, 3, 4, 5, 7, 8, 9, 11, 16))

# We can now train a model
model <- lgb.train(list(objective = "binary",
                        metric = "l2",
                        min_data = 1,
                        learning_rate = 0.1,
                        min_data = 0,
                        min_hessian = 1,
                        max_depth = 2),
                   lgb_data,
                   100,
                   valids = list(train = lgb_data))

# Try to find split_feature: 2
# If you find it, it means it used a categorical feature in the first tree
lgb.dump(model, num_iteration = 1) %>% 
  cat
