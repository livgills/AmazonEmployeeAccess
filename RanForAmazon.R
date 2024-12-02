library(tidymodels)
library(vroom)
library(dplyr)
library(ranger)
library(embed)

amazon_train <- vroom("train.csv")
amazon_test <- vroom("test.csv")

amazon_train$ACTION = as.factor(amazon_train$ACTION)

my_recipe <- recipe(ACTION ~ ., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) 
 


for_mod <- rand_forest(mtry = 1,
                       min_n= 15,
                       trees= 1000) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(for_mod)



fin_wf <-
  forest_wf %>%
  fit(data=amazon_train)
amazon_pred <- predict(fin_wf, new_data=amazon_test, type="prob")

kaggle_submission <- amazon_pred %>%
  bind_cols(., amazon_test) %>%
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)


## Write out the file
vroom_write(x=kaggle_submission, file="./RanForletsgo!.csv", delim=",")
