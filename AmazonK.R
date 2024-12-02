library(tidymodels)
library(vroom)
library(embed) 
library(lme4)
library(kknn)
amazon_train <- vroom("train.csv")
amazon_test <- vroom("test.csv")

amazon_train$ACTION = as.factor(amazon_train$ACTION)

my_recipe <- recipe(ACTION ~ ., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_lencode_bayes(all_nominal_predictors(), outcome = vars(ACTION))


dough <- prep(my_recipe)
baked <- bake(dough, new_data = amazon_train)

knn_model <- nearest_neighbor(neighbors=50) %>% 
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

## Fit or Tune Model HERE
final_wf <-
  knn_wf %>% 
  fit(data = amazon_train)
amazon_pred <- predict(final_wf, new_data=amazon_test, type="prob")

kaggle_submission <- amazon_pred %>%
  bind_cols(., amazon_test) %>%
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)


## Write out the file
vroom_write(x=kaggle_submission, file="./Kthn.csv", delim=",")


