library(tidymodels)
library(vroom)
library(embed) 
library(themis)
library(ranger)

amazon_train <- vroom("train.csv")
amazon_test <- vroom("test.csv")

amazon_train$ACTION = as.factor(amazon_train$ACTION)

my_recipe <- recipe(ACTION ~ ., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>%
  step_smote(all_outcomes(), neighbors=10) %>% 
  step_upsample(all_outcomes())

prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = amazon_train)

for_mod <- rand_forest(mtry = tune(),
                       min_n=tune(),
                       trees= 600) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(for_mod)

forestgrid <- grid_regular(mtry(range = c(1,10)),
                           min_n(),
                           levels = 5)
forest_folds <- vfold_cv(amazon_train, v = 5, repeats=1)

CV_results <- forest_wf %>%
  tune_grid(resamples=forest_folds,
            grid=forestgrid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

fin_wf <-
  forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)
amazon_pred <- predict(fin_wf, new_data=amazon_test, type="prob")

kaggle_submission <- amazon_pred %>%
  bind_cols(., amazon_test) %>%
  rename(ACTION = .pred_1) %>% 
  select(id, ACTION)


## Write out the file
vroom_write(x=kaggle_submission, file="./RanFor1.csv", delim=",")
