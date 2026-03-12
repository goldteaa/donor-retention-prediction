# -----------------------------
# 0. Setup
# -----------------------------
library(tidyverse)
library(lubridate)
library(tidymodels)
library(pROC)
library(rpart)
library(ranger)
library(xgboost)
library(yardstick)
library(baguette)

set.seed(2025)

# -----------------------------
# 1. Load Data
# -----------------------------

donors          <- read_csv("donors.csv")
gifts           <- read_csv("gifts.csv")
basetable_2017  <- read_csv("basetable_2017.csv")
basetable_2018  <- read_csv("basetable_2018.csv")
feature_catalog <- read_csv("feature_catalog.csv")

# -----------------------------
# 2. Quick Exploration
# -----------------------------

glimpse(donors)
summary(donors)

gifts <- gifts %>%
  mutate(
    donor_id = as.integer(id),
    date = as.Date(date)
  )

glimpse(gifts)
summary(gifts)

# Date range of donations
range(gifts$date, na.rm = TRUE)
length(unique(gifts$donor_id))

glimpse(basetable_2017)
glimpse(basetable_2018)

summary(basetable_2017$target_donated)
summary(basetable_2018$target_donated)

colSums(is.na(basetable_2017))
colSums(is.na(basetable_2018))

head(feature_catalog)

# -----------------------------
# 3. Prepare Data for Modeling
# -----------------------------

# Fix avg_days_between NAs
for (df_name in c("basetable_2017", "basetable_2018")) {
  df <- get(df_name)
  if ("avg_days_between" %in% names(df)) {
    med_val <- median(df$avg_days_between, na.rm = TRUE)
    df$avg_days_between[is.na(df$avg_days_between)] <- med_val
  }
  assign(df_name, df)
}

# Ensure categorical variables are factors (if they exist)
cat_vars <- intersect(
  c("gender", "country", "region", "segment"),
  names(basetable_2017)
)

basetable_2017 <- basetable_2017 %>%
  mutate(across(all_of(cat_vars), as.factor))

basetable_2018 <- basetable_2018 %>%
  mutate(across(all_of(cat_vars), as.factor))

# -----------------------------
# 4. Logistic Regression (Base R)
# -----------------------------

# Split 2017 into train/validation
set.seed(2025)
n_2017   <- nrow(basetable_2017)
train_idx <- sample(seq_len(n_2017), size = floor(0.75 * n_2017))

train_2017 <- basetable_2017[train_idx, ]
valid_2017 <- basetable_2017[-train_idx, ]

# Full formula: all predictors except donor_id & target_amount
full_formula <- as.formula("target_donated ~ . - donor_id - target_amount")

logit_full <- glm(
  full_formula,
  data   = train_2017,
  family = binomial
)

summary(logit_full)

valid_probs_full <- predict(logit_full, newdata = valid_2017, type = "response")
roc_full <- roc(valid_2017$target_donated, valid_probs_full)
auc_full <- auc(roc_full)
cat("Full Logistic Model - Validation AUC:", round(auc_full, 3), "\n")

# ---- Forward Stepwise Selection (AUC-based) ----

forward_stepwise_auc <- function(
    candidate_vars,
    data_train,
    data_valid,
    target = "target_donated",
    max_vars = 10
) {
  selected_vars  <- character(0)
  remaining_vars <- candidate_vars
  best_auc       <- 0.5
  history <- tibble(
    step = integer(),
    variable_added = character(),
    auc = numeric()
  )
  
  for (step in seq_len(max_vars)) {
    step_results <- map_dfr(remaining_vars, function(v) {
      vars_try <- c(selected_vars, v)
      f <- as.formula(
        paste(target, "~", paste(vars_try, collapse = " + "))
      )
      fit <- glm(f, data = data_train, family = binomial)
      preds <- predict(fit, newdata = data_valid, type = "response")
      r <- tryCatch(roc(data_valid[[target]], preds), error = function(e) NA)
      auc_val <- if (any(is.na(r))) NA_real_ else as.numeric(auc(r))
      tibble(variable = v, auc = auc_val)
    })
    
    step_results <- step_results %>%
      filter(!is.na(auc))
    
    if (nrow(step_results) == 0) break
    
    # safest way: pick row with max AUC using base indexing
    best_idx   <- which.max(step_results$auc)
    best_step  <- step_results[best_idx, , drop = FALSE]
    
    if (best_step$auc <= best_auc + 1e-4) {
      # no meaningful improvement
      break
    }
    
    best_auc <- best_step$auc
    best_var <- best_step$variable
    selected_vars  <- c(selected_vars, best_var)
    remaining_vars <- setdiff(remaining_vars, best_var)
    
    history <- history %>%
      add_row(
        step = step,
        variable_added = best_var,
        auc = best_auc
      )
  }
  
  list(
    selected_vars = selected_vars,
    performance_history = history
  )
}

candidate_vars <- setdiff(
  names(train_2017),
  c("donor_id", "target_amount", "target_donated")
)

fs_result <- forward_stepwise_auc(
  candidate_vars = candidate_vars,
  data_train = train_2017,
  data_valid = valid_2017,
  target = "target_donated",
  max_vars = 10
)

fs_result$selected_vars
fs_result$performance_history

# Fit selected logistic model
if (length(fs_result$selected_vars) > 0) {
  selected_formula <- as.formula(
    paste("target_donated ~", paste(fs_result$selected_vars, collapse = " + "))
  )
  logit_selected <- glm(
    selected_formula,
    data   = train_2017,
    family = binomial
  )
  
  valid_probs_sel <- predict(logit_selected, newdata = valid_2017, type = "response")
  roc_sel <- roc(valid_2017$target_donated, valid_probs_sel)
  auc_sel <- auc(roc_sel)
  cat("Selected Logistic Model - Validation AUC:", round(auc_sel, 3), "\n")
} else {
  logit_selected <- NULL
  auc_sel <- NA_real_
}

# -----------------------------
# 5. Prepare for Tree-Based Models (tidymodels)
# -----------------------------

data_2017 <- basetable_2017 %>%
  mutate(
    target = factor(if_else(target_donated == 1, "yes", "no"))
  )

data_2018 <- basetable_2018 %>%
  mutate(
    target = factor(if_else(target_donated == 1, "yes", "no"))
  )

set.seed(2025)
data_split <- initial_split(data_2017, prop = 0.75, strata = target)
train_tree <- training(data_split)
valid_tree <- testing(data_split)
test_tree  <- data_2018

tree_rec <- recipe(target ~ ., data = train_tree) %>%
  update_role(donor_id, target_donated, target_amount, new_role = "ID") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

set.seed(2025)
cv_folds <- vfold_cv(train_tree, v = 5, strata = target)

# -----------------------------
# 6. Single Decision Tree
# -----------------------------

tree_spec <- decision_tree(
  mode = "classification",
  cost_complexity = 0.01,
  tree_depth      = 10,
  min_n           = 20
) %>%
  set_engine("rpart")

tree_wf <- workflow() %>%
  add_model(tree_spec) %>%
  add_recipe(tree_rec)

tree_fit <- tree_wf %>%
  fit(data = train_tree)

tree_pred_test <- predict(tree_fit, new_data = test_tree, type = "prob")

tree_auc_test <- auc(
  roc(
    as.numeric(test_tree$target == "yes"),
    tree_pred_test$.pred_yes
  )
)
cat("Single Decision Tree - 2018 Test AUC:", round(tree_auc_test, 3), "\n")

# -----------------------------
# 7. Tuned Decision Tree
# -----------------------------

tree_tune_spec <- decision_tree(
  mode = "classification",
  cost_complexity = tune(),
  tree_depth      = tune(),
  min_n           = tune()
) %>%
  set_engine("rpart")

tree_grid <- grid_regular(
  cost_complexity(),
  tree_depth(range = c(3L, 15L)),
  min_n(range = c(5L, 50L)),
  levels = 4
)

tree_tune_wf <- workflow() %>%
  add_model(tree_tune_spec) %>%
  add_recipe(tree_rec)

set.seed(2025)
tree_tune_res <- tune_grid(
  tree_tune_wf,
  resamples = cv_folds,
  grid      = tree_grid,
  metrics   = metric_set(roc_auc)
)

best_tree_params <- select_best(tree_tune_res, metric = "roc_auc")
best_tree_params

best_tree_wf <- finalize_workflow(tree_tune_wf, best_tree_params) %>%
  fit(data = train_tree)

best_tree_pred_test <- predict(best_tree_wf, new_data = test_tree, type = "prob")

best_tree_auc_test <- auc(
  roc(
    as.numeric(test_tree$target == "yes"),
    best_tree_pred_test$.pred_yes
  )
)
cat("Tuned Decision Tree - 2018 Test AUC:", round(best_tree_auc_test, 3), "\n")

# -----------------------------
# 8. Bagging (Bagged Trees)
# -----------------------------

bag_spec <- bag_tree(
  mode = "classification",
  cost_complexity = 0.01,
  tree_depth      = 15,
  min_n           = 10
) %>%
  set_engine("rpart", times = 50)

bag_wf <- workflow() %>%
  add_model(bag_spec) %>%
  add_recipe(tree_rec)

bag_fit <- bag_wf %>%
  fit(data = train_tree)

bag_pred_test <- predict(bag_fit, new_data = test_tree, type = "prob")

bag_auc_test <- auc(
  roc(
    as.numeric(test_tree$target == "yes"),
    bag_pred_test$.pred_yes
  )
)
cat("Bagged Trees - 2018 Test AUC:", round(bag_auc_test, 3), "\n")

# -----------------------------
# 9. Random Forest
# -----------------------------

rf_tune_spec <- rand_forest(
  mode  = "classification",
  mtry  = tune(),
  trees = 500,
  min_n = tune()
) %>%
  set_engine("ranger", importance = "impurity")

# Rough upper bound for mtry
p_train <- ncol(model.matrix(target ~ ., data = train_tree)) - 1

rf_grid <- grid_regular(
  mtry(range = c(5L, min(20L, p_train))),
  min_n(range = c(5L, 50L)),
  levels = 4
)

rf_wf <- workflow() %>%
  add_model(rf_tune_spec) %>%
  add_recipe(tree_rec)

set.seed(2025)
rf_tune_res <- tune_grid(
  rf_wf,
  resamples = cv_folds,
  grid      = rf_grid,
  metrics   = metric_set(roc_auc)
)

best_rf_params   <- select_best(rf_tune_res,   metric = "roc_auc")
best_rf_params

best_rf_wf <- finalize_workflow(rf_wf, best_rf_params) %>%
  fit(data = train_tree)

rf_pred_test <- predict(best_rf_wf, new_data = test_tree, type = "prob")

rf_auc_test <- auc(
  roc(
    as.numeric(test_tree$target == "yes"),
    rf_pred_test$.pred_yes
  )
)
cat("Random Forest - 2018 Test AUC:", round(rf_auc_test, 3), "\n")

# Variable importance
rf_fit_obj <- extract_fit_parsnip(best_rf_wf)$fit
rf_varimp <- as.data.frame(rf_fit_obj$variable.importance) %>%
  rownames_to_column("variable") %>%
  rename(importance = 2) %>%
  arrange(desc(importance))

rf_varimp_top15 <- head(rf_varimp, 15)

rf_varimp_top15 %>%
  ggplot(aes(x = reorder(variable, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 15 Variable Importance - Random Forest") +
  xlab("Variable") + ylab("Importance")

# -----------------------------
# 10. Gradient Boosting (XGBoost)
# -----------------------------

boost_tune_spec <- boost_tree(
  mode       = "classification",
  trees      = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n      = tune()
) %>%
  set_engine("xgboost")

boost_grid <- grid_latin_hypercube(
  trees(range = c(100L, 700L)),
  tree_depth(range = c(3L, 8L)),
  learn_rate(range = c(0.01, 0.3)),
  min_n(range = c(5L, 50L)),
  size = 20
)

boost_wf <- workflow() %>%
  add_model(boost_tune_spec) %>%
  add_recipe(tree_rec)

set.seed(2025)
boost_tune_res <- tune_grid(
  boost_wf,
  resamples = cv_folds,
  grid      = boost_grid,
  metrics   = metric_set(roc_auc)
)

best_boost_params <- select_best(boost_tune_res, metric = "roc_auc")
best_boost_params

best_boost_wf <- finalize_workflow(boost_wf, best_boost_params) %>%
  fit(data = train_tree)

boost_pred_test <- predict(best_boost_wf, new_data = test_tree, type = "prob")

boost_auc_test <- auc(
  roc(
    as.numeric(test_tree$target == "yes"),
    boost_pred_test$.pred_yes
  )
)
cat("Gradient Boosting (XGBoost) - 2018 Test AUC:", round(boost_auc_test, 3), "\n")

# -----------------------------
# 11. Refit Logistic Models on Full 2017 and Test on 2018
# -----------------------------

logit_full_all <- glm(
  full_formula,
  data   = basetable_2017,
  family = binomial
)

test_probs_full <- predict(
  logit_full_all,
  newdata = basetable_2018,
  type    = "response"
)

logit_full_auc_test <- auc(
  roc(
    basetable_2018$target_donated,
    test_probs_full
  )
)
cat("Full Logistic (2017->2018) - Test AUC:", round(logit_full_auc_test, 3), "\n")

if (!is.null(logit_selected) && length(fs_result$selected_vars) > 0) {
  selected_formula_all <- as.formula(
    paste("target_donated ~", paste(fs_result$selected_vars, collapse = " + "))
  )
  logit_sel_all <- glm(
    selected_formula_all,
    data   = basetable_2017,
    family = binomial
  )
  
  test_probs_sel <- predict(
    logit_sel_all,
    newdata = basetable_2018,
    type    = "response"
  )
  
  logit_sel_auc_test <- auc(
    roc(
      basetable_2018$target_donated,
      test_probs_sel
    )
  )
  cat("Selected Logistic (2017->2018) - Test AUC:", round(logit_sel_auc_test, 3), "\n")
} else {
  logit_sel_all      <- NULL
  test_probs_sel     <- rep(NA_real_, nrow(basetable_2018))
  logit_sel_auc_test <- NA_real_
}

# -----------------------------
# 12. Model Comparison Table (2018 Test Set)
# -----------------------------

compute_metrics <- function(truth_num, preds_prob, model_name) {
  truth_fac <- factor(if_else(truth_num == 1, "yes", "no"),
                      levels = c("no", "yes"))
  
  # Handle missing predictions
  if (all(is.na(preds_prob))) {
    return(tibble(
      model       = model_name,
      auc         = NA_real_,
      sensitivity = NA_real_,
      specificity = NA_real_
    ))
  }
  
  # AUC
  auc_val <- as.numeric(auc(roc(truth_num, preds_prob)))
  
  # Class predictions at 0.5 threshold
  class_pred <- factor(if_else(preds_prob >= 0.5, "yes", "no"),
                       levels = c("no", "yes"))
  
  # Build data frame for yardstick
  df_metrics <- tibble(
    truth    = truth_fac,
    estimate = class_pred
  )
  
  # Sensitivity & specificity
  sens_val <- sensitivity(df_metrics, truth, estimate,
                          event_level = "second")$.estimate
  spec_val <- specificity(df_metrics, truth, estimate,
                          event_level = "second")$.estimate
  
  tibble(
    model       = model_name,
    auc         = auc_val,
    sensitivity = sens_val,
    specificity = spec_val
  )
}


test_truth_num <- basetable_2018$target_donated

comparison_table <- bind_rows(
  compute_metrics(test_truth_num, test_probs_full,                "Logistic_Full"),
  compute_metrics(test_truth_num, test_probs_sel,                 "Logistic_Selected"),
  compute_metrics(test_truth_num, tree_pred_test$.pred_yes,       "Decision_Tree_Default"),
  compute_metrics(test_truth_num, best_tree_pred_test$.pred_yes,  "Decision_Tree_Tuned"),
  compute_metrics(test_truth_num, bag_pred_test$.pred_yes,        "Bagged_Trees"),
  compute_metrics(test_truth_num, rf_pred_test$.pred_yes,         "Random_Forest"),
  compute_metrics(test_truth_num, boost_pred_test$.pred_yes,      "Gradient_Boosting")
)

comparison_table

# -----------------------------
# 13. Business Impact & Lift Chart (Best Model)
# -----------------------------

# Clean out NA AUCs and pick best
comparison_clean <- comparison_table %>%
  filter(!is.na(auc))

if (nrow(comparison_clean) == 0) {
  stop("No valid models with non-NA AUC.")
}

best_model_row <- comparison_clean[order(-comparison_clean$auc), ][1, , drop = FALSE]
best_model_row
best_model_name <- best_model_row$model[1]
best_model_name

best_probs <- case_when(
  best_model_name == "Logistic_Full"         ~ test_probs_full,
  best_model_name == "Logistic_Selected"     ~ test_probs_sel,
  best_model_name == "Decision_Tree_Default" ~ tree_pred_test$.pred_yes,
  best_model_name == "Decision_Tree_Tuned"   ~ best_tree_pred_test$.pred_yes,
  best_model_name == "Bagged_Trees"          ~ bag_pred_test$.pred_yes,
  best_model_name == "Random_Forest"         ~ rf_pred_test$.pred_yes,
  best_model_name == "Gradient_Boosting"     ~ boost_pred_test$.pred_yes,
  TRUE                                       ~ test_probs_full
)

business_df <- tibble(
  donor_id = basetable_2018$donor_id,
  donated  = basetable_2018$target_donated,
  score    = best_probs
) %>%
  arrange(desc(score)) %>%
  mutate(
    rank   = row_number(),
    decile = ntile(rank, 10)
  )

# Business parameters
cost_per_mailing  <- 2.50
average_donation  <- 75

current_n_donors  <- nrow(business_df)
current_n_donated <- sum(business_df$donated == 1)

current_total_cost    <- current_n_donors * cost_per_mailing
current_total_revenue <- current_n_donated * average_donation
current_net_profit    <- current_total_revenue - current_total_cost

cat("Current strategy (mail everyone in 2018):\n")
cat("  Donors mailed:", current_n_donors, "\n")
cat("  Donations:", current_n_donated, "\n")
cat("  Cost:", current_total_cost, "\n")
cat("  Revenue:", current_total_revenue, "\n")
cat("  Net profit:", current_net_profit, "\n")

# Try different contact percentages
threshold_grid <- seq(0.1, 1.0, by = 0.1)

strategy_results <- map_dfr(threshold_grid, function(pct) {
  n_contact <- floor(pct * current_n_donors)
  chosen <- head(business_df, n_contact)
  cost <- n_contact * cost_per_mailing
  revenue <- sum(chosen$donated == 1) * average_donation
  net <- revenue - cost
  tibble(
    contact_pct = pct,
    n_contact   = n_contact,
    cost        = cost,
    revenue     = revenue,
    net_profit  = net
  )
})

strategy_results

strategy_results %>%
  ggplot(aes(x = contact_pct * 100, y = net_profit)) +
  geom_line() +
  geom_point() +
  ggtitle("Net Profit vs % of Donors Contacted (Best Model)") +
  xlab("% of donors contacted") +
  ylab("Net profit")

# Lift chart by decile
lift_df <- business_df %>%
  mutate(donated_flag = donated == 1) %>%
  group_by(decile) %>%
  summarise(
    n         = n(),
    positives = sum(donated_flag),
    .groups   = "drop"
  ) %>%
  mutate(
    cum_n            = cumsum(n),
    cum_positives    = cumsum(positives),
    baseline_rate    = sum(positives) / sum(n),
    baseline_cum_pos = baseline_rate * cum_n,
    lift             = cum_positives / baseline_cum_pos
  )

lift_df

lift_df %>%
  ggplot(aes(x = decile, y = cum_positives)) +
  geom_line() +
  geom_line(aes(y = baseline_cum_pos), linetype = "dashed") +
  ggtitle("Lift Chart - Cumulative Responders by Decile") +
  xlab("Decile (1 = highest scores)") +
  ylab("Cumulative responders")

lift_df %>%
  ggplot(aes(x = decile, y = lift)) +
  geom_col() +
  ggtitle("Lift by Decile - Best Model") +
  xlab("Decile (1 = highest scores)") +
  ylab("Lift")
