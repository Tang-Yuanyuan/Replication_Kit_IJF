# ==============================================================================
# Code for Reproducing Results in Section 4.1/ Appendix B 
# R Version: 4.5.2
# 
# IMPORTANT: This R script is designed to work WITH the Python file
#            "05_Appendix_G_4_2.py"
# 
# DO NOT run this script independently! It must be used in conjunction with
# the Python file "05_Appendix_G_4_2.py" for full reproducibility.
# 
# Usage:
#   1. Run "05_Appendix_G_4_2.py" first to generate necessary data
#   2. It will automatically run this R script to produce empirical results
# ==============================================================================

rm(list = ls())
cat("\014") 
graphics.off()
gc()

pkg_list <- c("MASS", "dplyr")

new_packages <- pkg_list[!(pkg_list %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

lapply(pkg_list, library, character.only = TRUE)

# ----------------------------------------------------------------------------
# 1. Read data passed from Python.
# ----------------------------------------------------------------------------

train_df <- read.csv("../../Data/Intermediate/train_data.csv")
test_df  <- read.csv("../../Data/Intermediate/test_data.csv")

# ----------------------------------------------------------------------------
# 2. 数据处理函数(不对复杂变量设置ref，以便模型正常运行)
# ----------------------------------------------------------------------------

prepare_data <- function(data) {

  wta_levels <- c("1", "2", "3", "4", "5", "6", "7")
  data$wta_car   <- factor(data$wta_car, levels = wta_levels, ordered = TRUE)
  data$wta_elec  <- factor(data$wta_elec, levels = wta_levels, ordered = TRUE)
  data$wta_green <- factor(data$wta_green, levels = wta_levels, ordered = TRUE)

  data$location <- as.factor(data$location)
  data$province <- as.factor(data$province)
  data$weekday  <- as.factor(data$weekday)
  data$heard_about_global_warming <- as.factor(data$heard_about_global_warming)
  data$know_about_low_carbon      <- as.factor(data$know_about_low_carbon)
  data$know_about_carbon_neutrality <- as.factor(data$know_about_carbon_neutrality)
  data$know_about_carbon_policy     <- as.factor(data$know_about_carbon_policy)
  
  data$location <- relevel(data$location, ref = "city")
  data$heard_about_global_warming <- factor(data$heard_about_global_warming, 
                                     levels = c("no", "yes and agree", "yes but disagree"))  
  levels_know <- c("never", "heard but do not know", "heard and know", "familiar")
  data$know_about_low_carbon <- factor(data$know_about_low_carbon, levels = levels_know)
  data$know_about_carbon_neutrality <- factor(data$know_about_carbon_neutrality, levels = levels_know)
  data$know_about_carbon_policy <- factor(data$know_about_carbon_policy, levels = levels_know)
  
  return(data)
}

train_df <- prepare_data(train_df)
test_df  <- prepare_data(test_df)

# ----------------------------------------------------------------------------
# 3. Training an ologit Model (Ordered Logit)
# ----------------------------------------------------------------------------
base_demos <- "ifpollution + living_area_ln + age_ln + is_bachelor + location + female + married + income_level + youth + older_adults + partymember + province + weekday + ifsunny"
base_all   <- paste(base_demos, "+ heard_about_global_warming + know_about_low_carbon + know_about_carbon_neutrality + know_about_carbon_policy")

train_df_car   <- subset(train_df, publictrans < 5)
train_df_elec  <- subset(train_df, conditionernumber == 1)
train_df_green <- subset(train_df, energy_consume2020 > 1000)

model_demo_car   <- polr(as.formula(paste("wta_car ~", base_demos, " + caruse")), data = train_df_car, Hess = TRUE)
model_demo_elec  <- polr(as.formula(paste("wta_elec ~", base_demos, " + conditioner1month")), data = train_df_elec, Hess = TRUE)
model_demo_green <- polr(as.formula(paste("wta_green ~", base_demos, " + mainuseelec")), data = train_df_green, Hess = TRUE)

model_all_car    <- polr(as.formula(paste("wta_car ~", base_all, " + caruse")), data = train_df_car, Hess = TRUE)
model_all_elec   <- polr(as.formula(paste("wta_elec ~", base_all, " + conditioner1month")), data = train_df_elec, Hess = TRUE)
model_all_green  <- polr(as.formula(paste("wta_green ~", base_all, " + mainuseelec")), data = train_df_green, Hess = TRUE)

# ----------------------------------------------------------------------------
# 4. 执行预测并生成数据表
# ----------------------------------------------------------------------------

calc_expected_wta_safe <- function(model, train_data, test_data) {

  resp_var <- as.character(formula(model)[[2]])
  all_vars <- all.vars(formula(model))
  predictor_vars <- setdiff(all_vars, resp_var)
  

  for (v in predictor_vars) {
    if (is.factor(train_data[[v]])) {
      test_data[[v]] <- factor(test_data[[v]], levels = levels(train_data[[v]]))
    }
  }
  

  res <- tryCatch({
    probs <- predict(model, newdata = test_data, type = "probs", na.action = na.pass)
    levels_num <- as.numeric(colnames(probs))
    expected_score <- probs %*% levels_num
    as.vector(expected_score)
  }, error = function(e) {
    cat("\n⚠️ Prediction failed; returning NA. Error message: ", e$message)
    return(rep(NA, nrow(test_data)))
  })
  
  return(res)
}

# --- (A) Demos ---
logit_preds_demos <- data.frame(
  pred_wta_car   = calc_expected_wta_safe(model_demo_car,   train_df_car,   test_df),
  pred_wta_elec  = calc_expected_wta_safe(model_demo_elec,  train_df_elec,  test_df),
  pred_wta_green = calc_expected_wta_safe(model_demo_green, train_df_green, test_df)
)

# --- (B) All ---
logit_preds_all <- data.frame(
  pred_wta_car   = calc_expected_wta_safe(model_all_car,   train_df_car,   test_df),
  pred_wta_elec  = calc_expected_wta_safe(model_all_elec,  train_df_elec,  test_df),
  pred_wta_green = calc_expected_wta_safe(model_all_green, train_df_green, test_df)
)

# ----------------------------------------------------------------------------
# 5. Export to a specified directory
# ----------------------------------------------------------------------------

write.csv(logit_probs_demos, 
          "../../Data/Intermediate/wta_preds_demos.csv", 
          row.names = FALSE)

write.csv(logit_probs_all, 
          "../../Data/Intermediate/wta_preds_all.csv", 
          row.names = FALSE)