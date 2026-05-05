# ==============================================================================
# Code for Reproducing Results in Section 4.1/ Appendix B 
# R Version: 4.5.2
# 
# IMPORTANT: This R script is designed to work WITH the Python file
#            "03_Main_Section_4.2.py"
# 
# DO NOT run this script independently! It must be used in conjunction with
# the Python file "03_Main_Section_4.2.py" for full reproducibility.
# 
# Usage:
#   1. Run "03_Main_Section_4.2.py" first to generate necessary data
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
# 2. Processing Data
# ----------------------------------------------------------------------------

prepare_data <- function(data) {
  data$location <- as.factor(data$location)
  data$province <- as.factor(data$province)
  data$weekday  <- as.factor(data$weekday)
  data$heard_about_global_warming <- as.factor(data$heard_about_global_warming)
  data$know_about_low_carbon      <- as.factor(data$know_about_low_carbon)
  data$know_about_carbon_neutrality <- as.factor(data$know_about_carbon_neutrality)
  data$know_about_carbon_policy     <- as.factor(data$know_about_carbon_policy)
  
  data$location <- relevel(data$location, ref = "city")
  data$heard_about_global_warming <- relevel(data$heard_about_global_warming, ref = "no")
  
  levels_know <- c("never", "heard but do not know", "heard and know", "familiar")
  data$know_about_low_carbon <- factor(data$know_about_low_carbon, levels = levels_know)
  data$know_about_carbon_neutrality <- factor(data$know_about_carbon_neutrality, levels = levels_know)
  data$know_about_carbon_policy <- factor(data$know_about_carbon_policy, levels = levels_know)
  
  return(data)
}

train_df <- prepare_data(train_df)
test_df  <- prepare_data(test_df)

# ----------------------------------------------------------------------------
# 3. Train the Model
# ----------------------------------------------------------------------------

base_demos <- "ifpollution + living_area_ln + age_ln + 
              is_bachelor + 
              location +   
              female + 
              married + 
              income_level + 
              youth + older_adults + 
              partymember +  
              province + weekday + ifsunny"

base_all <- "ifpollution + living_area_ln + age_ln + 
              is_bachelor + 
              location +   
              female + 
              married + 
              income_level + 
              youth + older_adults + 
              partymember +  
              province + weekday + ifsunny + 
              heard_about_global_warming + know_about_low_carbon + know_about_carbon_neutrality + know_about_carbon_policy"

train_df_car <- subset(train_df, publictrans < 5 )
train_df_elec <- subset(train_df, conditionernumber == 1)
train_df_green <- subset(train_df, energy_consume2020 > 1000)

model_demo_car <- glm(as.formula(paste("y_car ~", base_demos, " + caruse")), 
                      data = train_df_car, 
                      family = binomial(link = "logit"))
model_all_car <- glm(as.formula(paste("y_car ~", base_all, " + caruse")), 
                      data = train_df_car, 
                      family = binomial(link = "logit"))

model_demo_elec <- glm(as.formula(paste("y_elec ~", base_demos, " + conditioner1month ")), 
                      data = train_df_elec, 
                      family = binomial(link = "logit"))
model_all_elec <- glm(as.formula(paste("y_elec ~", base_all, " + conditioner1month ")), 
                      data = train_df_elec, 
                      family = binomial(link = "logit"))

model_demo_green <- glm(as.formula(paste("y_green ~", base_demos, " + mainuseelec ")), 
                      data = train_df_green, 
                      family = binomial(link = "logit"))
model_all_green <- glm(as.formula(paste("y_green ~", base_all, " + mainuseelec ")), 
                      data = train_df_green, 
                      family = binomial(link = "logit"))

# ----------------------------------------------------------------------------
# 4. Predict and group into two separate tables (Demos Group & All Group).
# ----------------------------------------------------------------------------

# --- (A) Demos ---
logit_probs_demos <- data.frame(
  prob_car   = predict(model_demo_car,   newdata = test_df, type = "response"),
  prob_elec  = predict(model_demo_elec,  newdata = test_df, type = "response"),
  prob_green = predict(model_demo_green, newdata = test_df, type = "response")
)

# --- (B) All ---
logit_probs_all <- data.frame(
  prob_car   = predict(model_all_car,   newdata = test_df, type = "response"),
  prob_elec  = predict(model_all_elec,  newdata = test_df, type = "response"),
  prob_green = predict(model_all_green, newdata = test_df, type = "response")
)

# ----------------------------------------------------------------------------
# 5. Export to a specified directory
# ----------------------------------------------------------------------------

write.csv(logit_probs_demos, 
          "../../Data/Intermediate/logit_probs_demos.csv", 
          row.names = FALSE)

write.csv(logit_probs_all, 
          "../../Data/Intermediate/logit_probs_all.csv", 
          row.names = FALSE)
