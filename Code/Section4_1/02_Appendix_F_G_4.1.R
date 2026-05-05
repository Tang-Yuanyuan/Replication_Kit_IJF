# ==============================================================================
# Code for Reproducing Results in Appendix F & Appendix G.1-3 （Robustness Check）
# R Version: 4.5.2
# Note：This code should follows 01_Main_Section_4_1.R
# ==============================================================================

#----------------------------------------------------------------------------
# 1.  Alternative Specifications
#----------------------------------------------------------------------------

#（1）Trans

# （A）Replace caruse with carown 

model_carA1 <- polr(as.formula(paste("wta_car ~", base_vars, " + carown + heard_about_global_warming")), 
                data = df_car_sub, Hess = TRUE)
model_carA2 <- polr(as.formula(paste("wta_car ~", base_vars, " + carown + know_about_low_carbon")), 
                data = df_car_sub, Hess = TRUE)
model_carA3 <- polr(as.formula(paste("wta_car ~", base_vars, " + carown + know_about_carbon_neutrality")), 
                data = df_car_sub, Hess = TRUE)
model_carA4 <- polr(as.formula(paste("wta_car ~", base_vars, " + carown + know_about_carbon_policy")), 
                data = df_car_sub, Hess = TRUE)

#（B）Replace Caruse with MADT

model_carB1 <- polr(as.formula(paste("wta_car ~", base_vars, " + MADT + heard_about_global_warming")), 
                data = df_car_sub, Hess = TRUE)
model_carB2 <- polr(as.formula(paste("wta_car ~", base_vars, " + MADT + know_about_low_carbon")), 
                data = df_car_sub, Hess = TRUE)
model_carB3 <- polr(as.formula(paste("wta_car ~", base_vars, " + MADT + know_about_carbon_neutrality")), 
                data = df_car_sub, Hess = TRUE)
model_carB4 <- polr(as.formula(paste("wta_car ~", base_vars, " + MADT + know_about_carbon_policy")), 
                data = df_car_sub, Hess = TRUE)

#（C）Replace province with cityanswer

robust_vars <- "ifpollution + living_area_ln + age_ln + 
              is_bachelor + 
              location +   
              female + 
              married + 
              income_level + 
              youth + older_adults + 
              partymember +  
              cityanswer + weekday + ifsunny"

model_carC1 <- polr(as.formula(paste("wta_car ~", robust_vars, " + caruse + heard_about_global_warming")), 
                data = df_car_sub, Hess = TRUE)
model_carC2 <- polr(as.formula(paste("wta_car ~", robust_vars, " + caruse + know_about_low_carbon")), 
                data = df_car_sub, Hess = TRUE)
model_carC3 <- polr(as.formula(paste("wta_car ~", robust_vars, " + caruse + know_about_carbon_neutrality")), 
                data = df_car_sub, Hess = TRUE)
model_carC4 <- polr(as.formula(paste("wta_car ~", robust_vars, " + caruse + know_about_carbon_policy")), 
                data = df_car_sub, Hess = TRUE)

all_models <- list(model_carA1, model_carB1, model_carC1, 
                   model_carA2, model_carB2, model_carC2,
                   model_carA3, model_carB3, model_carC3, 
                   model_carA4, model_carB4, model_carC4)

aic_line <- c("AIC", sapply(all_models, function(x) round(AIC(x), 2)))
bic_line <- c("BIC", sapply(all_models, function(x) round(BIC(x), 2)))

stargazer(all_models, 
          type = "latex",
          out = paste0(output_path_table, "Table_F_7.tex"),
          column.labels = c("Global Warming", "Low Carbon", "Neutrality", "Policy"),
          column.separate = c(3, 3, 3, 3),
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          keep = c("^location", "^heard_about_global_warming", "^know_about_low_carbon", 
                   "^know_about_carbon_neutrality", "^know_about_carbon_policy"),
          omit = c("province", "weekday", "ifsunny", "cityanswer"),
          add.lines = list(aic_line, bic_line)) 

#（2）Home-energy
#  Replace conditioner1month with conditioner1time

model_elecA1 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1time + heard_about_global_warming")), 
                data = df_elec_sub, Hess = TRUE)
model_elecA2 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1time + know_about_low_carbon")), 
                data = df_elec_sub, Hess = TRUE)
model_elecA3 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1time + know_about_carbon_neutrality")), 
                data = df_elec_sub, Hess = TRUE)
model_elecA4 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1time + know_about_carbon_policy")), 
                data = df_elec_sub, Hess = TRUE)

stargazer(model_elecA1, model_elecA2, model_elecA3, model_elecA4, 
          type = "latex",
          out = paste0(output_path_table, "Table_F_8.tex"),
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          keep = c("female",
                   "is_bachelor",
                   "^know_about_low_carbon", 
                   "^know_about_carbon_neutrality", 
                   "^know_about_carbon_policy"),

          omit = c("province", "weekday", "weather", "cityanswer"),
          
          add.lines = get_ic_lines(model_elecA1, model_elecA2, model_elecA3, model_elecA4)
)

#（3）GreenElec
#  Replace mainelec with impgreenele

model_greenA1 <- polr(as.formula(paste("wta_green ~", base_vars, " + impgreenele + heard_about_global_warming")), 
                data = df_green_sub, Hess = TRUE)
model_greenA2 <- polr(as.formula(paste("wta_green ~", base_vars, " + impgreenele + know_about_low_carbon")), 
                data = df_green_sub, Hess = TRUE)
model_greenA3 <- polr(as.formula(paste("wta_green ~", base_vars, " + impgreenele + know_about_carbon_neutrality")), 
                data = df_green_sub, Hess = TRUE)
model_greenA4 <- polr(as.formula(paste("wta_green ~", base_vars, " + impgreenele + know_about_carbon_policy")), 
                data = df_green_sub, Hess = TRUE)

stargazer(model_greenA1, model_greenA2, model_greenA3, model_greenA4, 
          type = "latex",
          out = paste0(output_path_table, "Table_F_9.tex"),
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          keep = c("^location",
                   "income_level",
                   "^know_about_carbon_neutrality", 
                   "^know_about_carbon_policy"),

          omit = c("province", "weekday", "weather", "cityanswer"),
          
          add.lines = get_ic_lines(model_greenA1, model_greenA2, model_greenA3, model_greenA4)
)


#----------------------------------------------------------------------------
# 2.  Post-Statification
#----------------------------------------------------------------------------

df_clean <- df %>%
  mutate(
    edu_strat = case_when(
      education == "uneducated"  ~ "Uneducated",
      education == "primary"     ~ "Primary",
      education == "junior high" ~ "Junior high",
      education == "senior high" ~ "Senior high",
      education %in% c("associate", "bachelor", "postgraduate") ~ "College or higher",
      TRUE ~ NA_character_
    ),
    loc_strat = case_when(
      location %in% c("city", "county seat") ~ "Urban",
      location == "rural"                    ~ "Rural",
      TRUE ~ NA_character_
    ),
    gender_strat = factor(as.character(female), levels = c("0", "1")),
    loc_strat    = factor(loc_strat,    levels = c("Rural", "Urban")),
    edu_strat    = factor(edu_strat,    levels = c("Uneducated", "Primary", "Junior high", "Senior high", "College or higher"))
  ) %>%
  filter(!is.na(gender_strat), !is.na(loc_strat), !is.na(edu_strat), !is.na(age))

n_clean <- nrow(df_clean)

targets <- list(
  "(Intercept)" = n_clean,
  "gender_strat1" = n_clean * 0.488,
  "loc_stratUrban" = n_clean * 0.639,
  "edu_stratPrimary" = n_clean * 0.244,
  "edu_stratJunior high" = n_clean * 0.345,
  "edu_stratSenior high" = n_clean * 0.151,
  "edu_stratCollege or higher" = n_clean * 0.155,
  "age" = n_clean * 38.8
)


mm <- model.matrix(~ gender_strat + loc_strat + edu_strat + age, data = df_clean)
pop_totals <- numeric(ncol(mm))
names(pop_totals) <- colnames(mm)

for(n in names(pop_totals)) {
  if(n %in% names(targets)) {
    pop_totals[n] <- targets[[n]]
  } else {
    stop(paste("Variable name with an undefined target value:", n))
  }
}

ids <- svydesign(ids = ~1, data = df_clean)
cal_design_joint <- survey::calibrate(
  design     = ids,
  formula    = ~ gender_strat + loc_strat + edu_strat + age,
  population = pop_totals,
  calfun     = "raking",  
  epsilon    = 1e-7,
  maxit      = 2000
)


raw_w <- weights(cal_design_joint)
mean_w <- mean(raw_w)
lower_bound <- mean_w * 0.29
upper_bound <- mean_w * 9.5

cal_trimmed <- survey::trimWeights(
  cal_design_joint, 
  lower = lower_bound, 
  upper = upper_bound, 
  strict = TRUE
)

df_clean$weights <- weights(cal_trimmed)
df_clean$weights <- df_clean$weights / mean(df_clean$weights)

df$weights <- NA
df[rownames(df_clean), "weights"] <- df_clean$weights

check_design <- svydesign(ids = ~1, weights = ~weights, data = df_clean)

# --- Education ---
res_table_edu <- prop.table(svytable(~edu_strat, check_design))
target_edu <- c(
  "Uneducated"       = 0.027,
  "Primary"          = 0.244,
  "Junior high"      = 0.345,
  "Senior high"      = 0.151,
  "College or higher"= 0.155
)
diffs_edu     <- round(abs(res_table_edu[names(target_edu)] - target_edu), 3)
mean_diff_edu <- round(mean(diffs_edu, na.rm = TRUE), 3)

# --- Gender ---
res_gender  <- prop.table(svytable(~gender_strat, check_design))
diff_gender <- round(abs(res_gender["1"] - 0.488), 3)

# --- Urban ---
res_loc  <- prop.table(svytable(~loc_strat, check_design))
diff_loc <- round(abs(res_loc["Urban"] - 0.639), 3)

# --- Age ---
weighted_age_mean <- svymean(~age, check_design)
diff_age <- round(abs(coef(weighted_age_mean) - 38.8), 3)

# --- Print ---
target_age <- 38.8
edu_names <- c("Uneducated", "Primary", "Junior high", "Senior high", "College or higher")

report_weighted <- data.frame(
  Variable = c("Age (Mean)", "Gender (Female)", "Location", edu_names),
  Diff_Pct = c(
    (diff_age / target_age) * 100,  
    diff_gender * 100,              
    diff_loc * 100,
    diffs_edu * 100
  )
)

write.csv(report_weighted, 
          file = paste0(output_path_table, "Table_G_11.csv"), 
          row.names = FALSE)

# Balance Check Before Weighting)

raw_age_mean   <- mean(df_clean$age, na.rm = TRUE)
raw_gender_pct <- mean(as.numeric(as.character(df_clean$gender_strat)) == 1, na.rm = TRUE)
raw_urban_pct  <- mean(df_clean$loc_strat == "Urban", na.rm = TRUE)
raw_edu_table  <- prop.table(table(df_clean$edu_strat))

target_vals <- c(
  "Age (Mean)"         = 38.8,
  "Gender (Female)"    = 0.488,
  "Location (Urban)"   = 0.639,
  "Uneducated"         = 0.027, 
  "Primary"            = 0.244,
  "Junior high"        = 0.345,
  "Senior high"        = 0.151,
  "College or higher"  = 0.155
)

report_unweighted <- data.frame(
  Variable = c("Age (Mean)", "Gender (Female)", "Location (Urban)", 
               "Uneducated", "Primary", "Junior high", "Senior high", "College or higher"),
  
  Target   = c(38.8, 0.488, 0.639, 0.027, 0.244, 0.345, 0.151, 0.155),
  
  Sample   = c(
    raw_age_mean,
    raw_gender_pct,
    raw_urban_pct,
    as.numeric(raw_edu_table["Uneducated"]),
    as.numeric(raw_edu_table["Primary"]),
    as.numeric(raw_edu_table["Junior high"]),
    as.numeric(raw_edu_table["Senior high"]),
    as.numeric(raw_edu_table["College or higher"])
  )
)

report_unweighted$Bias <- report_unweighted$Sample - report_unweighted$Target

report_unweighted$Bias_Pct_Display <- ifelse(
  report_unweighted$Variable == "Age (Mean)",
  (report_unweighted$Bias / report_unweighted$Target) * 100, 
  report_unweighted$Bias * 100                        
)

report_pre_final <- report_unweighted %>%
  mutate(
    Target = round(Target, 3),
    Sample = round(Sample, 3),
    Bias_Display = round(Bias_Pct_Display, 2)
  ) %>%
  select(Variable, Target, Sample, `Bias_Percent(%)` = Bias_Display)

print("--- Pre-weighting Comparison Table ---")
print(report_pre_final)

write.csv(report_pre_final, 
          file = paste0(output_path_table, "Table_G_10.csv"), 
          row.names = FALSE)


# --- Weighted Regression ---
df$wta_car  <- factor(df$wta_car,  levels = 1:7, ordered = TRUE)
df$wta_elec <- factor(df$wta_elec, levels = 1:7, ordered = TRUE)
df$wta_green<- factor(df$wta_green,levels = 1:7, ordered = TRUE)

df$education  <- relevel(factor(df$education),  ref = "uneducated")
df$location   <- relevel(factor(df$location),   ref = "city")
df$marriage   <- relevel(factor(df$marriage),   ref = "unmarried")
df$youth      <- relevel(factor(df$youth),      ref = "0")
df$older_adults<-relevel(factor(df$older_adults),ref = "0")
df$province   <- as.factor(df$province)
df$weekday    <- as.factor(df$weekday)
df$partymember<- relevel(factor(df$partymember),ref = "0")
df$ifsunny    <- relevel(factor(df$ifsunny),    ref = "0")
df$mainuseelec<- relevel(factor(df$mainuseelec),ref = "0")
df$heard_about_global_warming    <- relevel(factor(df$heard_about_global_warming), ref = "no")
df$know_about_low_carbon         <- factor(df$know_about_low_carbon,
                   levels = c("never","heard but do not know","heard and know","familiar"))
df$know_about_carbon_neutrality  <- factor(df$know_about_carbon_neutrality,
                   levels = c("never","heard but do not know","heard and know","familiar"))
df$know_about_carbon_policy      <- factor(df$know_about_carbon_policy,
                   levels = c("never","heard but do not know","heard and know","familiar"))

df$ifpollution <- ifelse(df$aqi >= 101, 1, 0)
df$married     <- as.numeric(df$marriage == "married")
df$age_ln      <- log(df$age)
df$living_area_ln <- log(df$living_area)
df$is_bachelor <- ifelse(df$education %in% c("bachelor", "postgraduate"), 1, 0)
df$caruse <- ifelse(df$carusetime == 0, 1, 0)


base_vars <- "ifpollution + living_area_ln + age_ln + 
              is_bachelor + location + female + married + 
              income_level + youth + older_adults + partymember + 
              province + weekday + ifsunny"


df_car_sub   <- subset(df, publictrans < 5)
df_elec_sub  <- subset(df, conditionernumber == 1)
df_green_sub <- subset(df, energy_consume2020 > 1000)

svy_car   <- svydesign(ids = ~1, weights = ~weights, data = df_car_sub)
svy_elec  <- svydesign(ids = ~1, weights = ~weights, data = df_elec_sub)
svy_green <- svydesign(ids = ~1, weights = ~weights, data = df_green_sub)


print_svyolr <- function(model, title = "") {
  cat("\n============================\n", title, "\n============================\n")
  coef_mat <- coef(summary(model))
  print(round(coef_mat, 4))
}


model_car1_post <- svyolr(as.formula(paste("wta_car ~", base_vars, "+ caruse + heard_about_global_warming")),
                     design = svy_car)
model_car2_post <- svyolr(as.formula(paste("wta_car ~", base_vars, "+ caruse + know_about_low_carbon")),
                     design = svy_car)
model_car3_post <- svyolr(as.formula(paste("wta_car ~", base_vars, "+ caruse + know_about_carbon_neutrality")),
                     design = svy_car)
model_car4_post <- svyolr(as.formula(paste("wta_car ~", base_vars, "+ caruse + know_about_carbon_policy")),
                     design = svy_car)


model_elec1_post <- svyolr(as.formula(paste("wta_elec ~", base_vars, "+ conditioner1month + heard_about_global_warming")),
                      design = svy_elec)
model_elec2_post <- svyolr(as.formula(paste("wta_elec ~", base_vars, "+ conditioner1month + know_about_low_carbon")),
                      design = svy_elec)
model_elec3_post <- svyolr(as.formula(paste("wta_elec ~", base_vars, "+ conditioner1month + know_about_carbon_neutrality")),
                      design = svy_elec)
model_elec4_post <- svyolr(as.formula(paste("wta_elec ~", base_vars, "+ conditioner1month + know_about_carbon_policy")),
                      design = svy_elec)


model_green1_post <- svyolr(as.formula(paste("wta_green ~", base_vars, "+ mainuseelec + heard_about_global_warming")),
                       design = svy_green)
model_green2_post <- svyolr(as.formula(paste("wta_green ~", base_vars, "+ mainuseelec + know_about_low_carbon")),
                       design = svy_green)
model_green3_post <- svyolr(as.formula(paste("wta_green ~", base_vars, "+ mainuseelec + know_about_carbon_neutrality")),
                       design = svy_green)
model_green4_post <- svyolr(as.formula(paste("wta_green ~", base_vars, "+ mainuseelec + know_about_carbon_policy")),
                       design = svy_green)

# Prepaer for Section 4.2
keep_vars <- c(
  "id",   
  "wta_car", "wta_elec", "wta_green",
  "ifpollution", "living_area_ln", "age_ln", "is_bachelor",
  "location", "female", "married", "income_level",
  "youth", "older_adults", "partymember",
  "province", "weekday", "ifsunny",
  "caruse", "conditioner1month", "mainuseelec",
  "heard_about_global_warming", "know_about_low_carbon",
  "know_about_carbon_neutrality", "know_about_carbon_policy",
  "publictrans", "conditionernumber", "energy_consume2020",
  "weights"
)

df <- df %>% select(any_of(keep_vars))

write.csv(df, 
          file = paste0(output_path_data, "energy_wta_with_post_weights.csv"), 
          row.names = FALSE, 
          fileEncoding = "UTF-8")

# Print Regress Table

# Post-Stra Results
m_wt_car   <- list(model_car1_post, model_car2_post, model_car3_post, model_car4_post)
m_wt_elec  <- list(model_elec1_post, model_elec2_post, model_elec3_post, model_elec4_post)
m_wt_green <- list(model_green1_post, model_green2_post, model_green3_post, model_green4_post)

export_weighted_results <- function(wt_list, group_name, keep_pattern) {
  
  labels <- c("GW", "LC", "Neu", "Pol")

  tidy_model <- function(m) {
    s <- coef(summary(m))
    is_threshold <- grepl("\\|", rownames(s))
    s_coef <- s[!is_threshold, , drop = FALSE]
    
    c_idx <- if("Value" %in% colnames(s_coef)) "Value" else "Estimate"
    
    data.frame(
      term  = rownames(s_coef),
      coef  = as.numeric(s_coef[, c_idx]),
      se    = as.numeric(s_coef[, "Std. Error"]),
      tstat = as.numeric(s_coef[, ncol(s_coef)]), 
      stringsAsFactors = FALSE
    )
  }
  
  all_vars <- unique(unlist(lapply(wt_list, function(x) rownames(coef(summary(x))))))

  display_vars <- all_vars[grepl(keep_pattern, all_vars)]

  if(length(display_vars) == 0) {
    stop(paste("Error: No variables matched the keep_pattern for", group_name))
  }

  final_tab <- data.frame(Variable = display_vars, stringsAsFactors = FALSE)

  for (i in 1:4) {
    res <- tidy_model(wt_list[[i]])
    col_name <- labels[i]
    
    vals <- character(length(display_vars))
    for (j in seq_along(display_vars)) {
      match <- which(res$term == display_vars[j])
      if (length(match) > 0) {
        co <- res$coef[match]
        se <- res$se[match]
        ts <- res$tstat[match]
        stars <- ifelse(abs(ts) > 2.576, "***", 
                 ifelse(abs(ts) > 1.960, "**", 
                 ifelse(abs(ts) > 1.645, "*", "")))
        vals[j] <- sprintf("%.3f%s (%.3f)", co, stars, se)
      } else {
        vals[j] <- "" 
      }
    }
    final_tab[[col_name]] <- vals
  }

  file_name <- paste0(output_path_table, "Table_G_", group_name, ".tex")
  write.csv(final_tab, file = file_name, row.names = FALSE, fileEncoding = "UTF-8")
  
  cat("Success: Table for", group_name, "saved to", file_name, "\n")
  return(final_tab)
}

pattern_car   <- "^location|^heard_about_global_warming|^know_about_low_carbon|^know_about_carbon_neutrality|^know_about_carbon_policy"
pattern_elec  <- "female|is_bachelor|^know_about_low_carbon|^know_about_carbon_neutrality|^know_about_carbon_policy"
pattern_green <- "^location|income_level|^know_about_carbon_neutrality|^know_about_carbon_policy"

table_car   <- export_weighted_results(m_wt_car,   "12",   pattern_car)
table_elec  <- export_weighted_results(m_wt_elec,  "13",  pattern_elec)
table_green <- export_weighted_results(m_wt_green, "14", pattern_green)
