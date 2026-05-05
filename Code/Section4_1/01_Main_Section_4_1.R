# ==============================================================================
# Code for Reproducing Results in Section 4.1/ Appendix B 
# R Version: 4.5.2
# ==============================================================================

rm(list = ls())
cat("\014") 
graphics.off()
gc()

pkg_list <- c("MASS", "dplyr", "stargazer", "brant", "survey", 
              "VGAM", "ggplot2", "tidyr", "broom","knitr")

new_packages <- pkg_list[!(pkg_list %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

lapply(pkg_list, library, character.only = TRUE)

df <- read.csv("../../Data/energy_wta.csv")
output_path_table <- "../../Results/tables/"
output_path_picture <- "../../Results/pictures/"
output_path_data <- "../../Data/Intermediate/"

save_msg <- function(path, type = "file") {
  if(type == "table") cat("  -> Table:", basename(path), "\n")
  else if(type == "figure") cat("  -> Figure:", basename(path), "\n")
  else cat("  -> Saved:", basename(path), "\n")
}

#----------------------------------------------------------------------------
# 1. Data Processing
#----------------------------------------------------------------------------

#（1）

df$wta_car <- factor(df$wta_car, levels = 1:7, ordered = TRUE)
df$wta_elec <- factor(df$wta_elec, levels = 1:7, ordered = TRUE)
df$wta_green <- factor(df$wta_green, levels = 1:7, ordered = TRUE)

df$education<- as.factor(df$education)
df$location<- as.factor(df$location)
df$marriage<- as.factor(df$marriage)
df$youth<- as.factor(df$youth)
df$older_adults<- as.factor(df$older_adults)
df$province<- as.factor(df$province)
df$weekday<- as.factor(df$weekday)
df$partymember<- as.factor(df$partymember)
df$weather<- as.factor(df$weather)
df$mainuseelec<- as.factor(df$mainuseelec)
df$heard_about_global_warming <- as.factor(df$heard_about_global_warming )
df$know_about_low_carbon <- as.factor(df$know_about_low_carbon )
df$know_about_carbon_neutrality <- as.factor(df$know_about_carbon_neutrality )
df$know_about_carbon_policy <- as.factor(df$know_about_carbon_policy )

df$mainuseelec<- relevel(df$mainuseelec, ref = "0")
df$education<- relevel(df$education, ref = "uneducated")
df$location<- relevel(df$location, ref = "city")
df$marriage<- relevel(df$marriage, ref = "unmarried")
df$youth<- relevel(df$youth, ref = "0")
df$older_adults<- relevel(df$older_adults, ref = "0")
df$weather<- relevel(df$weather, ref = "0")
df$partymember<- relevel(df$partymember, ref = "0")
df$heard_about_global_warming <- relevel(df$heard_about_global_warming, ref = "no")
df$know_about_low_carbon <- factor(df$know_about_low_carbon, 
                 levels = c("never", "heard but do not know", "heard and know", "familiar"),
                 ordered = FALSE)
df$know_about_carbon_neutrality <- factor(df$know_about_carbon_neutrality, 
                 levels = c("never", "heard but do not know", "heard and know", "familiar"),
                 ordered = FALSE)
df$know_about_carbon_policy <- factor(df$know_about_carbon_policy, 
                 levels = c("never", "heard but do not know", "heard and know", "familiar"),
                 ordered = FALSE)

#（2）

df <- df %>%
  mutate(ifpollution = ifelse(aqi >= 101, 1, 0))
table(df$ifpollution)

df$married <- as.numeric(df$marriage == "married")
table(df$married )

df$age_ln <- log(df$age)
table(df$age_ln)

df$living_area_ln <- log(df$living_area)
table(df$living_area_ln)

df$is_bachelor <- ifelse(df$education %in% c("bachelor", "postgraduate"), 1, 0)
table(df$is_bachelor)

df$caruse <- ifelse(df$carusetime == 0, 1, 0)
table(df$caruse)

#(3) Table 1

# ================= 1. Upper Part of Table 1 =================
table1_var <- df %>% 
  select(all_of(c("income_level", "living_area", "age", "conditioner1month", 
                  "female", "youth", "older_adults", "ifpollution", 
                  "is_bachelor", "married", "partymember", "caruse", "mainuseelec")))

table1_var <- table1_var %>%
  mutate(across(everything(), ~as.numeric(as.character(.))))

datasummary(All(table1_var) ~ N + Mean + SD + Min + Max, 
            data = table1_var,
            fmt = 3,
            title = "Table 1 - Continuous",
            output = paste0(output_path_table, "Table1_upperpart.tex"))
save_msg(paste0(output_path_table, "Table1_upperpart.tex"), "table")

# ================= 2. Lower Half of Table 1 =================
cat_vars <- c("location", "province", "education", "weather", 
              "heard_about_global_warming", "know_about_carbon_neutrality", 
              "know_about_carbon_policy", "know_about_low_carbon", "weekday")

cat_data <- df %>% select(all_of(cat_vars))

datasummary_skim(cat_data, type = "categorical",
                 output = paste0(output_path_table, "Table1_lowerpart.tex"))
save_msg(paste0(output_path_table, "Table1_lowerpart.tex"), "table")

(4) Table 2

wta_trans <- prop.table(table(df$wta_car))
wta_homeenergy <- prop.table(table(df$wta_elec))
wta_greenelec <- prop.table(table(df$wta_green))

table2_data <- rbind(wta_trans, wta_homeenergy, wta_greenelec)

write.csv(table2_data, file = paste0(output_path_table, "Table_2.tex"))
save_msg(paste0(output_path_table, "Table_2.tex"), "table")

#----------------------------------------------------------------------------
# 2. Regression Model
#----------------------------------------------------------------------------

#（1）Basic Settings

base_vars <- "ifpollution + living_area_ln + age_ln + 
              is_bachelor + 
              location +   
              female + 
              married + 
              income_level + 
              youth + older_adults + 
              partymember +  
              province + weekday + weather"

df_car_sub <- subset(df, publictrans < 5 )# Unit of publictrans: Round
df_elec_sub <- subset(df, conditionernumber == 1)
df_green_sub <- subset(df, energy_consume2020 > 1000)

#（2）Trans

model_car1 <- polr(as.formula(paste("wta_car ~", base_vars, " + caruse + heard_about_global_warming")), 
                data = df_car_sub, Hess = TRUE)
model_car2 <- polr(as.formula(paste("wta_car ~", base_vars, " + caruse + know_about_low_carbon")), 
                data = df_car_sub, Hess = TRUE)
model_car3 <- polr(as.formula(paste("wta_car ~", base_vars, " + caruse + know_about_carbon_neutrality")), 
                data = df_car_sub, Hess = TRUE)
model_car4 <- polr(as.formula(paste("wta_car ~", base_vars, " + caruse + know_about_carbon_policy")), 
                data = df_car_sub, Hess = TRUE)

#（3）Home-energy

model_elec1 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1month + heard_about_global_warming")), 
                data = df_elec_sub, Hess = TRUE)
model_elec2 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1month + know_about_low_carbon")), 
                data = df_elec_sub, Hess = TRUE)
model_elec3 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1month + know_about_carbon_neutrality")), 
                data = df_elec_sub, Hess = TRUE)
model_elec4 <- polr(as.formula(paste("wta_elec ~", base_vars, " + conditioner1month + know_about_carbon_policy")), 
                data = df_elec_sub, Hess = TRUE)

#（4）GreenElec

model_green1 <- polr(as.formula(paste("wta_green ~", base_vars, " + mainuseelec + heard_about_global_warming")), 
                data = df_green_sub, Hess = TRUE)
model_green2 <- polr(as.formula(paste("wta_green ~", base_vars, " + mainuseelec + know_about_low_carbon")), 
                data = df_green_sub, Hess = TRUE)
model_green3 <- polr(as.formula(paste("wta_green ~", base_vars, " + mainuseelec + know_about_carbon_neutrality")), 
                data = df_green_sub, Hess = TRUE)
model_green4 <- polr(as.formula(paste("wta_green ~", base_vars, " + mainuseelec + know_about_carbon_policy")), 
                data = df_green_sub, Hess = TRUE)

#（5）Print Regression Results

get_ic_lines <- function(m1, m2, m3, m4) {
  list(
    c("AIC", round(AIC(m1), 2), round(AIC(m2), 2), round(AIC(m3), 2), round(AIC(m4), 2)),
    c("BIC", round(BIC(m1), 2), round(BIC(m2), 2), round(BIC(m3), 2), round(BIC(m4), 2))
  )
}

# ================= (6). Trans Models =================

stargazer(model_car1, model_car2, model_car3, model_car4, 
          type = "latex",
          out = paste0(output_path_table, "Table_3.tex"),
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          keep = c("^location", 
                   "^heard_about_global_warming", 
                   "^know_about_low_carbon", 
                   "^know_about_carbon_neutrality", 
                   "^know_about_carbon_policy"),

          omit = c("province", "weekday", "weather", "cityanswer"),
          
          add.lines = get_ic_lines(model_car1, model_car2, model_car3, model_car4)
)
save_msg(paste0(output_path_table, "Table_3.tex"), "table")

# ================= (7). Home-Energy Models =================

stargazer(model_elec1, model_elec2, model_elec3, model_elec4, 
          type = "latex",
          out = paste0(output_path_table, "Table_4.tex"),
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          keep = c("female",
                   "is_bachelor",
                   "^know_about_low_carbon", 
                   "^know_about_carbon_neutrality", 
                   "^know_about_carbon_policy"),

          omit = c("province", "weekday", "weather", "cityanswer"),
          
          add.lines = get_ic_lines(model_elec1, model_elec2, model_elec3, model_elec4)
)
save_msg(paste0(output_path_table, "Table_4.tex"), "table")

# ================= (8). GreenElec Models =================

stargazer(model_green1, model_green2, model_green3, model_green4, 
          type = "latex",
          out = paste0(output_path_table, "Table_5.tex"),
          model.numbers = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01),
          no.space = TRUE,
          keep = c("^location",
                   "income_level",
                   "^know_about_carbon_neutrality", 
                   "^know_about_carbon_policy"),

          omit = c("province", "weekday", "weather", "cityanswer"),
          
          add.lines = get_ic_lines(model_green1, model_green2, model_green3, model_green4)
)
save_msg(paste0(output_path_table, "Table_5.tex"), "table")

#----------------------------------------------------------------------------
# 4. Appendix B: Pictures
#----------------------------------------------------------------------------

if (!dir.exists(output_path_picture)) {
  dir.create(output_path_picture, recursive = TRUE)
}

plot_df_neutrality <- rbind(
  get_plot_data(model_car3, "know_about_carbon_neutrality", "WTA Transportation"),
  get_plot_data(model_elec3, "know_about_carbon_neutrality", "WTA Home Energy"),
  get_plot_data(model_green3, "know_about_carbon_neutrality", "WTA Green Electricity")
)

plot_df_policy <- rbind(
  get_plot_data(model_car4, "know_about_carbon_policy", "WTA Transportation"),
  get_plot_data(model_elec4, "know_about_carbon_policy", "WTA Home Energy"),
  get_plot_data(model_green4, "know_about_carbon_policy", "WTA Green Electricity")
)

Upic1 <- ggplot(plot_df_neutrality, aes(x = term, y = estimate, color = group, group = group, shape = group)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(position = position_dodge(0.2), linewidth = 1) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), 
                width = 0.1, linewidth = 0.8, 
                position = position_dodge(0.2)) +
  geom_point(size = 4, position = position_dodge(0.2)) +
  scale_color_manual(values = c("black", "gray40", "gray70")) + 
  labs(x = NULL, y = "Marginal Effect") + 
  theme_classic() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 15, hjust = 1)
  )

ggsave(filename = file.path(output_path_picture, "Figure_B.1.png"), 
       plot = Upic1, width = 8, height = 6, dpi = 300)
save_msg(file.path(output_path_picture, "Figure_B.1.png"), "figure")

Upic2 <- ggplot(plot_df_policy, aes(x = term, y = estimate, color = group, group = group, shape = group)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(position = position_dodge(0.2), linewidth = 1) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), 
                width = 0.1, linewidth = 0.8, 
                position = position_dodge(0.2)) +
  geom_point(size = 4, position = position_dodge(0.2)) +
  scale_color_manual(values = c("black", "gray40", "gray70")) + 
  labs(x = NULL, y = "Marginal Effect") + 
  theme_classic() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 15, hjust = 1)
  )

ggsave(filename = file.path(output_path_picture, "Figure_B.2.png"), 
       plot = Upic2, width = 8, height = 6, dpi = 300)
save_msg(file.path(output_path_picture, "Figure_B.2.png"), "figure")

# ==============================================================================
# 5. Appendix E: Brant Test
# ==============================================================================

model_car2_withoutfixed <- polr(wta_elec ~ ifpollution + living_area_ln + age_ln + 
                is_bachelor + 
                location +   
                female + 
                married + 
                income_level + 
                youth + older_adults + 
                partymember +  
                conditioner1month + know_about_low_carbon, 
                data = df_elec_sub, Hess = TRUE)

model_green3_withoutfixed <- polr(wta_green ~ifpollution + living_area_ln + age_ln + 
              is_bachelor + 
              location +   
              female + 
              married + 
              income_level + 
              youth + older_adults + 
              partymember + 
              mainuseelec + know_about_carbon_neutrality, 
                data = df_green_sub, Hess = TRUE)

model_green4_withoutfixed <- polr(wta_green ~ifpollution + living_area_ln + age_ln + 
              is_bachelor + 
              location +   
              female + 
              married + 
              income_level + 
              youth + older_adults + 
              partymember + 
              mainuseelec + know_about_carbon_policy, 
                data = df_green_sub, Hess = TRUE)


extract_brant_omnibus <- function(model, name) {
  res <- brant(model)
  chi2 <- res[1, 1]
  df   <- res[1, 2]
  pvar <- res[1, 3]
  
  cat(paste0("\nModel: ", name, 
             "\nOmnibus Chi2: ", round(chi2, 3), 
             "\nProbability: ", round(pvar, 3), "\n"))
  
  return(data.frame(Model = name, Chi2 = chi2, DF = df, P_value = pvar))
}

brant_summary <- list()

# (1) Trans
brant_summary[[1]] <- extract_brant_omnibus(model_car1, "Car1")
brant_summary[[2]] <- extract_brant_omnibus(model_car2_withoutfixed, "Car2")
brant_summary[[3]] <- extract_brant_omnibus(model_car3, "Car3")
brant_summary[[4]] <- extract_brant_omnibus(model_car4, "Car4")

# (2) Home-energy
brant_summary[[5]] <- extract_brant_omnibus(model_elec1, "Elec1")
brant_summary[[6]] <- extract_brant_omnibus(model_elec2, "Elec2")
brant_summary[[7]] <- extract_brant_omnibus(model_elec3, "Elec3")
brant_summary[[8]] <- extract_brant_omnibus(model_elec4, "Elec4")

# (3) GreenElec
brant_summary[[9]]  <- extract_brant_omnibus(model_green1, "Green1")
brant_summary[[10]] <- extract_brant_omnibus(model_green2, "Green2")
brant_summary[[11]] <- extract_brant_omnibus(model_green3_withoutfixed, "Green3")
brant_summary[[12]] <- extract_brant_omnibus(model_green4_withoutfixed, "Green4")

final_brant_table <- do.call(rbind, brant_summary)
final_brant_table$Model <- paste0("(", 1:nrow(final_brant_table), ")")
write.csv(final_brant_table, paste0(output_path_table, "Table_E_6.csv"), row.names = FALSE)

cat("\n========================================\n")
cat("01_Main_Section 4.1.R completed!\n")
cat("========================================\n")