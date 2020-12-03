###################################################################### ########
#
# Extension 
# Original Paper: Comparing Random Forest with
# Logistic Regression for Predicting Class-Imbalanced Civil War Onset Data
# ###################################################################### ########


# data for prediction 
data=read.csv(file="SambanisImp.csv")
# data for looking at Variable Importance Plots 
data2<-read.csv(file="Amelia.Imp3.csv")
data_imp<-read.csv(file="data_full_fromonline.csv")

library(dplyr)
library(tidyr)
# Analyze if imputation even makes sense
missing_value_cnt <- data_imp %>% group_by(since_2000 = year >= 2000) %>% summarize_all(~sum(is.na(.x))) %>%
  pivot_longer(-since_2000, names_to = "variable", values_to = "missing_value_count")

missing_value_cnt$group <- ifelse(missing_value_cnt$since_2000, "Since 2000", "Before 2000")
missing_value_cnt$category <- ifelse(missing_value_cnt$missing_value_count > 0, "Has Missing Values", "No Missing Values")
missing_value_cnt %>% dplyr::select(-c(since_2000,variable)) %>%
  dplyr::group_by(group, category) %>%
  dplyr::summarize(variable_count = n()) %>%
  tidyr::pivot_wider(group, names_from = category, values_from = variable_count, values_fill = 0) %>%
  write.csv("table_variables_missingvalues_analysis.csv")

ggplot(missing_value_cnt , aes(factor(missing_value_count))) +
  geom_bar() +
  facet_wrap(~ group)+labs(x = "Number of Observations with Missing Values", y = "Number of Variables")  

ggplot(missing_value_cnt , aes(group, missing_value_count)) +
  geom_boxplot() +
  labs(y = "Number of Observations with Missing Values")+
  theme_bw()

ggplot(missing_value_cnt , aes(missing_value_count)) +
  geom_dotplot(dotsize = 0.3, method = "histodot") +
  facet_wrap(~group)+
  scale_y_continuous( breaks = NULL, name = "Number of Variables")+
  labs(x = "Number of Observations with Missing Values", y = "Number of Variables")  +
  theme_bw()
