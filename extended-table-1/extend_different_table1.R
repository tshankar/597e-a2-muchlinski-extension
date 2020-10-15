# This code attempts to compare the four models the prediction results 

library(tidyverse)

results <- read.csv("testing_file_CW_onset_RF_rep_final.csv")


# select 19 random country-year among the cases that don't have 
ind <- which(results$CW_Onset == 0)
set.seed(08540)
sample_ind <- sample(ind, 19)

table1_ext_1 <- results[sample_ind, ]


model_comp <- results %>% 
  dplyr::select(-X) %>%
  tidyr::pivot_longer(`Fearon.and.Latin..2003.` : `Random.Forest`, names_to = "model", values_to = "model_score") 

table1_ext_1_min <- table1_ext_1 %>% 
  dplyr::select(-X) %>%
  tidyr::pivot_longer(`Fearon.and.Latin..2003.` : `Random.Forest`, names_to = "model", values_to = "model_score") %>%
  group_by(COWcode, year, CW_Onset) %>%
  slice_min(model_score) %>%
  rename("best_model" = model )

table1_ext_1 <- table1_ext_1 %>%
  left_join(table1_ext_1_min) %>% 
  select(-X)

write.csv(table1_ext_1, "table1_extended_negative_examples.csv", 
          row.names = FALSE)

# what if we analyze the whole predict results
ggplot(model_comp %>% filter(CW_Onset == 0), aes(model_score)) +
  geom_histogram(bins = 40) +
  facet_wrap(~ model)+
  theme_bw()+
  ggtitle("Comparing model scores among all cases without civil war", 
          subtitle = "Each observation is a unique country-year case.") +
  ggsave("graph_table_ext2_nocivilwar.png", 
         width = 150, height = 120, 
         units = "mm")

ggplot(model_comp %>% filter(CW_Onset == 1), aes(model_score)) +
  geom_histogram(bins = 40) +
  facet_wrap(~ model)+
  theme_bw()+
  ggtitle("Comparing model scores among all cases with civil war", 
          subtitle = "Each observation is a unique country-year case.")+
  ggsave("graph_table_ext2_civilwar.png", 
         width = 150, height = 120, 
         units = "mm")

model_comp %>% group_by(model) %>%
  summarize(min_score = min(model_score), 
            max_score = max(model_score), 
            mean_score = mean(model_score), 
            range_score =  max(model_score) - min(model_score))

top20 <- model_comp %>% ungroup() %>% group_by(model) %>% slice_max(order_by = model_score, n= 20)
top20 %>% group_by(model) %>% summarise(have_civil_wars = sum(CW_Onset))


top30 <- model_comp %>% ungroup() %>% group_by(model) %>% slice_max(order_by = model_score, n= 30)
top30 %>% group_by(model) %>% summarise(sum(CW_Onset))

