library(titanic)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(cowplot)

titanic_train

summary(is.na(titanic_train))

ggplot(titanic_train, aes(Age)) + 
  geom_histogram(color = "#000000", fill = "#0099F8") + 
  ggtitle("Variable Distribution") + theme_classic() + 
  theme(plot.title = element_text(size = 18))

value_imputed <- data.frame(
  original = titanic_train$Age,
  
  imputed_zero = replace(titanic_train$Age,
                         is.na(titanic_train$Age), 0),
  
  imputed_mean = replace(titanic_train$Age,
                         is.na(titanic_train$Age), mean(titanic_train$Age, na.rm = TRUE)),
  
  imputed_median = replace(titanic_train$Age,
                         is.na(titanic_train$Age), median(titanic_train$Age, na.rm = TRUE))
)

value_imputed

h1 <- ggplot(value_imputed, aes (x = original)) + 
  geom_histogram(fill = "#ad1538", color = "#000000", position = "identity") + ggtitle("Original Distribution") + theme_classic()
h2 <- ggplot(value_imputed, aes (x = imputed_zero)) + 
  geom_histogram(fill = "#15ad4f", color = "#000000", position = "identity") + ggtitle("Imputed Zero Distribution") + theme_classic()
h3 <- ggplot(value_imputed, aes (x = imputed_mean)) + 
  geom_histogram(fill = "#1543ad", color = "#000000", position = "identity") + ggtitle("Imputed Mean Distribution") + theme_classic()
h4 <- ggplot(value_imputed, aes (x = imputed_median)) + 
  geom_histogram(fill = "#ad8415", color = "#000000", position = "identity") + ggtitle("Imputed Median Distribution") + theme_classic()

plot_grid(h1, h2, h3, h4, nrow = 2, ncol = 2)

library(mice)
titanic_numeric <- titanic_train %>%
  select(Survived, Pclass, SibSp, Parch, Age)
md.pattern(titanic_numeric)

# Activity 2
mice_imputed <- data.frame(original = titanic_train$Age, 
                           imputed_pmm = complete(mice(titanic_numeric, method = "pmm"))$Age,
                           imputed_cart = complete(mice(titanic_numeric, method = "cart"))$Age,
                           imputed_lasso = complete(mice(titanic_numeric, method = "lasso.norm"))$Age)
mice_imputed

h1 <- ggplot(mice_imputed, aes (x = original)) + 
  geom_histogram(fill = "#ad1538", color = "#000000", position = "identity") + ggtitle("Original Distribution") + theme_classic()
h2 <- ggplot(mice_imputed, aes (x = imputed_pmm)) + 
  geom_histogram(fill = "#15ad4f", color = "#000000", position = "identity") + ggtitle("Imputed Zero Distribution") + theme_classic()
h3 <- ggplot(mice_imputed, aes (x = imputed_cart)) + 
  geom_histogram(fill = "#1543ad", color = "#000000", position = "identity") + ggtitle("Imputed Mean Distribution") + theme_classic()
h4 <- ggplot(mice_imputed, aes (x = imputed_lasso)) + 
  geom_histogram(fill = "#ad8415", color = "#000000", position = "identity") + ggtitle("Imputed Median Distribution") + theme_classic()

plot_grid(h1, h2, h3, h4, nrow = 2, ncol = 2)

#Activity 3
missForest_imputed <- data.frame(original = titanic_numeric$Age,
                                 imputed_missForest = missForest(titanic_numeric)$ximp$Age)
missForest_imputed
h1 <- ggplot(missForest_imputed, aes(x = original)) + geom_histogram(fill = "#", color = "")