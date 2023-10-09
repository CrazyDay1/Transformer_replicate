library(titanic)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(cowplot)

Churn_Train

summary(is.na(Churn_Train)) # found out that total.charges has null value

#Basic Imputation methods
ggplot(Churn_Train, aes(Total.Charges)) + geom_histogram(color = "#000000", fill = "#0099F8") + ggtitle("Variable Distribution") + theme_classic() + theme(plot.title = element_text(size = 18))

value_imputed <- data.frame(
  original = Churn_Train$Total.Charges,
  
  imputed_zero = replace(Churn_Train$Total.Charges, is.na(Churn_Train$Total.Charges), 0),
  
  imputed_mean = replace(Churn_Train$Total.Charges, is.na(Churn_Train$Total.Charges), mean(Churn_Train$Total.Charges, na.rm = TRUE)),
    
  imputed_median = replace(Churn_Train$Total.Charges, is.na(Churn_Train$Total.Charges), median(Churn_Train$Total.Charges, na.rm = TRUE))
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

#Impute Missing Values in R with MICE
library(mice)

churn_numeric <- Churn_Train %>%
  select(Tenure, Monthly.Charges, Total.Charges)
md.pattern(titanic_numeric)

mice_imputed <- data.frame(original = Churn_Train$Total.Charges, 
                           imputed_pmm = complete(mice(churn_numeric, method = "pmm"))$Total.Charges,
                           imputed_cart = complete(mice(churn_numeric, method = "cart"))$Total.Charges,
                           imputed_lasso = complete(mice(churn_numeric, method = "lasso.norm"))$Total.Charges)
mice_imputed

h1 <- ggplot(mice_imputed, aes (x = original)) + 
  geom_histogram(fill = "#ad1538", color = "#000000", position = "identity") + ggtitle("Original Distribution") + theme_classic()
h2 <- ggplot(mice_imputed, aes (x = imputed_pmm)) + 
  geom_histogram(fill = "#15ad4f", color = "#000000", position = "identity") + ggtitle("Mice Imputed Zero Distribution") + theme_classic()
h3 <- ggplot(mice_imputed, aes (x = imputed_cart)) + 
  geom_histogram(fill = "#1543ad", color = "#000000", position = "identity") + ggtitle("Mice Imputed Mean Distribution") + theme_classic()
h4 <- ggplot(mice_imputed, aes (x = imputed_lasso)) + 
  geom_histogram(fill = "#ad8415", color = "#000000", position = "identity") + ggtitle("Mice Imputed Median Distribution") + theme_classic()

plot_grid(h1, h2, h3, h4, nrow = 2, ncol = 2)

#Imputation with R missForest Package
library(missForest)

missForest_imputed <- data.frame(
  original = churn_numeric$Total.Charges,
  imputed_missForest = missForest(churn_numeric)$ximp$Total.Charges
)

missForest_imputed

h1 <- ggplot(missForest_imputed, aes(x = original)) + geom_histogram(fill = "#ad1538", color = "#000000", position = "identity") + ggtitle("Original Distribution") + theme_classic()

h2 <- ggplot(missForest_imputed, aes(x = imputed_missForest)) + geom_histogram(fill = "#15ad4f", color = "#000000", position = "identity") + ggtitle("Miss Forest-imputed Distribution") + theme_classic()
plot_grid(h1, h2)

#Normalize data with scaling methods
log_scale = log(as.data.frame(Churn_Train$Total.Charges))

library(caret)
process <- preProcess(as.data.frame(Churn_Train$Total.Charges), method = c("range"))
scale_data <- as.data.frame(scale(Churn_Train$Total.Charges))
scale_data

# Label encoding to change char data to binary
gender_encode <- ifelse(Churn_Train$Gender == 'Male', 1, 0)
table(gender_encode)

new_dat = data.frame(Churn_Train$CustomerID, Churn_Train$Tenure, Churn_Train$Online.Security)
summary(new_dat)

dmy <- dummyVars(" ~ .", data = new_dat, fullRank = T)
dat_transformed <- data.frame(predict(dmy, newdata = new_dat))

glimpse(dat_transformed)