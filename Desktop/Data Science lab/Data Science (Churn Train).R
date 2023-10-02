library(dplyr)
library(tidyverse)
library(dlookr)

# Read data 
churn <- read.csv("C:/Users/User/Dropbox/My PC (LAPTOP-9QUGRTLC)/Downloads/Churn_Train.csv")

# Check for missing values 
sum(is.na(churn))
summary(is.na(churn))

# Replace missing value (total charges) with mean 
churn <- churn%>% 
  mutate(Total.Charges = replace(Total.Charges, is.na(Total.Charges), median(Total.Charges, TRUE)))
is.na(churn$Total.Charges)
summary(is.na(churn))

# Brief overview of the data in the dataset
describe(churn)

# Check the normality of the dataset
normality(churn)

plot_normality(churn)

# Bivariate/ multivariate analysis 
correlate(churn)
correlate(churn, Tenure, Monthly.Charges, Total.Charges)

plot_correlate(churn)

# Perform EDA based on target variable
churn$Payment.Method <- as.factor(churn$Payment.Method)
categ <- target_by(churn, Payment.Method)

# EDA when target(output) is categorical & predictor(input) is numerical
num_cat <- relate(categ, Monthly.Charges)
num_cat
summary(num_cat)
plot(num_cat)

# EDA when both input and out is categorical
churn$Paperless.Billing <- as.factor(churn$Paperless.Billing)
cat_cat <- relate(categ, Paperless.Billing)
cat_cat
summary(cat_cat)
plot(cat_cat)

# EDA when both input and output is numerical
num <- target_by(churn, Monthly.Charges)
num_num <- relate(num, Total.Charges)
num_num
summary(num_num)
plot(num_num)

# EDA when input is categorical & output is numerical
cat_num <- relate(num, Payment.Method)
cat_num
summary(cat_num)
plot(cat_num)

churn %>%
  eda_web_report(target = 'Total.Charges', subtitle = 'Churn_Train',
                   output_dir = "C:/Users/User/Desktop/Data Science lab", output_file = "EDA.html", theme = "blue")

churn %>%
  eda_paged_report(target = "Total.Charges", subtitle = 'Churn_Train',
                   output_dir = "C:/Users/User/Desktop/Data Science lab", output_file = "EDA.pdf", theme = "blue",
                   output_format = "pdf")

# load the webshot library
library(webshot)

# Convert HTML to PDF as a bootstrap (Use this because there is some dependency issues with eda_paged_report)
webshot("C:/Users/User/Desktop/Data Science lab/EDA.html", "C:/Users/User/Desktop/Data Science lab/EDA.pdf")

