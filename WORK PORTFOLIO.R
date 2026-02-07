# Load libraries
library(tidyverse)
library(broom)
library(ggplot2)
library(caret)

# 1️⃣ Load the data
url <- "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic <- read.csv(url, stringsAsFactors = FALSE)

# 2️⃣ Quick look
head(titanic)
str(titanic)

# 3️⃣ Data Cleaning
titanic_clean <- titanic %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) %>%
  mutate(
    Survived = factor(Survived, levels = c(0,1), labels = c("No","Yes")),
    Pclass   = factor(Pclass),
    Sex      = factor(Sex),
    Embarked = factor(Embarked)
  ) %>%
  drop_na()  # remove rows with missing values

# 4️⃣ Exploratory Data Analysis
# Summary
summary(titanic_clean)

# Survival by Sex
sex_survival <- titanic_clean %>%
  group_by(Sex, Survived) %>%
  tally() %>%
  mutate(prop = n / sum(n))

# 5️⃣ Visualization: Survival counts
ggplot(titanic_clean, aes(x = Sex, fill = Survived)) +
  geom_bar(position = "fill") +
  labs(title="Proportion Survived by Sex")

# Age distribution by survival
ggplot(titanic_clean, aes(x = Age, fill = Survived)) +
  geom_histogram(position="identity", alpha=0.6, bins=25) +
  labs(title="Age Distribution by Survival")

# 6️⃣ Statistical test (Chi-square for categorical)
chi_sex <- chisq.test(table(titanic_clean$Sex, titanic_clean$Survived))
chi_pclass <- chisq.test(table(titanic_clean$Pclass, titanic_clean$Survived))

# Print test results
chi_sex
chi_pclass

# 7️⃣ Logistic Regression Model
model <- glm(Survived ~ Sex + Pclass + Age + Fare + SibSp + Parch,
             data = titanic_clean, family = binomial)

# Model summary (coefficients)
model_summary <- tidy(model)
model_summary

# Model interpretation
odds <- exp(coef(model))
odds

# 8️⃣ Predict & Accuracy (simple split)
set.seed(123)
trainIndex <- createDataPartition(titanic_clean$Survived, p = .8, list=FALSE)
train_data <- titanic_clean[trainIndex, ]
test_data  <- titanic_clean[-trainIndex,]

fit <- glm(Survived ~ Sex + Pclass + Age + Fare + SibSp + Parch,
           data = train_data, family = binomial)

pred <- predict(fit, test_data, type = "response")
pred_class <- ifelse(pred > 0.5, "Yes", "No")

confusionMatrix(factor(pred_class), test_data$Survived)

# 9️⃣ Save cleaned dataset
write.csv(titanic_clean, "data/titanic_clean.csv", row.names = FALSE)
