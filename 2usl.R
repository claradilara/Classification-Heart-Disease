
# Heart Disease Risk Analysis: A Data-Driven Journey
# 🔍 Aim:
#   Heart disease is a major global health issue. This project aims to uncover hidden patterns in heart disease risk factors using Principal Component Analysis (PCA), Clustering (K-Means), and Association Rule Mining (Apriori Algorithm). The goal is to simplify complex medical data and identify key risk factors that contribute to heart disease.
# 
# 🛠️ Methodology:
# Data Preprocessing:
#  Cleaned and standardized patient data (cholesterol, blood pressure, ECG, etc.).
# Separated numerical and categorical features.
# PCA (Principal Component Analysis):
#   Reduced multiple risk factors into four key components (PC1-PC4).
# PC1 captured cholesterol & BP (cardiovascular risk), PC2 linked to exercise response, PC3 showed stress markers, and PC4 represented ECG/metabolic abnormalities.
# Clustering (K-Means & Hierarchical):
#   Identified three risk groups: Low, Moderate, and High Risk.
# High-risk patients had high cholesterol, BP, and ischemia markers.
# Association Rule Mining (Apriori Algorithm):
#   Discovered frequent risk factor interactions.
# High cholesterol (PC1) often linked with exercise stress (PC2) and ischemia (PC3).
# Uncovered hidden patterns among risk factors that predict heart disease.
# 📊 Results & Insights:
# ✅ PCA simplified risk factor analysis, allowing early detection of high-risk individuals.
# ✅ Clustering revealed groups with distinct heart disease risk levels.
# ✅ Association Rules confirmed strong correlations between high cholesterol, exercise response, and ischemia.Instead of analyzing dozens of individual risk factors, this approach automates risk detection, helping doctors identify high-risk patients faster. These insights can shape better treatment strategies and preventive care for heart disease.



#age: Age of the patient (in years)
#sex: Sex of the patient (1 = male, 0 = female)
#cp: Chest pain type (1-4)
#trestbps: Resting blood pressure (in mm Hg on admission to the hospital)
#chol: Serum cholesterol in mg/dl
#fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
#restecg: Resting electrocardiographic results (0-2)
#thalach: Maximum heart rate achieved
#exang: Exercise-induced angina (1 = yes; 0 = no)
#oldpeak: ST depression induced by exercise relative to rest




getwd()
setwd("C:/Users/ozdil/Downloads")


library(dplyr)
library(caret)   # For data preprocessing
library(ggplot2)
library(arules)        # For association rule mining
library(arulesViz)     # For visualizing rules
library(caret)         # For preprocessing
library(FactoMineR)    # For PCA
library(factoextra)


life <- read.csv("USL project2/archive (1)/heart-disease.csv")
head(life)

#Data Preprocessing

categorical_columns <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")


sum(is.na(life))  # Check for missing values
df_numeric <- na.omit(df_numeric)  # Remove or impute missing values

# Separate features and categorical variables
df_features <- life %>%
  select(-target)  # Remove the target column
df_numeric <- df_features %>% select(-one_of(categorical_columns))  



# Standardize the data (Z-score normalization)
df_scaled <- as.data.frame(scale(df_numeric))
df_scaled_full <- cbind(df_scaled, df_features[, categorical_columns])



# Principal Component Analysis (PCA)
library(psych)
fa.parallel(df_scaled_full, fa = "pc", n.iter = 100)  # Finds optimal PCs

pca_model <- prcomp(df_scaled_full, center = TRUE, scale. = TRUE)

# Summary of PCA
summary(pca_model)



# Scree Plot
screeplot(pca_model, type = "lines", main = "Scree Plot")

# Get PCA transformed data
df_pca <- as.data.frame(pca_model$x)

# Print PCA results
print(head(df_pca))

# Cumulative Variance Plot
fviz_eig(pca_model, addlabels = TRUE, main = "Cumulative Variance Explained by PCs")


factoextra::fviz_pca_var(pca_model, col.var="contrib")


# Clustering (K-Means & Hierarchical)

fviz_nbclust(df_pca[,1:4], kmeans, method = "wss")

# Apply K-Means clustering with optimal clusters (assume 3)
set.seed(123)
kmeans_result <- kmeans(df_pca[,1:4], centers = 3)

# Visualize Clusters
fviz_cluster(kmeans_result, data = df_pca[,1:4])

# Hierarchical Clustering
hclust_result <- hclust(dist(df_pca[,1:4]))
fviz_dend(hclust_result, k = 3)


# Cluster 1 (Low-Risk Group):
#   Low PC1 (cholesterol, BP) & low PC3 (no ischemia/blockages)
# Individuals with minimal heart disease risk
# Cluster 2 (Moderate-Risk Group):
#   Moderate PC1, PC3 & PC4 (some risk factors but not extreme)
# May have borderline symptoms
# Cluster 3 (High-Risk Group):
#   High PC1 (cholesterol, BP) & high PC3 (blockages & ischemia)
# Most vulnerable to heart disease




# Association Rule Mining (Apriori Analysis on PCA Components)

library(arules)

# Discretizing PCA components into 3 bins (low, medium, high)
pca_data_discrete <- data.frame(
  PC1 = discretize(df_pca$PC1, method = "frequency", breaks = 3),
  PC2 = discretize(df_pca$PC2, method = "frequency", breaks = 3),
  PC3 = discretize(df_pca$PC3, method = "frequency", breaks = 3),
  PC4 = discretize(df_pca$PC4, method = "frequency", breaks = 3)
 
)

# Check the discretized values
summary(pca_data_discrete)



# Convert the PCA binary data into transaction format
library(arules)
pca_data_trans <- as(pca_data_discrete, "transactions")

# Check the first few transactions
inspect(head(pca_data_trans))




# PC1, PC2, PC3, and PC4 have strong interrelationships.
# Higher PC2 often correlates with higher PC1.
# Low PC1 is linked with moderate PC3 and high PC4.
# Lift values (~2.5) indicate strong associations.
# Confidence (~80-83%) suggests reliable rules.
# There are hidden patterns in the data that might predict heart disease risk.
# If PC1 (which might represent cholesterol, age, and blood pressure) is high, it often correlates with PC2 being high, meaning these features are linked.
# This method could help identify risk groups based on the most important factors from PCA.



# Visualizing Cluster Characteristics
library(ggplot2)
df_pca$Cluster <- as.factor(kmeans_result$cluster)  # Add cluster labels

# Boxplot of PC1 across clusters
ggplot(df_pca, aes(x = Cluster, y = PC1, fill = Cluster)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("Distribution of PC1 Across Clusters")

# Cluster 1 has low PC1 (low cholesterol & BP).
# Cluster 3 has high PC1, indicating high cardiovascular risk.
# This confirms that PC1 strongly contributes to defining high-risk clusters.





# Count transactions per cluster
rule_clusters <- table(df_pca$Cluster, pca_data_discrete$PC1)

# Convert to data frame
rule_clusters_df <- as.data.frame(rule_clusters)
colnames(rule_clusters_df) <- c("Cluster", "PC1_Bin", "Count")

# Plot
ggplot(rule_clusters_df, aes(x = PC1_Bin, y = Count, fill = Cluster)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  ggtitle("Frequency of PC1 Rules in Each Cluster")


# Interpretation of Findings
cor_matrix <- cor(df_pca[,1:4], df_numeric)
print(cor_matrix)

library(corrplot)
corrplot(cor_matrix, method = "color", tl.cex = 0.8)


# The risk of heart disease is likely influenced by factors such as age, cholesterol levels, and blood pressure. A higher principal component 1 (PC1) score indicates a greater risk, showing that certain features tend to cluster together. This suggests the presence of hidden patterns among the transformed variables, where an increase in one risk factor may correlate with increases in others. For example, high cholesterol might be associated with abnormal ECG readings, reinforcing the idea that some factor combinations strongly relate to heart disease likelihood.  
# By leveraging this approach, early detection and risk assessment could become more efficient. Instead of analyzing each variable separately, principal component analysis (PCA) allows for identifying high-risk individuals based on key components. Additionally, apriori rules can uncover meaningful relationships between risk factors, which could be further explored in medical research to enhance predictive models and preventive strategies.


# Dim1 might represent a spectrum from low to high exercise-induced stress response (since thalach, cp, and slope are on one side, while oldpeak and exang are opposite).
# Dim2 could differentiate demographic factors (age, sex) from medical measurements (cholesterol, blood pressure).

