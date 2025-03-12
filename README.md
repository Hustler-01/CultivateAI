# Crop-Recommendation-System
You can check live at https://cultivateai-at2y.onrender.com/
![image](https://github.com/user-attachments/assets/44680544-5050-4dde-913f-a42e54b4e7e2)
### 1. Objective
The Crop Recommendation System aims to assist farmers in identifying the most suitable crops for cultivation based on environmental and soil conditions. By leveraging advanced machine learning models and analyzing data such as soil nutrient content, rainfall, temperature, and humidity, the system provides data-driven recommendations that optimize agricultural productivity and sustainability.

### 2. Introduction
Agriculture plays a pivotal role in sustaining livelihoods and ensuring food security. However, selecting the appropriate crop to grow often involves complex decisions influenced by diverse factors such as soil characteristics, climatic conditions, and market demands. Traditionally, these decisions have relied heavily on experience and intuition, which can lead to inefficiencies and suboptimal yields.

This project seeks to develop a Crop Recommendation System that bridges this gap using data-driven insights. By employing supervised learning algorithms, the system predicts the most suitable crop for a given set of input parameters, thereby empowering farmers to make informed decisions.
The primary motivation for this project stems from the increasing challenges faced by the agricultural sector due to climate change, population growth, and resource constraints. The implementation of such a system has the potential to revolutionize agricultural practices by enhancing productivity, ensuring resource efficiency, and supporting sustainable farming.

### 3. Data Description
The dataset used for this project was sourced from kaggle (https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) and contains the following features:

Nitrogen (N): Nitrogen content in the soil (kg/ha).

Phosphorus (P): Phosphorus content in the soil (kg/ha).

Potassium (K): Potassium content in the soil (kg/ha).

Temperature (°C): Average temperature during the growing season.

Humidity (%): Atmospheric humidity.

pH: Soil pH value.

Rainfall (mm): Annual rainfall received.

Crop: Target variable indicating the suitable crop.

This dataset consists of 2,200 rows and 8 features, with crops classified into 22 distinct categories.

### 4. Methodology
#### a. Data Acquisition
The dataset was collected from agricultural research platforms, ensuring a comprehensive range of soil and climate conditions to cover diverse geographies and cropping patterns.
#### b. Data Preparation
Exploratory Data Analysis (EDA):

Analyzed feature distributions.
Identified missing values and outliers.

Data Cleaning:

Addressed missing data by imputing values where applicable.
Removed anomalies to maintain data consistency.

Data Transformation:
Scaled features using normalization to ensure uniformity in model input.
Encoded target labels (crops) into numerical format for machine learning compatibility.
#### c. Data Preprocessing
Feature Selection: Selected essential features based on their relevance to crop yield prediction.

Train-Test Split: Split the dataset into training (80%) and testing (20%) subsets to evaluate model performance effectively.



#### d. Model Development
Algorithms Used:
  
  Random Forest
  
  Naive Bayes
  
  Decision Tree
  
  Support Vector Machine (SVM)
  
  K-Nearest Neighbors (KNN)
  
  Logistic Regression
  
Training: Models were trained using the training dataset. Hyperparameter tuning was conducted to optimize performance.

Evaluation Metrics: Models were evaluated using accuracy, precision, recall, and F1-score.
  
  Random Forest: 99.3%
  
  Naive Bayes: 99.5%
  
  Decision Tree: 98.6%
  
  Support Vector Machine (SVM): 96.8%
  
  K-Nearest Neighbors (KNN): 95.9%
  
  Logistic Regression: 96.3%

### 5. Data Visualizations
   
   Distribution of Soil Nutrients: Visualizations such as histograms and boxplots were created to analyze the distribution of Nitrogen, Phosphorus, and Potassium 
levels.       
   
   Correlation Heatmap: Highlighted correlations among features like pH, rainfall, and humidity.
   
   Crop Distribution: A bar plot representing the frequency of each crop type in the dataset.
   
   Distribution of Nitrogen: The plot shows the distribution of nitrogen levels is bimodal (has multiple peaks), suggesting that the data could represent different 
groups or conditions.

### 6. Implementation
Libraries Used: 
pandas, numpy, scikit-learn, matplotlib, seaborn

Model Training and Validation:

Conducted using scikit-learn’s pipeline for preprocessing and classification.

Key Steps:
 
 Data loading and preprocessing.
 Feature engineering (scaling and encoding).
 Model building and evaluation.

Deployment: The model was exported as a serialized file (using joblib) and integrated into a user-friendly interface for end-user accessibility.



### 7. Web Interface
The project includes a user-friendly web interface developed using Flask. This interface enables users to input soil and environmental parameters and receive crop recommendations instantly.

Link- https://cultivateai-at2y.onrender.com/

User Interaction

Input fields for:
Nitrogen (N), Phosphorus (P), Potassium (K).
Temperature, Humidity, pH, Rainfall.

"Get Recommendation" button to generate results.
Screenshot
The following screenshot demonstrates the interface:

### 8. Evaluation and Validation
The Naive Bayes model achieved the highest accuracy of 99.5% on the testing dataset. Key performance metrics are as follows:
Precision: 1.00
Recall: 1.00
F1-Score: 1.00
The confusion matrix indicated robust performance across all crop categories.

### 9. Inference
The Crop Recommendation System demonstrates significant potential in aiding farmers by recommending suitable crops based on environmental and soil conditions. Key insights include:

Nitrogen and pH levels are critical determinants of crop suitability.

Rainfall patterns significantly influence predictions for water-intensive crops.

Machine learning algorithms, particularly Random Forest, provide reliable recommendations with high accuracy.

### 10. Limitations and Future Incorporations
Limitations:

The dataset is geographically limited and may not represent global agricultural conditions.
The model does not incorporate real-time environmental changes.

Lack of integration with market dynamics (e.g., crop prices, demand trends).
Future Enhancements:

Incorporate More Data: Expand the dataset to include global soil and climatic conditions.

Hybrid Models: Combine machine learning with expert systems for enhanced accuracy.

Real-Time Monitoring: Integrate IoT devices for real-time data collection on soil and weather conditions.

Economic Factors: Include market trends and pricing data for more holistic recommendations.

Mobile Application: Develop a mobile app to make the system accessible to farmers in remote areas.

### 11. References
Agricultural Dataset Repository (Kaggle)


Scikit-learn Documentation: https://scikit-learn.org
Research Papers on Crop Prediction Models
