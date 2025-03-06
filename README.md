## INTELLIGENT PREDICTION SYSTEM FOR DRUG USE VICTIMS:
## PREDICTION, CAUSAL ANALYSIS, AND DEPENDENCY RISK DOCUMENTATION

![7](https://github.com/user-attachments/assets/31424376-cd52-415e-9aea-b1624d30f1b7)

- 📍 **Location**: Dodoma, Tanzania
- 📧 **Email**: [sashashamsia@gmail.com](mailto:sashashamsia@gmail.com)
- 📱 **WhatsApp**: [+255675839840](https://wa.me/255675839840)
- 🌐 **Demo**: [Online](https://openanalytics.pythonanywhere.com/)

## Disclaimer

⚠️ This application is intended for educational purposes only. The data used in this system is randomly generated through programming and does not represent any specific region or real-world dataset. The data has been balanced to avoid biases that could lead to emotionally charged or misleading results. The findings and analyses presented are for learning purposes and do not reflect the views or positions of any group or institution.


## Introduction

Drug abuse is a complex issue affecting individuals across various demographics. Drugs, in this context, refer to substances that alter a person's physical or mental state, often leading to addiction and detrimental health effects. The victims of drug abuse are diverse, encompassing individuals of all ages, genders, and socio-economic backgrounds. This system aims to provide insights into the factors influencing drug use and predict future trends.

## Problem

The prevalence of drug abuse poses a significant challenge to communities worldwide. Understanding the specific demographics and factors contributing to drug use is crucial for developing effective intervention strategies. This application addresses the need to:

* **Identify high-risk groups:** Determine the age ranges, genders, and socio-economic factors most associated with drug use.
* **Predict future trends:** Forecast drug use prevalence over the next three years to inform proactive measures.
* **Analyze contributing factors:** Understand the impact of various socio-economic and lifestyle factors on drug use frequency.

## Importance of Project

This project provides valuable insights that can be used to:

* Inform public health policies and interventions.
* Target resources to high-risk populations.
* Raise awareness about the factors contributing to drug abuse.
* Enable early detection and intervention programs.
* Provide data-driven recommendations for prevention and treatment.

## Main Objective

The primary objective of this application is to:

* Develop a predictive model to analyze and forecast drug use frequency (represented by 'Y').
* Identify the most influential factors contributing to drug use.
* Determine the most affected demographics (gender and age groups).
* Provide predictions for drug use prevalence in the next three years.
* Provide a summary of the data, and influential factors.
* Provide a list of recommendations based on the analysis.

## Methodology

The following technologies and methods were used in this project:

* **Python:** Used for data processing, model building, and application development.
* **Flask:** A web framework used to create the web application.
* **Pandas:** Used for data manipulation and analysis.
* **NumPy:** Used for numerical operations.
* **Scikit-learn (sklearn):** Used for machine learning, including:
    * `LinearRegression`: Used for predictive modeling.
    * `LabelEncoder`: Used for converting categorical data to numerical data.
    * `train_test_split`: Used for splitting the data into training and testing sets.
    * `r2_score`: Used for evaluating the model's performance.
* **datetime:** Used for obtaining the current year.
* **scipy.stats:** used for statistical analysis, such as calculating confidence intervals.
* **HTML/CSS/Bootstrap:** Used for the web application's front-end design.
* **DataTables:** Used for displaying data in a table format.
* **Font Awesome:** Used for icons.
* **Linear Regression:** the main model used to make predictions.
* **R2 Score:** Used to evaluate the effectiveness of the model.

### Data Description

The dataset used in this analysis includes a wide range of features to capture various aspects of drug use:

* **Year:** The year the data was recorded.
* **Age:** The age of the individual.
* **Gender:** The gender of the individual (Male/Female).
* **EducationLevel:** The individual's education level.
* **EmploymentStatus:** The individual's employment status.
* **Income:** The individual's income.
* **MaritalStatus:** The individual's marital status.
* **FamilyHistory:** Whether the individual has a family history of drug abuse.
* **PeerPressure:** The level of peer pressure the individual experiences.
* **StressLevel:** The individual's stress level.
* **AccessToDrugs:** The ease of access to drugs.
* **MentalHealthIssues:** Whether the individual has mental health issues.
* **SocialSupport:** The level of social support the individual receives.
* **NeighborhoodEnvironment:** The risk level of the individual's neighborhood.
* **LeisureActivities:** The individual's leisure activities.
* **DrugType:** The type of drug used.
* **FrequencyOfUse:** The frequency of drug use.
* **YearsOfUse:** The number of years of drug use.
* **LegalIssues:** Whether the individual has legal issues related to drug use.
* **FinancialProblems:** Whether the individual has financial problems related to drug use.
* **RelationshipProblems:** Whether the individual has relationship problems related to drug use.
* **PhysicalHealthProblems:** Whether the individual has physical health problems related to drug use.
* **AcademicPerformance:** The individual's academic performance.
* **LifeSatisfaction:** The individual's life satisfaction.
* **UrbanRural:** Whether the individual lives in an urban or rural area.
* **LifeStyle:** The individual's lifestyle.
* **Y:** The frequency of drug use (target variable).

### Analysis and Interpretation

1.  **Variable Importance (R2 Strength):**
    * The R2 score for each variable against 'Y' indicates the strength of their relationship. Variables with higher R2 scores contribute more significantly to predicting drug use frequency. These are ranked and displayed to show the most influential factors.
    * Interpretation: "Variable_name contributes to Y in the intensity of {{r2}}."
2.  **Gender and Age Group Analysis:**
    * The system identifies the most affected gender (male or female) based on R2 scores.
    * For each gender, the most affected age range is determined using R2 scores, highlighting specific demographic vulnerabilities.
3.  **Overall Variable Contribution:**
    * The strength of each variable's contribution to 'Y' is displayed, providing a comprehensive view of the factors influencing drug use.
4.  **Future Predictions (3 Years):**
    * A predictive model forecasts the number of individuals affected by drug use in the next three years, both overall and broken down by gender.
5.  **Lifestyle Impact:**
    * The system identifies the lifestyle most associated with drug use based on predictive modeling.
6.  **Age Group Predictions (3 Years):**
    * Predictions are made for drug use frequency in various age intervals (e.g., 10-20, 20-30, 30-40) for the next three years, considering upper and lower class boundaries to ensure accurate analysis.
7.  **Researcher Recommendations:**
    * Based on the analysis, the system provides actionable recommendations for addressing drug abuse, including targeted interventions and prevention strategies.

### Summary of Findings

From the surveyed drug use victims, a total of {{data_length}} individuals were analyzed from a population of 100,000. The estimated average age of the entire population is {{interval estimation from ? to ? use 95% CI or alpha 0.05}}. The following key findings were observed:

1.  **Primary Cause:** The most significant factor contributing to drug use is (Variable Importance - highest R2).
2.  **Most Affected Gender:** The gender most affected by drug use is (Gender Impact - highest R2).
3.  **Age Range Impact:** For males, the most affected age range is (Age and Gender Impact - male highest R2), and for females, it is (Age and Gender Impact - female highest R2).
4.  **Future Prevalence:** In the next three years ({{years}}), the predicted number of affected individuals will be (highest future prediction).
5.  **Lifestyle Influence:** The lifestyle most contributing to drug use is (Lifestyle Impact - highest R2).
6.  **Future Age Group Impact:** In the next three years ({{years}}), the age range with the highest predicted drug use frequency will be (Age Group Predictions - highest prediction), with a predicted number of (highest prediction value).

### Interpretation of Drug Analysis Results

#### Summary Overview

The provided data presents an analysis of drug use, focusing on identifying key factors, demographic impacts, and future predictions. The analysis is based on a sample of 10,000 individuals from a population of 100,000, with an estimated average age between 39.04 and 39.6 years (95% confidence).

#### Key Findings

* **Primary Cause:**
    * The most significant factor influencing drug use is **Peer Pressure**. This suggests that social influences play a crucial role in drug-related behaviors.
* **Most Affected Gender:**
    * **Females** are more affected by drug use than males, indicating a need for gender-specific intervention strategies.
* **Age Range Impact:**
    * For **males**, the most affected age range is **15-25 years**, highlighting the vulnerability of young males.
    * For **females**, the most affected age range is **45-55 years**, suggesting different life-stage influences for women.
* **Future Prevalence:**
    * In the next three years (2026-2028), the highest predicted number of affected individuals is approximately **49.44**.
* **Lifestyle Influence:**
    * A **sedentary lifestyle** is the most significant lifestyle factor contributing to drug use.

### Detailed Analysis

#### Predicted Drug Usage Frequency

* **Age Group Predictions:**
    * The predicted drug usage frequency varies across age groups, with the **55-65** age group showing the highest expected frequency (50.476 times per year), closely followed by the 15-25 age range (49.454). This implies that drug use is a concern across all age groups, but that it is most prevalent in those two age ranges.
* **Future Projections:**
    * The overall drug usage frequency is predicted to be approximately 49.435 times per year in the next three years.
    * **Female** drug usage is projected to be slightly higher than **male** drug usage.

#### Variable Importance

* **R2 Scores:**
    * **Peer Pressure** has the highest R2 score (0.0005), confirming its significant impact.
    * **Academic Performance**, **Frequency of Use**, and **Gender** also show relatively higher influence.
    * Many variables have very low or zero R2 scores, indicating minimal linear relationships with drug use frequency. This could mean that these variables do not have a linear relationship with Y, or that other models should be investigated.
* **Interpretation:**
    * These scores indicate the strength of the linear relationship between each variable and drug use frequency. Higher scores suggest a stronger relationship.

#### Gender Impact

* **R2 Scores:**
    * The R2 score for **females** (0.0081) is higher than that for **males** (0.0045), reinforcing the finding that females are more affected.

#### Lifestyle Impact

* **R2 Scores:**
    * A **sedentary lifestyle** has the highest R2 score (0.0108), indicating a strong correlation with drug use frequency.

#### Age and Gender Impact

* **R2 Scores:**
    * The highest R2 scores are observed in the **15-25** age range for males and the **45-55** age range for females, aligning with the earlier findings.

#### Overall Accuracy

* **R2 Score:**
    * The overall model accuracy is **-0.0033**, which is very low. This suggests that the linear regression model used may not be the most appropriate for this data, or that the features need more preprocessing. A negative R2 score indicates that the model performs worse than simply predicting the mean of the target variable.

#### Implications

* Targeted interventions should focus on addressing peer pressure, especially among young males and middle-aged females.
* Gender-specific strategies are essential, with a particular focus on female populations.
* Promoting active lifestyles could help mitigate drug use.
* The low overall accuracy of the model indicates a need for further model refinement or exploration of alternative modeling techniques.
* The low r2 values across the board indicate that a linear model is not a good fit for this data. Other models should be investigated.

### Researcher Recommendations

Based on the analysis, the following actions are recommended:

* Implement targeted interventions for high-risk age groups and genders.
* Focus on addressing the most influential factors contributing to drug use.
* Develop programs to promote healthy lifestyles and mitigate risk factors.
* Conduct further research to understand the underlying causes of drug abuse.
* Implement early detection and intervention programs in schools and communities.
* Create programs that deal with financial and legal problems, as this can be a trigger.
* Focus on education and job training to reduce unemployment and increase income.
* Create programs that focus on stress management.
