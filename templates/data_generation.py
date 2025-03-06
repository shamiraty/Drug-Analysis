import numpy as np
import pandas as pd
import datetime

np.random.seed(42)
num_records = 10000

current_year = datetime.datetime.now().year
years = np.random.choice(range(current_year - 10, current_year + 1), num_records)

data = {
    'Year': years,
    'Age': np.random.randint(15, 65, num_records),
    'Gender': np.random.choice(['Male', 'Female'], num_records),
    'EducationLevel': np.random.choice(['None', 'Primary', 'Secondary', 'Tertiary'], num_records),
    'Income': np.random.randint(50000, 500000, num_records),
    'EmploymentStatus': np.random.choice(['Employed', 'Unemployed', 'Student'], num_records),
    'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], num_records),
    'FamilyHistory': np.random.choice(['Yes', 'No'], num_records),
    'PeerPressure': np.random.choice(['High', 'Medium', 'Low'], num_records),
    'StressLevel': np.random.randint(1, 10, num_records),
    'AccessToDrugs': np.random.choice(['Easy', 'Moderate', 'Difficult'], num_records),
    'MentalHealthIssues': np.random.choice(['Yes', 'No'], num_records),
    'SocialSupport': np.random.randint(1, 10, num_records),
    'NeighborhoodEnvironment': np.random.choice(['HighRisk', 'ModerateRisk', 'LowRisk'], num_records),
    'LeisureActivities': np.random.choice(['None', 'Sports', 'Arts', 'Socializing'], num_records),
    'DrugType': np.random.choice(['Alcohol', 'Tobacco', 'Cannabis', 'Opiates', 'Stimulants'], num_records),
    'FrequencyOfUse': np.random.randint(0, 30, num_records),
    'YearsOfUse': np.random.randint(0, 15, num_records),
    'LegalIssues': np.random.choice(['Yes', 'No'], num_records),
    'FinancialProblems': np.random.choice(['Yes', 'No'], num_records),
    'RelationshipProblems': np.random.choice(['Yes', 'No'], num_records),
    'PhysicalHealthProblems': np.random.choice(['Yes', 'No'], num_records),
    'AcademicPerformance': np.random.randint(0, 100, num_records),
    'LifeSatisfaction': np.random.randint(1, 10, num_records),
    'UrbanRural': np.random.choice(['Urban', 'Rural'], num_records),
    'LifeStyle': np.random.choice(['Sedentary', 'Active', 'Mixed'], num_records),
    'Y': np.random.randint(0, 100, num_records)
}

df = pd.DataFrame(data)
df.to_csv('generated_data.csv', index=False)

print("CSV file 'generated_data.csv' has been created successfully!")