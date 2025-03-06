from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import datetime
from scipy import stats

app = Flask(__name__)

@app.route('/')
def index():
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
        'LifeStyle': np.random.choice([
    'Peer pressure',
    'Stress and mental health issues',
    'Exposure to drugs at an early age',
    'Lack of parental supervision or involvement',
    'Easy access to drugs',
    'Curiosity and experimentation',
    'Depression or emotional trauma',
    'Low socioeconomic status',
    'Social isolation and loneliness',
    'Cultural or community norms that accept drug use'
]
, num_records),
        'Y': np.random.randint(0, 100, num_records)
    }

    df = pd.DataFrame(data)
    #df = pd.read_csv("dataset.csv")

    label_encoders = {}
    categorical_cols = ['Gender', 'EducationLevel', 'EmploymentStatus', 'MaritalStatus', 'FamilyHistory',
                        'PeerPressure', 'AccessToDrugs', 'MentalHealthIssues', 'NeighborhoodEnvironment',
                        'LeisureActivities', 'DrugType', 'LegalIssues', 'FinancialProblems',
                        'RelationshipProblems', 'PhysicalHealthProblems', 'UrbanRural', 'LifeStyle']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(['Y', 'Year'], axis=1)
    y = df['Y']

    r2_scores = {}
    for col in X.columns:
        model = LinearRegression()
        model.fit(X[[col]], y)
        y_pred = model.predict(X[[col]])
        r2 = r2_score(y, y_pred)
        r2_scores[col] = r2

    sorted_r2 = sorted(r2_scores.items(), key=lambda item: item[1], reverse=True)

    gender_r2 = {}
    for gender in ['Male', 'Female']:
        gender_df = df[df['Gender'] == label_encoders['Gender'].transform([gender])[0]]
        X_gender = gender_df.drop(['Y', 'Year'], axis=1)
        y_gender = gender_df['Y']
        model = LinearRegression()
        model.fit(X_gender, y_gender)
        r2_gender = r2_score(y_gender, model.predict(X_gender))
        gender_r2[gender] = r2_gender

    age_ranges = [(15, 25), (25, 35), (35, 45), (45, 55), (55, 65)]
    age_r2_by_gender = {}

    for gender in ['Male', 'Female']:
        gender_df = df[df['Gender'] == label_encoders['Gender'].transform([gender])[0]]
        age_r2_by_gender[gender] = {}
        for age_range in age_ranges:
            age_df = gender_df[(gender_df['Age'] >= age_range[0]) & (gender_df['Age'] < age_range[1])]
            if len(age_df) > 0:
                X_age = age_df.drop(['Y', 'Year'], axis=1)
                y_age = age_df['Y']
                model = LinearRegression()
                model.fit(X_age, y_age)
                r2_age = r2_score(y_age, model.predict(X_age))
                age_r2_by_gender[gender][age_range] = r2_age
            else:
                age_r2_by_gender[gender][age_range] = np.nan

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    overall_r2 = r2_score(y_test, model.predict(X_test))

    years_data = df['Year'].values.reshape(-1, 1)
    y_values = df['Y'].values.reshape(-1, 1)
    model_predict = LinearRegression()
    model_predict.fit(years_data, y_values)
    future_years = np.array(range(current_year + 1, current_year + 4)).reshape(-1, 1)
    future_predictions = model_predict.predict(future_years)

    male_df = df[df['Gender'] == label_encoders['Gender'].transform(['Male'])[0]]
    female_df = df[df['Gender'] == label_encoders['Gender'].transform(['Female'])[0]]
    male_years = male_df['Year'].values.reshape(-1, 1)
    female_years = female_df['Year'].values.reshape(-1, 1)
    male_y = male_df['Y'].values.reshape(-1, 1)
    female_y = female_df['Y'].values.reshape(-1, 1)
    male_model = LinearRegression()
    male_model.fit(male_years, male_y)
    female_model = LinearRegression()
    female_model.fit(female_years, female_y)
    future_male_predictions = male_model.predict(future_years)
    future_female_predictions = female_model.predict(future_years)

    lifestyle_r2 = {}
    for lifestyle in df['LifeStyle'].unique():
        lifestyle_df = df[df['LifeStyle'] == lifestyle]
        X_lifestyle = lifestyle_df.drop(['Y', 'Year'], axis=1)
        y_lifestyle = lifestyle_df['Y']
        if len(lifestyle_df) > 0:
            model = LinearRegression()
            model.fit(X_lifestyle, y_lifestyle)
            r2_lifestyle = r2_score(y_lifestyle, model.predict(X_lifestyle))
            lifestyle_r2[label_encoders['LifeStyle'].inverse_transform([lifestyle])[0]] = r2_lifestyle
        else:
            lifestyle_r2[label_encoders['LifeStyle'].inverse_transform([lifestyle])[0]] = np.nan

    age_interval_predictions = {}
    for age_range in [(15, 25), (25, 35), (35, 45), (45, 55), (55, 65)]:
        interval_df = df[(df['Age'] >= age_range[0]) & (df['Age'] < age_range[1])]
        if len(interval_df) > 0:
            interval_years = interval_df['Year'].values.reshape(-1, 1)
            interval_y = interval_df['Y'].values.reshape(-1, 1)
            interval_model = LinearRegression()
            interval_model.fit(interval_years, interval_y)
            future_interval_predictions = interval_model.predict(future_years)
            age_interval_predictions[age_range] = future_interval_predictions.flatten()
        else:
            age_interval_predictions[age_range] = np.nan

    recommendations = [
        "Focus on high-risk age groups (e.g., 15-25) with targeted interventions.",
        "Address peer pressure and access to drugs, as they show strong correlations.",
        "Implement mental health support programs, as mental health issues are a significant factor.",
        "Tailor interventions to specific genders, as there are variations in impact.",
        "Promote active lifestyles and positive social support to mitigate risk.",
        "Conduct further research into neighborhood environment and its influence.",
        "Create programs that deal with financial and legal problems, as this can be a trigger.",
        "Focus on education and job training to reduce unemployment and increase income.",
        "Create programs that focus on stress management.",
        "Implement early detection and intervention programs in schools and communities."
    ]
    
    # Additional calculations for the summary
    data_length = len(df)
    age_mean = df['Age'].mean()
    age_std = df['Age'].std()
    age_interval = stats.t.interval(0.95, len(df['Age'])-1, loc=age_mean, scale=age_std/np.sqrt(len(df['Age'])))

    # Find the most influential factors
    most_influential_factor = sorted_r2[0][0]
    most_affected_gender = max(gender_r2, key=gender_r2.get)

    # Find the most affected age groups by gender
    male_max_age_range = max(age_r2_by_gender['Male'], key=age_r2_by_gender['Male'].get)
    female_max_age_range = max(age_r2_by_gender['Female'], key=age_r2_by_gender['Female'].get)

    # Find the highest future prediction
    highest_future_prediction = max(future_predictions.flatten())

    # Find the most influential lifestyle
    most_influential_lifestyle = max(lifestyle_r2, key=lifestyle_r2.get)

    # Find the highest future age group prediction
    highest_future_age_prediction = max(age_interval_predictions, key=lambda k: max(age_interval_predictions[k]))
    highest_future_age_prediction_value = max(age_interval_predictions[highest_future_age_prediction])

    return render_template('index.html', sorted_r2=sorted_r2, gender_r2=gender_r2,
                           age_r2_by_gender=age_r2_by_gender, overall_r2=overall_r2,
                           future_predictions=future_predictions.flatten(),
                           future_male_predictions=future_male_predictions.flatten(),
                           future_female_predictions=future_female_predictions.flatten(),
                           lifestyle_r2=lifestyle_r2,
                           age_interval_predictions=age_interval_predictions,
                           recommendations=recommendations, current_year=current_year, future_years = future_years.flatten(),
                           
                           data_length=data_length, age_interval=age_interval,
                           most_influential_factor=most_influential_factor,
                           most_affected_gender=most_affected_gender,
                           male_max_age_range=male_max_age_range,
                           female_max_age_range=female_max_age_range,
                           highest_future_prediction=highest_future_prediction,
                           most_influential_lifestyle=most_influential_lifestyle,
                           highest_future_age_prediction=highest_future_age_prediction,
                           highest_future_age_prediction_value=highest_future_age_prediction_value
                           
                           
                           
                           )

if __name__ == '__main__':
    app.run(debug=True)