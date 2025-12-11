import pandas as pd
import scipy.stats as stats
import plotly.express as px

# Load csv into pandas dataframe
df = pd.read_csv('Confessing Romantic Interest Study (Responses).csv')
print(df)
# Statistics for Invidiaul Variables
# Sum of Social Anxiety Questions
df['SocialAnxiety'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']].sum(axis=1)
print(df['SocialAnxiety'].describe())
# t-test based off Peters et al. (2012) mean and stddev
mean_sa = 6.63
stddev_sa = 5.09
n = len(df['SocialAnxiety'].dropna())
t_score = (df['SocialAnxiety'].mean() - mean_sa) / (stddev_sa / (n ** 0.5))
print(f"T-score for Social Anxiety: {t_score}, p-value: {pd.Series([t_score]).apply(lambda x: 2 * (1 - stats.t.cdf(abs(x), df=n-1))).values[0]}, d={(df['SocialAnxiety'].mean() - mean_sa) / stddev_sa}")

# Comfort Statistics
print(df[['LowComfort', 'MediumComfort', 'HighComfort']].describe())

# Columns expected to be numeric; coerce non-numeric to NaN for safe stats
comfort_cols = ['LowComfort', 'MediumComfort', 'HighComfort']
numeric_cols = comfort_cols + ['SocialAnxiety']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# First Question: Does the comfort rating change depending on the risk level of the scenario?
# ANOVA test, DV: Comfort Level (rating 1-7), IV: Risk Level (Low, Medium, High)

# Drop rows lacking comfort ratings for ANOVA
anova_df = df.dropna(subset=comfort_cols)
low_risk = anova_df['LowComfort']
medium_risk = anova_df['MediumComfort']
high_risk = anova_df['HighComfort']
f_stat, p_value = stats.f_oneway(low_risk, medium_risk, high_risk)
print(f"ANOVA results for Comfort Level by Risk Level (n={len(anova_df)}): F-statistic = {f_stat}, p-value = {p_value}")

# boxplot visualization of comfort levels by risk level
melted_df = df.melt(value_vars=['LowComfort', 'MediumComfort', 'HighComfort'],
                        var_name='RiskLevel', value_name='ComfortLevel')
melted_df['RiskLevel'] = melted_df['RiskLevel'].map({
    'LowComfort': 'Low',
    'MediumComfort': 'Medium',
    'HighComfort': 'High'
})

# Ensure RiskLevel is categorical and ordered so boxes render distinctly
melted_df['RiskLevel'] = pd.Categorical(melted_df['RiskLevel'], categories=['Low', 'Medium', 'High'], ordered=True)

# Improve box appearance: larger figure, grouped boxes, wider boxes, show only outliers as points
fig = px.box(melted_df,
                x='RiskLevel',
             y='ComfortLevel',
             title='Comfort Level by Risk Level',
             labels={'RiskLevel': 'Risk Level', 'ComfortLevel': 'Comfort Level (1-7)'})
# fig.show()

"""Second Question: Relationship between social anxiety and comfort level
Correlation test, DV: Comfort Level (rating 1-7), IV: Social Anxiety Score
"""

# Drop rows lacking either comfort or social anxiety for correlation
corr_df = df.dropna(subset=numeric_cols)
social_anxiety = corr_df['SocialAnxiety']
comfort_levels = corr_df[comfort_cols].mean(axis=1)
correlation, p_value = stats.pearsonr(social_anxiety, comfort_levels)
print(
    f"Correlation between Social Anxiety and Comfort Level (n={len(corr_df)}): "
    f"Correlation = {correlation}, p-value = {p_value}"
)

# visualisation of social anxiety vs comfort level
fig2 = px.scatter(x=social_anxiety,
                  y=comfort_levels,
                  title='Social Anxiety vs Comfort Level',
                  labels={'x': 'Social Anxiety Score', 'y': 'Average Comfort Level (1-7)'},
                  trendline='ols')
# fig2.show()

# Question: How does proportion of face-to-face vs text confession vary by risk level?

confession_counts = {'Low': df['LowMethod'].value_counts(),
                     'Medium': df['MediumMethod'].value_counts(),
                     'High': df['HighMethod'].value_counts()}
confession_df = pd.DataFrame(confession_counts).fillna(0)
print(confession_df)

# ANOVA test for proportion of confession methods by risk level
face_to_face = confession_df.loc['Face-to-face conversation']
text_message = confession_df.loc['Text Message']
f_stat_method, p_value_method = stats.f_oneway(face_to_face, text_message)
print(f"ANOVA results for Confession Method by Risk Level: F-statistic = {f_stat_method}, p-value = {p_value_method}")

# Question: How does proportion of face-to-face vs text confession vary by social anxiety?
# regression analysis, DV: Proportion of face-to-face, IV: Social Anxiety Score
df['FaceToFaceProportion'] = df.apply(
    lambda row: 1 if row['LowMethod'] == 'Face-to-face conversation' else 0 +
                1 if row['MediumMethod'] == 'Face-to-face conversation' else 0 +
                1 if row['HighMethod'] == 'Face-to-face conversation' else 0, axis=1) / 3
regression_df = df.dropna(subset=['FaceToFaceProportion', 'SocialAnxiety'])
correlation_method, p_value_method = stats.pearsonr(regression_df['SocialAnxiety'], regression_df['FaceToFaceProportion'])
print(
    f"Correlation between Social Anxiety and Face-to-Face Proportion (n={len(regression_df)}): "
    f"Correlation = {correlation_method}, p-value = {p_value_method}"
)