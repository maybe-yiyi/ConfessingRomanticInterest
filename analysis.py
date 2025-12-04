import pandas as pd
# load csv into pandas dataframe
df = pd.read_csv('Confessing Romantic Interest Study (Responses).csv')

# First Question: Does the comfort rating change depending on the risk level of the scenario?
# ANOVA test, DV: Comfort Level (rating 1-7), IV: Risk Level (Low, Medium, High)
import scipy.stats as stats
low_risk = df['LowComfort']
medium_risk = df['MediumComfort']
high_risk = df['HighComfort']
f_stat, p_value = stats.f_oneway(low_risk, medium_risk, high_risk)
print(f"ANOVA results for Comfort Level by Risk Level: F-statistic = {f_stat}, p-value = {p_value}")

# boxplot visualization of comfort levels by risk level
import plotly.express as px
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

# Second Question: Relationship between social anxiety and comfort level
# Correlation test, DV: Comfort Level (rating 1-7), IV: Social Anxiety Score
social_anxiety = df['SocialAnxiety']
comfort_levels = df[['LowComfort', 'MediumComfort', 'HighComfort']].mean(axis=1)
correlation, p_value = stats.pearsonr(social_anxiety, comfort_levels)
print(f"Correlation between Social Anxiety and Comfort Level: Correlation = {correlation}, p-value = {p_value}")

# visualisation of social anxiety vs comfort level
fig2 = px.scatter(x=social_anxiety,
                  y=comfort_levels,
                  title='Social Anxiety vs Comfort Level',
                  labels={'x': 'Social Anxiety Score', 'y': 'Average Comfort Level (1-7)'},
                  trendline='ols')
fig2.show()
