import pandas as pd

# PART 2 OF 3 : DUMMY VARIABLE CONCATENATION IN MAIN DATAFRAME

df = pd.read_csv('Imputed_User_DS_FullBCode.csv')

# 1. Data preparation

# 1.1. Dummy variables for 'job title'
dummy_jobtitle = pd.get_dummies(df.job_title_full_cat, prefix='JobTitle')
df = pd.concat([df, dummy_jobtitle], axis=1, sort=False)

# 1.2 Dummy variables for 'salary' : 1 = salary posted
sal_col = df['salary']
df.loc[df['salary'] > 0, 'salary'] = 1

dummy_salarygiven = pd.get_dummies(df.salary, prefix='salary_dummy')
df = pd.concat([df, dummy_salarygiven], axis=1, sort=False)

print("df shape after",df.shape)
#print(df.columns)
df=df.drop(['job_title_full','job_title_full_cat','company','salary','user_id'],axis=1)
#df=df.dropna()
print("df shape after",df.shape)


df.to_csv('Modeling_df.csv',index=0) # will be loaded in part 3 for modeling

###### END OF PART 2 OF 3 : DUMMY VARIABLE CONCATENATION ##############################
###### NEXT PART : MODELING #########