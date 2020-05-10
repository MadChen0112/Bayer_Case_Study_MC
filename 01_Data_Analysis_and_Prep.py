import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from textwrap import wrap
import  statistics

######### Bayer Case study, by Madhura Chendvankar #########

# PART 1 OF 3 : DATA ANALYSIS; VISUALIZATION AND IMPUTATION

# Step 1. Data structure analysis
df = pd.read_csv('Combined_user_job_desc.csv')
print(df.info())
# Observations
# 1. v25 and v30 have too many NaNs, imputation required
# 2. Categorical variables need deeper analysis and unique count calculation
print(df.head(5))
df_user_attr = df.iloc[:,5:]
print(df_user_attr.shape)
print(df_user_attr.describe())# mean and stdev of all user attributes are similar
print(df.groupby('has_applied').size()) #slightly imbalanced dataset, about 2:3 in favor of 1

# Step 2. Data anaylsis and visualization
# 2.1 Attribute 'company'
print(df.groupby('company').size()) #roughly equal distribution
company_names = sorted(df['company'].unique())
ones_comp, zeroes_comp = [],[]
for comp in company_names: # counting has_applied labels per company
    df_comp = df[df['company']==comp]
    x1 = df_comp.groupby('has_applied').size()
    zeroes_comp.append(x1[0]),ones_comp.append(x1[1])

print("Non applicants as per company :",zeroes_comp)
print("Applicants as per company",ones_comp)

# Visualization of applicants as per company
# Stacked bar chart for applicants as per company
N = len(company_names)
plt.figure(figsize=(5,4))
p1 = plt.bar(np.arange(N),ones_comp,width=0.7)
p2 = plt.bar(np.arange(N),zeroes_comp,bottom=ones_comp,width=0.7)
plt.xticks(np.arange(N),company_names)
for r1, r2 in zip(p1, p2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
plt.legend((p1[0],p2[0]),('Applied','Not Applied'))
plt.title('Application status for companies')
plt.xlabel('Companies'),plt.ylabel('Number of applicants')
plt.tight_layout()
plt.show() # looks the same for everyone

# Percentage application wise bar chart
perc_app_by_comp = [round(a*100/(a+b),2) for a,b in zip(ones_comp,zeroes_comp)]
plt.figure(figsize=(5,4))
p3 = plt.bar(np.arange(N),perc_app_by_comp,width=0.7)
plt.xticks(np.arange(N),company_names)
for r1 in p3:
    h1 = r1.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 , "%d" % h1, ha="center", va="bottom", color="black", fontsize=10)
plt.title('Application percentage for companies')
plt.xlabel('Companies'),plt.ylabel('Percentage applicants')
plt.ylim(0,100)
plt.tight_layout()
plt.show() # application % for all companies is roughly equal, can leave this attribute out during modeling

# 2.2 Attribute 'salary'

# Check 1 : How many times was salary mentioned in the job listing? (visualisation)

sal_col = df[['salary']]
sal_not_prov= sal_col.isna().sum()
sal_prov = 2000-sal_not_prov
lab_sal = 'Salary\n mentioned','Salary\nnot mentioned'
p_sizes = [sal_prov, sal_not_prov]
explode = (0,0.1)
plt.figure(figsize=(3,3))
plt.pie(p_sizes, explode=explode,labels=lab_sal,autopct='%1.1f%%',shadow=True,startangle=140,colors=['green','red'])
plt.axis('equal')
plt.title('Instances of salary \nmentioned in job listing')
plt.show()# 30% of job ads mentioned salary

# Check 2 : Do people apply more if salary is mentioned in job listing?

df_s_not_given = df[df['salary'].isna()]
df_s_given = df[df['salary'].notna()]
sal_stats = df_s_given.groupby('has_applied').size()
no_sal_stats = df_s_not_given.groupby('has_applied').size()
ns_app = [no_sal_stats[0],no_sal_stats[1]]
s_app = [sal_stats[0],sal_stats[1]]

lab_sal2 = 'Not applied','Applied'
# Pie chart visualization of has_applied label count when salary is mentioned and not mentioned
f,axes = plt.subplots(1,2)
axes[0].pie(s_app, explode=explode,labels=lab_sal2,autopct='%1.1f%%',shadow=True,startangle=90, colors=['greenyellow','green'])
axes[1].pie(ns_app, explode=explode,labels=lab_sal2,autopct='%1.1f%%',shadow=True,startangle=90,colors=['lightcoral','red'])
plt.text(-1, -1.5, 'Salary not mentioned', fontsize=12)
plt.axis('equal')
plt.title('Application statistics based\n on salary mention')
plt.tight_layout()
plt.show() # 75% people applied when salary was mentioned as compared to 50% when not given.

# Check 3 : Does the actual number matter when salary is mentioned? (Visualisation)
#plt.figure(figsize=(6,3.5))
sns.boxplot(x="has_applied",y="salary", data=df_s_given,palette=['darkorange','royalblue'],width=0.6)
plt.xticks([0,1],['not applied','applied'])
plt.ylabel('Salary',fontsize=15)
plt.xlabel('')
plt.title('Application statistics when salary is mentioned')
plt.show() # Actual number does not affect has_applied label

# Conclusion for salary : Only the mention matters, not the number. Can treat as a categorical variable: salary_mentioned and salary_n_mentioned

# 2.3 Attribute 'job_title_full'
# Check 1 : How many unique job titles?
job_title = df['job_title_full'].value_counts()
print('Total job titles :',len(job_title)) #156 unique job titles, highest freq = 22, lowest = 5

# Check 2 : Are some jobs more popularly applied for than others? (% application per job)

df["job_title_full"]=df["job_title_full"].astype('category')
df["job_title_full_cat"] = df["job_title_full"].cat.codes
list_jobs = list(df["job_title_full_cat"])
ones_jt, zeroes_jt = [],[]
for job_code in range(0,len(job_title)):
    df_job = df[df['job_title_full_cat']==job_code]
    x1 = df_job.groupby('has_applied').size()
    zeroes_jt.append(x1[0]),ones_jt.append(x1[1])

perc_app_by_job = [round(a*100/(a+b),2) for a,b in zip(ones_jt,zeroes_jt)] # see if % application for some jobs is higher than others

print("Min % application for a job is {} and max is {}".format(min(perc_app_by_job),max(perc_app_by_job)))

# Visualization of job titles popularly applied for
min_max_percs_index = [perc_app_by_job.index(max(perc_app_by_job)),perc_app_by_job.index(statistics.median(perc_app_by_job)),perc_app_by_job.index(min(perc_app_by_job)),
                 perc_app_by_job.index(statistics.mode(perc_app_by_job))]

min_max_percs = [max(perc_app_by_job),statistics.median(perc_app_by_job),statistics.mode(perc_app_by_job),min(perc_app_by_job)]
min_max_percs_index = [perc_app_by_job.index(x) for x in min_max_percs]

job_titl_for_plot = []

for i in min_max_percs_index:
    for j in range(0, 2000):
        if df['job_title_full_cat'][j]==i:
            job_titl_for_plot.append(df['job_title_full'][j])
            break

labels_minmax = [ '\n'.join(wrap(l, 20)) for l in job_titl_for_plot ]
p4 = plt.bar(np.arange(len(min_max_percs)),min_max_percs,width=0.5)
for r1 in p4:
    h1 = r1.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 , "%d" % h1, ha="center", va="bottom", color="black", fontsize=10)
plt.xticks(np.arange(len(min_max_percs)),labels_minmax)
plt.title('Application percentage for job titles')
plt.xlabel('Job titles'),plt.ylabel('Percentage applications')
plt.ylim(0,100)
plt.tight_layout()
plt.show() #some jobs are more popularly applied for than others, so necessary to include in modelling

# Check 3 : Are some jobs with particular keywords more popularly applied for?

df_job_keyword = df[['job_title_full']]
df_job_keyword = df_job_keyword.job_title_full.str.split(expand=True) # splitting long strings into columns
print(df_job_keyword.head()) #9 columns

all_words=[]
for i in range(0,9):
    all_words.append(df_job_keyword[i].tolist())
all_words = [a for b in all_words for a in b]

most_common=Counter(all_words).most_common(10)
print("Most common keywords in job titles",most_common) # ignore 'None' and '-'. First hit is 'Manager'

# now, check how many times ppl applied in manager position as opposed to non-manager

df_no_manager = df[~df.job_title_full.str.contains("Manager")]
df_manager = df[df.job_title_full.str.contains("Manager")]
print("When no *manager* in job title : ",df_no_manager.groupby('has_applied').size())
print("When *manager* in job title : ",df_manager.groupby('has_applied').size()) #not quite significant to include in modelling

# Check 4 : Are seniority levels affecting the application rate? (Junior/Senior/Lead)

df_with_split_strings = pd.concat([df, df_job_keyword], axis=1, sort=False) #make dataframe of job titles with 'junior' and unknown
df_job_low_level = df[~df.job_title_full.str.contains("Senior")]
df_job_low_level = df_job_low_level[~df_job_low_level.job_title_full.str.contains("Lead")]
print("When junior or unknown seniority level : ",df_job_low_level.groupby('has_applied').size())


df_job_high_level_1 = df[df.job_title_full.str.contains("Senior")] #make dataframe of job titles with 'senior' and 'lead'
df_job_high_level_2 = df[df.job_title_full.str.contains("Lead")]
df_job_high_level = pd.concat([df_job_high_level_1,df_job_high_level_2])
print("When senior or lead job level : ",df_job_high_level.groupby('has_applied').size())


low_lev_job_stats = df_job_low_level.groupby('has_applied').size()
high_lev_job_stats = df_job_high_level.groupby('has_applied').size()
llj_app = [low_lev_job_stats[0],low_lev_job_stats[1]]
hlj_app = [high_lev_job_stats[0],high_lev_job_stats[1]]

lab_sal2 = 'Not applied','Applied'
# plot of has_applied stats when salary is given and not given
f,axes = plt.subplots(1,2)
axes[0].pie(llj_app, explode=explode,labels=lab_sal2,autopct='%1.1f%%',shadow=True,startangle=90,colors=['darkorange','royalblue'])
axes[1].pie(hlj_app, explode=explode,labels=lab_sal2,autopct='%1.1f%%',shadow=True,startangle=90,colors=['darkorange','royalblue'])
plt.axis('equal')
plt.title('Application statistics based\n on job seniority level')
plt.tight_layout()
plt.show() # junior position application percentage is slightly higher, but likely to be negligible

#-------------------
# 2.4 User attributes v1-v56

# Check 1 : Are they correlated?
corrs= df_user_attr.corr().abs()
corrs_unstacked = corrs.unstack()
corrs_unstacked = corrs_unstacked.sort_values(kind="quicksort")

print ("Unstacked correlation coefficients",corrs_unstacked[3000:3100]) # User attributes are uncorrelated.

# Check 2 : Relation between user attributes and job description

# user attributes meaningless on their own, but meaningful based on jobs.
# example : job tag 1 (AI researcher, attributes v14 and v43) Visualization

df_job_tag_AI = df[df.job_title_full.str.contains("AI Researcher - Supplier Financing")]
f,axes = plt.subplots(2,2, sharey=True)

bp1 = sns.boxplot(x="has_applied",y="v14", data=df, ax=axes[0,0])
bp2 = sns.boxplot(x="has_applied",y="v43", data=df,  ax=axes[0,1],)
bp3 = sns.boxplot(x="has_applied",y="v14", data=df_job_tag_AI, ax=axes[1,0])
bp4 = sns.boxplot(x="has_applied",y="v43", data=df_job_tag_AI, ax=axes[1,1])

for i in range(0,2):
    for j in range(0,2):
        plt.sca(axes[i, j])
        plt.xticks(range(2), ['applied', 'not_applied'])
        plt.xlabel('')
        if i == 1:
            plt.title('\n'.join(wrap('For Job title : AI Researcher')))
        else:
            plt.title('Over the full dataset')

f.text(0, 0.5, 'User attributes', va='center', rotation='vertical',fontsize=12)
plt.tight_layout(h_pad=0.8,w_pad=0.5)
plt.show()

# Imputation strategy for NaN values in user attributes

df['salary'].fillna(0, inplace=True)
print('#########')
#print(df['salary'].head()) #temporarily replaced NA with 0, so it does not interfere with user attribute imputation
# (when modeling, salary column will be replaced by dummy variable anyway.)


# Since user attributes are dependent on job, imputation is done based on job and has_applied label
# e.g filter by Job title and has_applied label, then impute the mean of user attributes values into NaNs.
target_var_values = [0,1]
for job in range(0,len(job_title)):
    for elem in target_var_values:
        df[(df['job_title_full_cat'] == job) & (df['has_applied'] == elem)].fillna(df[(df['job_title_full_cat'] == job) & (df['has_applied'] == elem)].mean(),inplace=True)
        males = df[(df['job_title_full_cat'] == job) & (df['has_applied'] == elem)]
        males.fillna(males.mean(),inplace=True)
        df.update(males)
        print ('Job_title tag {} imputation in progress...'.format(job)) #slow process, needs to be tracked

print('Imputation complete!')

df.to_csv('Imputed_User_DS_FullBCode.csv',index=0) #will be the starting point of modeling code

# Check 3 : Feature importance index

df_wo_na=df.dropna()
df_user_attr_wo_na =df_wo_na.iloc[:,5:61]
rnd_clf = RandomForestClassifier(n_estimators=500,random_state=1)
rnd_clf.fit(df_user_attr_wo_na, df_wo_na['has_applied'])
for name, importance in zip(list(df_user_attr_wo_na), rnd_clf.feature_importances_):
    print(name, "=", importance) # No user attributes are particularly important on their own.

###### END OF PART 1 OF 3 : DATA ANALYSIS ##############################
###### NEXT PART : DUMMY VARIABLE CONCATENATION ##########






