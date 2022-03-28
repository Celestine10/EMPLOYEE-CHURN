# A Data Analysis and predcition of Employee churn with python.


#import modules
import pandas as pd  # for dataframes
import numpy as np
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
sns.set_style("darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing our Datasets

#Employess that left
left = pd.read_excel("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name="Employees who have left")
left.head()

#Viewing the datasets from below
left.tail()


emp_ex = pd.read_excel("Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx",sheet_name="Existing employees")
emp_ex.head()

#Viewing the datasets from below
emp_ex.tail()


# Reserving data for prediction 
"""I am taking just 0.01% of the Existing Employees dataset...I'm reserving it so I can use it to test for employees that may likely leave later on in the future."""



existing_emp = emp_ex.sample(frac=0.99,random_state=200)

reserved_data = emp_ex.drop(existing_emp.index)




# Basic information about our data

# Let's begin by inspecting the datasets


#shape of the datasets
print(left.shape)
print(existing_emp.shape)


# Column datatypes
print(existing_emp.dtypes)
print("\n")
print(left.dtypes)

print(existing_emp.columns)
print(left.columns)

# # Adding a new column  "status"
# Now we'll have to be creative.
# We are going to create a new column named *status* and set it to 0 for those employees that stayed.
# We'll also do the same for the same for the *left* dataset and set it to 1 for those employees that left.

# We will then explore the data and combine them so we can use them to make our prediction so we can predict those that will leave the company in the future.



existing_emp["status"] = 0


existing_emp.shape

left["status"]=1

left.head()#Let's confirm it


# Combining the Data


df = pd.concat([existing_emp,left],ignore_index=True)


#Let's Explore our Data and also Visualize


df.shape


df.isnull().sum()

"""
#The features are as follows: each row represent individual employee and attached attribute to them.
# - satisfaction_level (0–1)
# - last_evaluation (Time since last evaluation in years)
# - number_project (Number of projects completed while at work)
# - average_monthly_hours (Average monthly hours at workplace)
# - time_spend_company (Time spent at the company in years)
# - Work_accident (Whether the employee had a workplace accident)
# - status (Whether the employee left the workplace or not (1=left or 0=stayed))
# - promotion_last_5 years (Whether the employee was promoted in the last five years)
# - dept (Department in which they work for)
# - salary (Relative level of salary)"""



df.head(20)
df = df.drop("Emp ID",axis=1)#dropping the employee ID
reserved_data=reserved_data.drop("Emp ID", axis=1)


existing_emp["salary"].value_counts()

left["salary"].value_counts()

existing_emp_rate = (5093/11314) * 100
left_rate = (2172/3571) * 100
overall_rate = ((5093+2172)/14885)*100
print(existing_emp_rate,"%")
print(left_rate,"%")
print(overall_rate,"%")


# We can clearly see that over 50% of employees that left had low salaries while those that stayed  with low income are less than 50%...and the appr. 49% all the employees combined are on low salary.
# 
# 
# # Summary Statistics


left.describe()# Summarize numerical features for those that left


# Summarize numerical features
existing_emp.describe()

df.describe()


# Summarize categorical features
df.describe(include=['object'])


# status
df.status.value_counts()

percent_left = (len(left)/len(df))*100
percent_left

sns.countplot(df.status)


# Here, you can see that approx 3,571 left, and 11,314 stayed. The no of employees that left the company is 24% of the total employment dataset combined.

# Distribution
# Let's see how distributed our numerical columns are.

# Plot histogram grid with pandas plot
df.hist(figsize=(10,10), xrot=-45, edgecolor='black')
plt.show()


# We can use a loop to display bar plots for each of the categorical features.

# Plot bar plot for each categorical feature
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
plt.figure(figsize = (15,10))
for cat_feat in categorical_features:
    sns.countplot(y = cat_feat, data=df)
    plt.show()


# - It is obvious that from the overall data, majority of the employees are on low salary while just a few are on a high salary.
# 
# - Most of the employees are in the sales dept


pd.crosstab(df.dept,df.status).plot(kind="bar",color=("red","green"),figsize = (12,8))
plt.title("Turnover Frequency for Department")
plt.xlabel("Department")
plt.ylabel("Freq of Turnover")
plt.savefig("dept_turnover_bar_chart.png")
# Sales department has the highest number of employess that left, followed by technical and then support.


features =['number_project','time_spend_company','Work_accident','status', 'promotion_last_5years','dept','salary']

fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = df, hue='status')
    plt.xticks(rotation=90)
    plt.title("No. of Employees")
    
"""You can observe the following points in the above visualization:"""
# - Those employees who have the number of projects from 4 and above left the company it seems to like that they were overloaded with work.
# - Employees with 3 to 5 years experience are leaving more. The ones with more experience are not leaving because of affection/affiliation with the company.
# - Those who got promotion in last 5 years they didn't leave, i.e. all those left they didn't get the promotion in the previous 5 years.
# - More Employees left from the sales department.
# - Employees with low salary left the department more.


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),cmap="viridis",fmt='.2f', square=True,annot=True)
plt.savefig("time_spent_stautus_corr_heatmap_chart.png")
plt.show()
# From the heatmap above, we can deduce that the average time spent and monthly at the company is highly correlated to the status of those who left. Hence it might be part of the reason they left.
# We won't conclude yet until we've done our modelling.

 """Convert categorical variables to numerical"""
# Let's convert the categorical columns to numerical...so it can be reflected on the heatmap hence we'll see they impact on the employees' decision to leave.
# We will transorm the department and salary columns into numerical using pandas get_dummies.

df = pd.get_dummies(df, columns = ["salary","dept"],prefix=["salary","dept"])

plt.figure(figsize=(15,10))
sns.heatmap(df[["salary_high","salary_medium","salary_low","satisfaction_level","number_project","Work_               accident","promotion_last_5years","last_evaluation","status",]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.savefig("salary_status_corr_heatmap.png")
plt.show()
# From this heatmap, we can also tell that the low_salary is correlated with status as well.

# Finally, Group by 'status' and calculate the average value of each feature within each class
df.groupby("status").mean()

df.groupby("number_project").mean()


# # Summary:
# Those who left:
# - had less satisfaction at the company
# - spent the most hours at work in a month
# - spent more time at the company
# - had lower number of accidents
# - had less promotion in the last 5 years.
# - People with *3 years* of experience tend to leave often

"""The Following points can be drawn using above plots:"""
# - 1: The satisfaction level of people leaving the company is lower than people loyal to company. An important predictor.
# - 2: The average monthly hours for people leaving the company is more than people with less work hours. Point of frustration maybe :)
# - 3: Employees who got promotion in last 5 years tend to be loyal to company compared to people who were denied for promotion.
# - 4: Employees with less accidents tend to remain in the company as compared to people with higher number of accidents.
# - 5: Employee who has carried out more project tend to leave the company ans could be seen using: data.groupby(“number_project”).mean()

#Cluster Analysis:
"""Let's find out the groups of employees who left. You can observe that one of the most important factor for any employee to stay or leave is his satisfaction, salary, evaluation and time spent in performance at the company. So let's bunch them in the group of people using cluster analysis."""

# Satisfaction Level and Last evaluation
#  Wewill group them into 3 different clusters

#import module
from sklearn.cluster import KMeans
# Filter data
left_empl =  df[['satisfaction_level', 'last_evaluation']][df.status == 1]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(left_empl)


# Add new column "label" annd assign cluster labels.
left_empl['label'] = kmeans.labels_

# We plot a scatter plot
plt.figure(figsize=(12,8))
plt.scatter(left_empl['satisfaction_level'], left_empl['last_evaluation'], c=left_empl['label'],cmap='viridis')
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.title('3 Clusters of employees who left')
plt.savefig("Clustering based on satisfaction level on employees that left")
plt.show()


# From the 3 clusters:
# -  yellow = had high evaluation and high satisfaction rate...These ones must have left for due to other reasons.
# - green = had  high evaluation but low satisfaction...(They definitely thought about leaving because of low satisfaction)
# - Purple = Relatively low evaluation and low satisfaction rate. Hence the reason they left.


# ## Satisfaction Level and Average monthly hours
#import module
from sklearn.cluster import KMeans
# Filter data
left_empl =  df[['satisfaction_level', 'average_monthly_hours']][df.status == 1]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(left_empl)


# Add new column "label" annd assign cluster labels.
left_empl['label'] = kmeans.labels_

# We plot a scatter plot
plt.figure(figsize=(12,8))
plt.scatter(left_empl['satisfaction_level'], left_empl['average_monthly_hours'], c=left_empl['label'],cmap='coolwarm')
plt.xlabel('Satisfaction Level')
plt.ylabel('Average Monthly hours')
plt.title('3 Clusters of employees who left')
plt.savefig("Clustering based on time spent and satisfaction level for employess that left")
plt.show()


# From the 3 clusters:
# -  maroon = Majority spent a lot of time at the office with low satisfaction rate. Although few had spent much time with high satisfaction rate...it seems they had a higher promotion rate, hence they were satisfied.
# 
# - blue = spent an average amount of time with mostly high satisfaction rate...We'll conclude later after we,ve built our model.
# 
# - gray = Relatively spent lower time with low satisfaction rate. They would most likely have left due to lack of promotion and/or low salary.

# Existing Employes that may likely leave the company in the near future 


#import module
from sklearn.cluster import KMeans
# Filter data
exist_empl =  df[['satisfaction_level', 'average_monthly_hours']][df.status == 0]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(exist_empl)


# Add new column "label" annd assign cluster labels.
exist_empl['label'] = kmeans.labels_

# We plot a scatter plot
plt.figure(figsize=(15,10))
plt.scatter(exist_empl['satisfaction_level'], exist_empl['average_monthly_hours'], c=exist_empl['label'],cmap='coolwarm')
plt.xlabel('Satisfaction Level')
plt.ylabel('Average Monthly hours')
plt.title("3 Clusters of Existing Employees")
plt.savefig("Clustering based on average_monthlhy_hours and satisfaction level")
plt.show()


# So those that may likely leave the company in future:
# - blue = Those who spent a lot of hours(>=225 hrs) but have a low satisfaction rate.
# - red = Those that spend a monthly average >=170 hrs with low satisfaction rate(Not that many though).
# - gray = They spend a low monthly average <=170 hrs with high satisfaction rate. Majority of these ones will most likely remain in the company. The ones with very low satisfaction rate may also leave due to other factors.


#import module
from sklearn.cluster import KMeans
# Filter data
exist_empl =  df[['satisfaction_level', 'last_evaluation']][df.status == 0]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(exist_empl)


# Add new column "label" annd assign cluster labels.
exist_empl['label'] = kmeans.labels_

# We plot a scatter plot
plt.figure(figsize=(15,10))
plt.scatter(exist_empl['satisfaction_level'], exist_empl['last_evaluation'], c=exist_empl['label'],cmap='coolwarm')
plt.xlabel('Satisfaction Level')
plt.ylabel('last_evaluation')
plt.title("3 Clusters of Existing Employees")
plt.savefig("Clustering based on last_evaluation and satisfaction level")
plt.show()


# ## From this cluster, we can conclude that based on evaluation and satisfaction, the cluster of employees with gray all have low satisfaction and will therefore likely leave in the future.


#import module
from sklearn.cluster import KMeans
# Filter data
exist_empl =  df[['satisfaction_level', 'number_project']][df.status == 0]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(exist_empl)


# Add new column "label" and assign cluster labels.
exist_empl['label'] = kmeans.labels_

# We plot a scatter plot
plt.figure(figsize=(15,10))
plt.scatter(exist_empl['number_project'], exist_empl['satisfaction_level'], c=exist_empl['label'],cmap='coolwarm')
plt.xlabel('Satisfaction Level')
plt.ylabel('number_project')
plt.title("3 Clusters of Existing Employees")
plt.savefig("Clustering based on number_project and satisfaction level")
plt.show()

# Building a Prediction Model
# ### Pre-Processing Data
# Machine learning algorithms require numerical input data, we have already conerted all the categorical data earlier so let's proceed.


""" The data will be splitted into two sets in the ration 70% (for training set) and 30%(for the test set). 
# Recall that we also have some reserved data of existing exployees...We shall be using them to predict the employees that will likely leave the company."""

X=df.drop("status",axis=1)
y=df['status']

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test


""" Let's build our employee churn prediction model.
# We'll be using the GradientBoostingClassifier and the Random ForestClassifier algorithms. We will then fit the model and make our predictions."""


#Import Gradient Boosting Classifier model
from sklearn.ensemble import GradientBoostingClassifier

#Create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

#Train the model using the training sets
gb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gb.predict(X_test)


# ## Evaluating Model Performance

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",round(metrics.accuracy_score(y_test, y_pred)*100,2))
# Model Precision
print("Precision:",round(metrics.precision_score(y_test, y_pred)*100,2))
# Model Recall
print("Recall:",round(metrics.recall_score(y_test, y_pred)*100,2))
#Accuracy of our training set
print("Training set score: {:.2f}".format(gb.score(X_train, y_train)*100,2))


# Our model performed very well on the test set. Let's check how many it predicted rightly and wrongly by using the confusion matrix


# Import confusion_matrix
from sklearn.metrics import  confusion_matrix

# Display confusion matrix for y_test and pred
confusion_matrix(y_test, y_pred)


# #### Summary
# - 3407 = Those that won't leave ---correctly predicted
# - 968 = Those that left --- correctly predicted
# - 52 = Those that won't leave ---- wrongly predicted.
# - 39 = Those that left --- wrongly predicted.
# 
# Let's use the Random forest algorithm.

# Import RandomForestClassifier
from sklearn.ensemble import (RandomForestClassifier)

rf = RandomForestClassifier()
rf =  RandomForestClassifier(n_estimators=10,riterion="entropy",random_state=0)
rf.fit(X_train,y_train)


rf_pred = rf.predict(X_test)
y_prob =  rf.predict_proba(X_test)#Probability of the predictions


print("Accuracy:",round(metrics.accuracy_score(y_test, rf_pred)*100,2))
# Model Precision
print("Precision:",round(metrics.precision_score(y_test, rf_pred)*100,2))
# Model Recall
print("Recall:",round(metrics.recall_score(y_test, rf_pred)*100,2))
print("Training set score: {:.2f}".format(rf.score(X_train, y_train)*100,2))
print(y_prob)


# Display confusion matrix for y_test and pred
confusion_matrix(y_test, rf_pred)


# Summary
# - 3437 = Those that won't leave ---correctly predicted
# - 988 = Those that left --- correctly predicted
# - 32 = Those that won't leave ---- wrongly predicted.
# - 9 = Those that left --- wrongly predicted.
# 
# Our Random Forest performed better.
# 
# Remember our reserved data?
# Let's test it
#Converting categorical columns to numerical
reserved_data = pd.get_dummies(reserved_data, columns = ["salary","dept"],prefix=["salary","dept"]

reserved_data = reserved_data.drop("Emp ID",axis=1)
                               
reserved_data.shape

rf_pred = rf.predict_proba(reserved_data)

rf_pred[:5]

"""We will check for feature importance to see which feature greatly influenced our results"""         

plt.figure(figsize=(10,8))
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh') t
plt.savefig("Random forest_feature_importance.png")

#Feature importance for gradient boosting
plt.figure(figsize=(10,8))
feat_importances = pd.Series(gb.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh') #Remember to plot
plt.savefig("Gradient_boosting_feature_importance.png")

#Feature Importance for our Reseved Data                     
plt.figure(figsize=(10,8))
feat_importances = pd.Series(rf.feature_importances_, index=reserved_data.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh') #Remember to plot
plt.savefig("Reserved_data_feature_importance.png")
plt.show()

""" # Summary
# From the feature importance plots, we can definitly conclude that *Employee satisfaction* had the most important influence on the employee's decision to either leave or stay. Other influencing features were **time spent in the company, number of projects, last_evaluation and Average monthly hours**"""