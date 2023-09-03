#!/usr/bin/env python
# coding: utf-8

# # 311_NYC Customer Service Requests Analysis

# ### Import Required Libraries

# In[172]:


#import the required Libraries
import pandas as pd
import numpy as np
from pandas import Series
from datetime import datetime as dt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import the dataset & Visualize the dataset

# In[113]:


df_311_CSRA = pd.read_csv('311_Service_Requests_from_2010_to_Present.csv')
df_311_CSRA.head(30)


# In[114]:


df_311_CSRA.tail()


# ### Print the columns of the DataFrame

# In[115]:


df_311_CSRA.columns


# ### shape of the dataset

# In[116]:


df_311_CSRA.shape


# ### variables with null values

# In[117]:


df_nulldata=df_311_CSRA.isnull().sum()
df_nulldata


# ## Data Exploratory Analysis:

# ### frequency plot to show the number of null values in each column of the DataFrame

# In[118]:


plt.rcParams["figure.figsize"] = [15, 8]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()
df_nulldata.plot(ax=ax, kind='bar', xlabel='Null data', ylabel='frequency')
plt.xticks(rotation=45,ha='right')
plt.show()


# ### Missing value treatment

# In[119]:


df_311_CSRA['Closed Date'].isnull().sum()


# In[120]:


df_311_CSRA=df_311_CSRA.dropna(subset=['Closed Date'])


# In[121]:


df_311_CSRA.shape


# In[122]:


df_311_CSRA.info()


# In[123]:


df_311_CSRA['Closed Date']=pd.to_datetime(df_311_CSRA['Closed Date'])
df_311_CSRA['Created Date']=pd.to_datetime(df_311_CSRA['Created Date'])


# In[124]:


df_311_CSRA.info()


# In[125]:


df_311_CSRA.head()


# In[126]:


df_311_CSRA['Closed_Date']=df_311_CSRA['Closed Date'].dt.date
df_311_CSRA['Closed_Time']=df_311_CSRA['Closed Date'].dt.time
df_311_CSRA['Created_Date']=df_311_CSRA['Created Date'].dt.date
df_311_CSRA['Created_Time']=df_311_CSRA['Created Date'].dt.time


# In[127]:


df_311_CSRA.head()


# In[128]:


df_311_CSRA.tail()


# In[129]:


df_311_CSRA=df_311_CSRA[df_311_CSRA['Created_Date']<=df_311_CSRA['Closed_Date']]


# In[130]:


df_311_CSRA.shape


# ### Time elapsed in closed and creation date

# In[131]:


df_311_CSRA['time_elapsed'] =df_311_CSRA['Closed Date'] -df_311_CSRA['Created Date']


# ### Convert the calculated date to seconds to get a better representation

# In[132]:


from datetime import datetime
import time
df_311_CSRA['time_elapsed_seconds']=(df_311_CSRA['time_elapsed']).dt.total_seconds()


# In[133]:


df_311_CSRA.head()


# ### View the descriptive statistics for the newly created column

# In[134]:


(df_311_CSRA['time_elapsed_seconds']).describe()


# In[135]:


df_311_CSRA.columns


# ### Check the number of null values in the Complaint_Type and City columns

# In[136]:


#Check the number of null values in the Complaint_Type and City columns
#null values in the Complaint_Type columns
df_311_CSRA['Complaint Type'].isnull().sum()


# In[137]:


#null values in the City columns
df_311_CSRA['City'].isnull().sum()


# ###  Impute the NA value with Unknown City

# In[138]:


#Impute the NA value with Unknown City
df_311_CSRA['City']=df_311_CSRA['City'].fillna('Unknown_city')


# In[139]:


df_311_CSRA['City'].isnull().sum()


# In[140]:


complaintstype=df_311_CSRA['Complaint Type'].unique()
complaintstype


# In[141]:


cities=df_311_CSRA['City']
Max_compliants_city=df_311_CSRA['City'].value_counts()
 
percentage_compliants_city=Max_compliants_city/Max_compliants_city.sum()*100
freq_of_City_Complaints=pd.DataFrame({'Max_compliants_city':Max_compliants_city,'percentage_compliants_city':percentage_compliants_city})
freq_of_City_Complaints
#complaints=df_311_CSRA['Complaint Type']
#sns.countplot


# ###  Frequency plot for the complaints in each city

# In[142]:


#Draw a frequency plot for the complaints in each city
Max_compliants_city.plot(kind='bar',figsize=[15,8],title='complaints in each city')


# In[143]:


df_311_CSRA['Borough'].head(5)


# In[144]:


brooklyn_data = df_311_CSRA[df_311_CSRA['City']=='BROOKLYN']
brooklyn_data.head()


# ### Scatter and Hexbin plot of the concentration of complaints across Brooklyn

# In[145]:


#scatter plot
plt.figure(figsize=(10,6))
plt.scatter(brooklyn_data['Longitude'],brooklyn_data['Latitude'], alpha=0.5)
plt.title('Scatter plot of complaints concentration in Brooklyn')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()


# In[146]:


#hexbin plot
plt.figure(figsize=(10,6))
plt.hexbin(brooklyn_data['Longitude'],brooklyn_data['Latitude'], gridsize=50,cmap='inferno')
plt.title('Scatter plot of complaints concentration in Brooklyn')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Density')
plt.show()


# ## 3. Find major types of complaints:
# 

# ### 3.1 Plot a bar graph to show the types of complaints

# In[147]:


plt.figure(figsize=(12,6))
complaint_frequency_NYC = df_311_CSRA['Complaint Type'].value_counts()
complaint_frequency_NYC.plot(kind='bar')
plt.xlabel('Complaint Type')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[148]:


complaintstype=df_311_CSRA['Complaint Type'].unique()
complaintstype


# ### 3.2 Check the frequency of various types of complaints for NewYork City

# In[149]:


#3.2 Check the frequency of various types of complaints for NewYork City
complaint_frequency_NYC = df_311_CSRA['Complaint Type'].value_counts()
print('Frequency of various Type of complaints in New York:')
print(complaint_frequency_NYC)


# In[150]:


Citytype=df_311_CSRA['City'].unique()
Citytype


# In[151]:


NewYork_Complaints = df_311_CSRA[df_311_CSRA['City']=='NEW YORK']
NewYork_Complaints.head(2)


# In[152]:


Complaint_frequency_NewYork = NewYork_Complaints['Complaint Type'].value_counts()
print('Frequency of various Type of complaints in New York :')
print(Complaint_frequency_NewYork)


# ### 3.3 Find the top 10 complaint types in NewYork City
# 

# In[153]:


#3.3 Find the top 10 complaint types NewYork
Complaint_frequency_NewYork.head(10)


# ### 3.4 Display the various types of complaints in each city

# In[154]:


compliant_by_city = df_311_CSRA.groupby('City')['Complaint Type'].value_counts()
compliant_by_city.head(50)


# ### 3.5 Create a DataFrame, df_new, which contains cities as columns and complaint types in rows

# In[155]:


df_new = df_311_CSRA.pivot_table(index='Complaint Type',columns='City', aggfunc='size',fill_value=0)
df_new


# ## 4. Visualize the major types of complaints in each city
# 
# ### 4.1 Draw another chart that shows the types of complaints in each city in a single chart, where different colors show the different types of complaints

# In[156]:


Major_type_City_Complaints = df_311_CSRA.groupby(["City","Complaint Type"]).size().unstack()
plt.figure(figsize=(12,8))
Major_type_City_Complaints.plot(kind='bar',stacked=True,colormap='tab20')
plt.title('Major Complaints in city')
plt.xlabel("city")
plt.ylabel('Number of Complaints')
plt.legend(title='Complaint type', bbox_to_anchor=(1,1))
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()


# ### 4.2 Sort the complaint types based on the average Request_Closing_Time grouping them for different locations

# In[157]:


df_311_CSRA['Closed Date']=pd.to_datetime(df_311_CSRA['Closed Date'])
df_311_CSRA['Created Date']=pd.to_datetime(df_311_CSRA['Created Date'])
df_311_CSRA['Closed_Time_In_Seconds'] = (df_311_CSRA['Closed Date']-df_311_CSRA['Created Date'])


# In[158]:


Avrge_CloseTime_of_Complaint=df_311_CSRA['Closed_Time_In_Seconds'].mean()


# In[159]:


#sorted_avg_tm=Avrge_CloseTime_of_Complaint.sort_values()
print(f"Average_closing_time:{Avrge_CloseTime_of_Complaint}")


# In[160]:


Avrge_CloseTime_of_Complaint_time_location=df_311_CSRA.groupby(['City',"Complaint Type"])['Created Date'].mean()


# In[161]:


sorted_data=Avrge_CloseTime_of_Complaint_time_location.sort_values()
print(f'complaint types based on the average Request_Closing_Time:')
print(sorted_data)


# ### 5. See whether the average response time across different complaint types is similar (overall)

# In[162]:


print(Avrge_CloseTime_of_Complaint_time_location)


# ### 5.1 Visualize the average of Request_Closing_Time

# In[163]:


plt.figure(figsize=(12,6))
sorted_data.plot(kind='bar')
plt.xlabel('Complaint Type')
plt.ylabel('Request_Closing_time')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ### significant variables by performing statistical analysis using p-values

# In[173]:


#creating the contengency table
contingency_table = pd.crosstab(df_311_CSRA["Complaint Type"],df_311_CSRA["City"] )

#chi_square test

ch2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

print(f'chi-squared statistic:{ch2_stat}')
print(f'P-Value:{p_val}')


# ### Perform a Kruskal-Wallis H test

# In[192]:


#City_unique=df_311_CSRA["City"]
#data_groups = df_311_CSRA.groupby(['City',"Complaint Type"])['Created Date'] 
boroughs = df_311_CSRA['Borough'].unique()
data_groups = [df_311_CSRA[df_311_CSRA['Borough'] == Borough]['Created Date'] for Borough in boroughs]

#performing kruskal wallis test

h_stat, p_val = kruskal(*data_groups)

# Set significance level
alpha = 0.05

# Interpret the result
if p_val >= alpha:
    print("Fail to Reject H0: All sample distributions are equal.")
else:
    print("Reject H0: One or more sample distributions are not equal.")


# In[ ]:




