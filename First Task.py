#!/usr/bin/env python
# coding: utf-8

# # Prediction Using Supervised ML

# # Linear regression with Python Scikit Learn
# I am going to use Python Scikit Learn for supervised machine learning to implement linear regression for the given data. We will start with simple linear regression involving two variables.
# 
# 

# # Simple Linear Regression
# In this task we will predict the percentage of marks that a student is expected to score based upon the no of hours they studied.Lets do it.

# In[1]:


#Importing all libraries necessary for the task
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#I created an excelsheet in .csv format to import to jupyter notebook.I can easily access the data
print("\n Data is imported successfully")
data=pd.read_csv("C:\\Users\\jannu yamini\\Desktop\\linear regression\data.csv")
data


# In[3]:


#I have plotted the given data onto a simple graph to visualize the regression between the values
data.plot(x="Hours",y="Scores",style="o")
plt.title("Hours vs Percentage")
plt.xlabel("Hours studied")
plt.ylabel("Percentage Score")
plt.show()


# # From the above graph, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score. We can see the "percentage score" going up significantly. It means that, the more number of hours the student spends in studying, the more the chances of aquiring a high percentage.

# # Preparing the data
# The next step is to divide the data into "attributes"and "labels". These are going to be input and output sources.

# In[4]:


x=data.iloc[:, :-1].values
y=data.iloc[:, 1].values


# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
                            test_size=0.2,random_state=0)


# # Training the algorithm
# I have split the data into two sets i.e. Training sets and testing sets. And now the algorithm should be trained.

# In[6]:


from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print ("The algorithm is trained and ready!")


# # Let's plot the regression line for the data.
# 

# In[7]:


line = regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# # Making the prediction
# Now that I have the algorithm perfectly trained, Let's make some predictions!

# In[8]:


print(x_test)
y_pred = regressor.predict(x_test)


# In[9]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # You can test your own data from now. The algorithm gives the predicted value.

# In[10]:


hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Evaluation
# In the final step,we evaluate the performance of the algorithm. Out of many statistical metrics, I am choosing mean square error here.

# In[11]:


from sklearn import metrics
print("Mean Absolute Error: ",metrics.mean_absolute_error(y_test,y_pred))


# # Thank You
# I'd like to sincerely thank The Sparks Foundation for this great opportunity to perform my first ever hands-on work on this dataset.Thank you so much!

# In[ ]:




