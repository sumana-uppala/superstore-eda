#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df = pd.read_csv("/Users/supriyauppala/Downloads/SampleSuperstore.csv")


# In[7]:


df.info()


# In[8]:


for col in df.columns: 
    print(col) 


# In[9]:


ShipMode = df['Ship Mode'].tolist()
ShipModeSet=set(ShipMode)
print(ShipModeSet)


# From the above, it can be understood that there are a total of four Ship Modes : (1) 'Second Class' (2) 'Standard Class' (3) 'Same Day' (4) 'First Class'

# In[10]:


Segment = df['Segment'].tolist()
SegmentSet=set(Segment)
print(SegmentSet)


# From the above, it can be understood that there are a total of three segments : (1) 'Home Office' (2) 'Consumer' (3) 'Corporate'

# In[11]:


Country = df['Country'].tolist()
CountrySet=set(Country)
print(CountrySet)


# From the above, there is only one category in country i.e, United States

# In[12]:


City = df['City'].tolist()
CitySet=set(City)
print("There are a total of ",len(CitySet)," cities")


# In[13]:


State = df['State'].tolist()
StateSet=set(State)
print(StateSet)
print(len(StateSet))


# In[14]:


Region = df['Region'].tolist()
RegionSet=set(Region)
print(RegionSet)


# In[15]:


Category = df['Category'].tolist()
CategorySet=set(Category)
print(CategorySet)


# In[16]:


SubCategory = df['Sub-Category'].tolist()
SubCategorySet=set(SubCategory)
print(SubCategorySet)
print(len(SubCategorySet))


# In[17]:


Sales = df['Sales'].tolist()
Quantity = df['Quantity'].tolist()
Discount = df['Discount'].tolist()
Profit = df['Profit'].tolist()


# In[18]:


print("Range of Sales")
print(min(Sales))
print(max(Sales))
print(" ")
print("Range of Discount")
print(min(Discount))
print(max(Discount))
print(" ")
print("Range of Profit")
print(min(Profit))
print(max(Profit))


# In[19]:


SetDiscount=set(Discount)
print(SetDiscount)


# In[21]:


plt.figure(figsize = (20,8))

plt.subplot(1,3,1)
plt.scatter(df['Discount'],df['Profit'])
plt.title('Discount vs Profit')
plt.xlabel('Discount')
plt.ylabel('Profit')

plt.subplot(1,3,2)
plt.scatter(df['Profit'],df['Sales'])
plt.title('Profit vs Sales')
plt.xlabel('Profit')
plt.ylabel('Sales')

plt.subplot(1,3,3)
plt.scatter(df['Sales'],df['Discount'])
plt.title('Sales vs Discount')
plt.xlabel('Sales')
plt.ylabel('Discount')

plt.show()


# In[22]:


df_gender = df.groupby('Segment')
x = df_gender['Segment'].count().keys()


height = df_gender['Sales'].mean()
plt.bar(x = x, 
         height = height,color = 'green')
plt.title('Average Sales')
plt.xlabel('Segment')
plt.ylabel('Average Sales')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()

height = df_gender['Discount'].mean()
plt.bar(x = x, 
         height = height,color='blue')
plt.title('Average Discount')
plt.xlabel('Segment')
plt.ylabel('Average Discount')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()

height = df_gender['Profit'].mean()
plt.bar(x = x, 
         height = height,color='red')
plt.title('Average Profit')
plt.xlabel('Segment')
plt.ylabel('Average profit')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha = 'center', fontweight='bold')
plt.show()


# In[42]:


df_gender = df.groupby('Ship Mode')
x = df_gender['Ship Mode'].count().keys()


height = df_gender['Sales'].mean()
plt.bar(x = x, 
         height = height,color = 'pink')
plt.title('Average Sales')
plt.xlabel('Ship Mode')
plt.ylabel('Average Sales')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()

height = df_gender['Discount'].mean()
plt.bar(x = x, 
         height = height,color='purple')
plt.title('Average Discount')
plt.xlabel('Ship Mode')
plt.ylabel('Average Discount')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()

height = df_gender['Profit'].mean()
plt.bar(x = x, 
         height = height,color='violet')
plt.title('Average Profit')
plt.xlabel('Ship Mode')
plt.ylabel('Average profit')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha = 'center', fontweight='bold')
plt.show()


# In[43]:


df_gender = df.groupby('Region')
x = df_gender['Region'].count().keys()


height = df_gender['Sales'].mean()
plt.bar(x = x, 
         height = height,color = 'yellow')
plt.title('Average Sales')
plt.xlabel('Region')
plt.ylabel('Average Sales')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()

height = df_gender['Discount'].mean()
plt.bar(x = x, 
         height = height,color='indigo')
plt.title('Average Discount')
plt.xlabel('Region')
plt.ylabel('Average Discount')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()

height = df_gender['Profit'].mean()
plt.bar(x = x, 
         height = height,color='orange')
plt.title('Average Profit')
plt.xlabel('Region')
plt.ylabel('Average profit')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha = 'center', fontweight='bold')
plt.show()


# In[54]:


df_gender = df.groupby('Category')
x = df_gender['Category'].count().keys()


height = df_gender['Sales'].mean()
plt.bar(x = x, 
         height = height,color = 'lavender')
plt.title('Average Sales')
plt.xlabel('Category')
plt.ylabel('Average Sales')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()

height = df_gender['Discount'].mean()
plt.bar(x = x, 
         height = height,color='turquoise')
plt.title('Average Discount')
plt.xlabel('Category')
plt.ylabel('Average Discount')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()

height = df_gender['Profit'].mean()
plt.bar(x = x, 
         height = height,color='grey')
plt.title('Average Profit')
plt.xlabel('Category')
plt.ylabel('Average profit')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha = 'center', fontweight='bold')
plt.show()


# In[60]:


def barp_xyz(x, y, hue):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.barplot(x, y, hue=hue, data=df)
for i in ['Sales', 'Discount', 'Profit']:
    barp_xyz('Ship Mode' ,i,'Segment')


# In[61]:


sns.set(style="whitegrid")
def barp_xyz(x, y, hue):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.barplot(x, y, hue=hue, data=df, color='red')
for i in ['Sales', 'Discount', 'Profit']:
    barp_xyz('Segment' ,i,'Category')


# In[62]:


sns.set(style="whitegrid")
def barp_xyz(x, y, hue):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.barplot(x, y, hue=hue, data=df, color='blue')
for i in ['Sales', 'Discount', 'Profit']:
    barp_xyz('Region' ,i,'Category')


# In[63]:


sns.set(style="whitegrid")
def barp_xyz(x, y, hue):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.barplot(x, y, hue=hue, data=df, color='green')
for i in ['Sales', 'Discount', 'Profit']:
    barp_xyz('Region' ,i,'Ship Mode')


# In[64]:


sns.set(style="whitegrid")
def barp_xyz(x, y, hue):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.barplot(x, y, hue=hue, data=df, color='pink')
for i in ['Sales', 'Discount', 'Profit']:
    barp_xyz('Ship Mode' ,i,'Category')


# In[66]:


sns.set(style="whitegrid")
def barp_xyz(x, y, hue):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.barplot(x, y, hue=hue, data=df, color='purple')
for i in ['Sales', 'Discount', 'Profit']:
    barp_xyz('Segment' ,i,'Region')


# In[25]:


def get_grade(scores):
    if scores == 0.0:
        return '0.0'
    elif scores > 0.0 and scores < 0.3 :
        return '0.0 - 0.3'
    elif scores >= 0.3 and scores < 0.6 :
        return '0.3 - 0.6'
    elif scores >= 0.6 and scores < 0.9 :
        return '0.6 - 0.9'


# In[26]:


df['Discount_grades'] = df['Discount'].apply(get_grade)
plt.title("Pie chart of Discounts",fontsize = 15)
df['Discount_grades'].value_counts().plot.pie(autopct ="%1.1f%%")
plt.show()


# In[27]:


plt.title("Category vs Segment",fontsize = 15)
sns.countplot(x="Segment", hue="Category", data=df)
plt.show()

plt.title("Category vs Region",fontsize = 15)
sns.countplot(x="Region", hue="Category", data=df)
plt.show()

plt.title("Category vs Ship Mode",fontsize = 15)
sns.countplot(x="Ship Mode", hue="Category", data=df)
plt.show()


# In[28]:


pivot = pd.pivot_table(data = df, index = ["Region"], columns = ["Category"], aggfunc = {'Profit' : np.mean})
hm = sns.heatmap(data = pivot, annot = True, cmap = "Greens")
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[29]:


pivot = pd.pivot_table(data = df, index = ["Region"], columns = ["Category"], aggfunc = {'Discount' : np.mean})
hm = sns.heatmap(data = pivot, annot = True, cmap = "Reds")
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[30]:


pivot = pd.pivot_table(data = df, index = ["Region"], columns = ["Category"], aggfunc = {'Sales' : np.mean})
hm = sns.heatmap(data = pivot, annot = True, cmap = "Blues")
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[31]:


sale_category = df.groupby(["Category","Sub-Category"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False)
sale_category
sns.barplot(x = "Category",     # Data is groupedby this variable
            hue="Sub-Category",
            y= "Quantity",          
            data=sale_category)


# In[32]:


regionwiseSalesAndProfit = df.groupby("Region").agg({"Sales":np.sum, "Profit": np.sum})
regionwiseSalesAndProfit
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
p = sns.scatterplot(x="Sales", y="Profit", hue=regionwiseSalesAndProfit.index, data=regionwiseSalesAndProfit) # kind="scatter")
ax.set_title("Relationship between Sales and Profit by Region")
plt.tight_layout()
plt.show()


# In[33]:


data=df[['Sales','Quantity','Discount','Profit']]
sns.heatmap(data.corr(),annot=True)


# In[34]:


df['Sub-Category'].value_counts().plot(kind="bar")


# In[35]:


plt.figure(figsize=(16,8))
top20city = df.groupby('City')['State'].count().sort_values(ascending=False)
top20city = top20city [:15]
top20city.plot(kind='bar')
plt.title('Top 15 Cities in Sales')
plt.ylabel('Count')
plt.xlabel('Cities')
plt.show()


# In[71]:


plt.figure(figsize=(16,8))
top20state = df.groupby('State')['Country'].count().sort_values(ascending=False)
top20state = top20state [:15]
top20state.plot(kind='bar',color='green')
plt.title('Top 15 States in Sales')
plt.ylabel('Count')
plt.xlabel('States')
plt.show()


# In[36]:


df['Category'].value_counts()

sns.boxplot(
            "Category",
            "Profit",
             data= df
             )


# In[37]:


df['Category'].value_counts()

sns.boxplot(
            "Category",
            "Sales",
             data= df
             )


# In[38]:


df['Category'].value_counts()

sns.boxplot(
            "Category",
            "Discount",
             data= df
             )


# In[39]:


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
p = sns.countplot(x="Segment", data=df, ax=ax)
ax.set_title("Customer Distribution by Segment")
ax.set_xticklabels(p.get_xticklabels(), rotation=90)
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.tight_layout()
plt.show()


# In[72]:


df.describe()


# In[ ]:




