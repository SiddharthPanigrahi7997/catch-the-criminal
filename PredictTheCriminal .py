
# coding: utf-8

# In[1]:

import pandas
import numpy
import matplotlib.pyplot as plt


# In[2]:

df=pandas.read_csv('criminal_train_pre.csv')
df=df.dropna()


# In[3]:

#df.dtypes
df=df.drop(['PERID'],axis=1)


# In[4]:

numcols=[]
catcols=[]
for g in df.columns:
    if max(df[g])>=10:
        numcols.append(g)
    else:
        catcols.append(g)


# In[5]:

#deleting criminal from it
del catcols[len(catcols)-1]


# In[6]:

#converting to float,int in train
for g in df.columns:
    if g in numcols:
        df[g]=df[g].astype('float64')
    else:
        df[g]=df[g].astype('int64')


# In[7]:

#adding dummies
for m in catcols:
    dummies = pandas.get_dummies(df[m],prefix=m)
    df = df.join(dummies)
    del df[m]


# In[8]:

testdf=pandas.read_csv('criminal_test.csv')
print testdf.shape

ids=testdf['PERID'].values


# In[9]:

testdf=testdf.drop(['PERID'],axis=1)


# In[10]:

print testdf.shape

for g in testdf.columns:
    if g in catcols:
        testdf[g]=testdf[g].replace([-1],testdf[g].value_counts().index[0])
    else:
        testdf[g]=testdf[g].replace([-1],sum(testdf[g])/len(testdf[g]))
        


# In[11]:

for m in catcols:
    dummies = pandas.get_dummies(testdf[m],prefix=m)
    testdf = testdf.join(dummies)
    del testdf[m]


# In[12]:

k=set(testdf.columns)-set(df.columns)
for l in k:
    df[l]=0


# In[13]:

for g in testdf.columns:
    if g in numcols:
        testdf[g]=testdf[g].astype('float64')
    else:
        testdf[g]=testdf[g].astype('int64')


# In[14]:

#standardize rest of them
stddf   = df.copy()
stddf[numcols] = stddf[numcols].apply(lambda x: (x-x.mean())/(x.max()-x.min()))


# In[15]:

teststd   = testdf.copy()
teststd[numcols] = teststd[numcols].apply(lambda x: (x-x.mean())/(x.max()-x.min()))


# In[16]:

labels=df['Criminal'].values
print labels[:5],type(labels)


# In[17]:

data=df.drop(['Criminal'],axis=1).values
print data


# In[18]:

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data,labels, test_size=0.15, random_state=42)
print "x_train:",x_train.shape
print "y_train:",y_train.shape
print
print "x_test:",x_test.shape
print "y_test:",y_test.shape


# In[19]:

from sklearn import svm
clf = svm.SVC(C=5)
clf.fit(x_train,y_train)
print "SVM:",clf.score(x_test,y_test)*100,"%"


# In[20]:

pr=clf.predict(teststd.values)


# In[21]:

print "hello"


# In[22]:

print len(pr)


# In[23]:

print len(ids)


# In[29]:

import csv
datae=[] 
datae.append(['PERID','Criminal'])
for i in range(len(pr)):
    datae.append([ids[i],pr[i]])

myFile = open('output.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(datae)
#print datae     
print "Writing complete"

