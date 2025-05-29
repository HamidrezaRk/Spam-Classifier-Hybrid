#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier


# In[85]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"


# In[86]:


column_names = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
    'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
    'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
    'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
    'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
    'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
    'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
    'capital_run_length_longest', 'capital_run_length_total', 'is_spam'
]


# In[87]:


df = pd.read_csv(url, header=None, names=column_names)


# In[88]:


df


# In[89]:


df = pd.read_csv(url, header=None, names=column_names)


# In[90]:


x = df.drop('is_spam', axis=1)
y = df['is_spam']


# In[91]:


x


# In[92]:


y


# In[93]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[94]:


x_train


# In[95]:


dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train,y_train)
y_pred_dt = dt.predict(x_test)


# In[96]:


accuracy_score(y_test,y_pred_dt)


# In[97]:


rf = RandomForestClassifier(n_estimators=100 ,random_state=42)
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)


# In[98]:


accuracy_score(y_test,y_pred_rf)


# In[99]:


voting_model = VotingClassifier(estimators=[('Dt', dt), ('rf', rf)], voting='hard')
voting_model.fit(x_train, y_train)
voting_pred = voting_model.predict(x_test)
accuracy_score(y_test, voting_pred)


# In[ ]:




