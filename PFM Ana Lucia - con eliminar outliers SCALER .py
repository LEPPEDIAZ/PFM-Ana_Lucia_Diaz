#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Intel optimzation
from sklearnex import patch_sklearn
patch_sklearn("SVC")


# In[2]:


import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }


# In[5]:


pd.set_option('max_columns', None)
train = pd.read_csv('train.csv', dtype=dtypes, low_memory=True, nrows=1000000)
train


# In[6]:


train.describe()


# In[7]:


print(train.info())


# In[8]:


train['HasDetections'].unique()


# In[9]:


train.isna().any()[lambda x: x]


# # REMOVE NULL VALUES

# Fill null values in float values 

# In[10]:


train = train.replace(np.nan,0)


# In[11]:


train.isna().any()[lambda x: x]


# ## MAKE NEW INDEX FOR DATAFRAME

# In[12]:


train.set_index('MachineIdentifier')


# # Encoding
# Better coding leads to a better model and most algorithms cannot handle categorical variables unless they are cast to a numeric value.
# Categorical features are generally divided into 3 types:
#  
# 
# Binary: Either / or
# 
#     Examples:
#     If not
#     True False
# 
# 
# Ordinal: Specific ordered groups.
# 
#     Examples:
#     low medium high
#     cold, hot, hot lava
# 
# 
# Nominal: Unordered groups.
# 
#     Examples of
#     cat, dog, tiger
#     pizza, burger, coke
# ![alt text](https://miro.medium.com/max/1250/0*NBVi7M3sGyiUSyd5.png "encoding")

# ## ProductName 

# In[13]:


train['ProductName'].unique()


# # Weight of Evidence Encoding 
# Weight of Evidence (WoE) measures the “strength” of a grouping technique to separate good and bad. Weight of evidence (WOE) measures how much the evidence supports or undermines a hypothesis.
# 
# ![alt text](https://miro.medium.com/max/281/1*AqcqDwUB4fk8rcmbvxGiEQ.gif "Title")

# In[14]:


pr_df = train.groupby('ProductName')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']=(pr_df.Good/pr_df.Bad)
pr_df


# In[15]:


train.loc[:, 'ProductName'] = train['ProductName'].map(pr_df['PR'])
train


# In[16]:


train['ProductName'].unique()


# ## EngineVersion

# In[17]:


train['EngineVersion'].unique()


# ## Leave One Out Encoder
# 
# Leave One Out encoding essentially calculates the mean of the target variables for all records that contain the same value for the categorical feature variable in question.

# In[18]:


import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce


# In[19]:


ce_leave = ce.LeaveOneOutEncoder(cols = ['EngineVersion'])
y, X = train['HasDetections'], train['EngineVersion']
ce_leave.fit(X, y)        
ce_leave.transform(X, y)
result = ce_leave.transform(X, y)
result


# In[20]:


train = pd.merge(train, result, left_index=True, right_index=True)
train.head(10)


# In[21]:


train = train.rename(columns={"EngineVersion_y": "EngineVersion"})
train


# In[22]:


train.sort_values(by='EngineVersion', ascending=False)


# In[23]:


train =  train.drop(['EngineVersion_x'], axis=1)
train


# ## AppVersion

# In[24]:


import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce


# In[25]:


train['AppVersion'].unique()


# In[26]:


ce_leave = ce.LeaveOneOutEncoder(cols = ['AppVersion'])
y, X = train['HasDetections'], train['AppVersion']
ce_leave.fit(X, y)        
ce_leave.transform(X, y)
result = ce_leave.transform(X, y)
result


# In[27]:


train = pd.merge(train, result, left_index=True, right_index=True)
train.head(10)


# In[28]:


train = train.rename(columns={"AppVersion_y": "AppVersion"})
train


# In[29]:


train =  train.drop(['AppVersion_x'], axis=1)
train


# ## AvSigVersion

# In[30]:


import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce


# In[31]:


train['AvSigVersion'].unique()


# In[32]:


ce_leave = ce.LeaveOneOutEncoder(cols = ['AvSigVersion'])
y, X = train['HasDetections'], train['AvSigVersion']
ce_leave.fit(X, y)        
ce_leave.transform(X, y)
result = ce_leave.transform(X, y)
result


# In[33]:


train = pd.merge(train, result, left_index=True, right_index=True)
train.head(10)


# In[34]:


train = train.rename(columns={"AvSigVersion_y": "AvSigVersion"})
train


# In[35]:


train =  train.drop(['AvSigVersion_x'], axis=1)
train


# ## Platform

# In[36]:


train['Platform'].unique()


# In[37]:


pr_df = train.groupby('Platform')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']=(pr_df.Good/pr_df.Bad)
pr_df


# In[38]:


train.loc[:, 'Platform'] = train['Platform'].map(pr_df['PR'])
train


# ## Processor

# In[39]:


train['Processor'].unique()


# In[40]:


pr_df = train.groupby('Processor')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[41]:


train.loc[:, 'Processor'] = train['Processor'].map(pr_df['PR'])
train


# ## OsVer

# In[42]:


train['OsVer'].unique()


# In[43]:


ce_leave = ce.LeaveOneOutEncoder(cols = ['OsVer'])
y, X = train['HasDetections'], train['OsVer']
ce_leave.fit(X, y)        
ce_leave.transform(X, y)
result = ce_leave.transform(X, y)
result


# In[44]:


train = pd.merge(train, result, left_index=True, right_index=True)
train.head(10)


# In[45]:


train = train.rename(columns={"OsVer_y": "OsVer"})
train


# In[46]:


train.sort_values(by='OsVer', ascending=False)


# In[47]:


train =  train.drop(['OsVer_x'], axis=1)
train


# ## OsPlatformSubRelease

# In[48]:


train['OsPlatformSubRelease'].unique()


# In[49]:


pr_df = train.groupby('OsPlatformSubRelease')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[50]:


train.loc[:, 'OsPlatformSubRelease'] = train['OsPlatformSubRelease'].map(pr_df['PR'])
train


# ## OsBuildLab

# In[51]:


train['OsBuildLab'].unique()


# In[52]:


ce_leave = ce.LeaveOneOutEncoder(cols = ['OsBuildLab'])
y, X = train['HasDetections'], train['OsBuildLab']
ce_leave.fit(X, y)        
ce_leave.transform(X, y)
result = ce_leave.transform(X, y)
result


# In[53]:


train = pd.merge(train, result, left_index=True, right_index=True)
train.head(10)


# In[54]:


train = train.rename(columns={"OsBuildLab_y": "OsBuildLab"})
train


# In[55]:


train.sort_values(by='OsBuildLab', ascending=False)


# In[56]:


train.sort_values(by='OsBuildLab', ascending=True)


# In[57]:


train =  train.drop(['OsBuildLab_x'], axis=1)
train


# ### SkuEdition

# In[58]:


train['SkuEdition'].unique()


# In[59]:


pr_df = train.groupby('SkuEdition')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[60]:


train.loc[:, 'SkuEdition'] = train['SkuEdition'].map(pr_df['PR'])
train


# ## PuaMode

# In[61]:


train['PuaMode'].unique()


# In[62]:


pr_df = train.groupby('PuaMode')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[63]:


train.loc[:, 'PuaMode'] = train['PuaMode'].map(pr_df['PR'])
train


# ## SmartScreen

# In[64]:


train['SmartScreen'].unique()


# In[65]:


ce_leave = ce.LeaveOneOutEncoder(cols = ['SmartScreen'])
y, X = train['HasDetections'], train['SmartScreen']
ce_leave.fit(X, y)        
ce_leave.transform(X, y)
result = ce_leave.transform(X, y)
result


# In[66]:


train = pd.merge(train, result, left_index=True, right_index=True)
train.head(10)


# In[67]:


train = train.rename(columns={"SmartScreen_y": "SmartScreen"})
train


# In[68]:


train.sort_values(by='SmartScreen', ascending=False)


# In[69]:


train.sort_values(by='SmartScreen', ascending=True)


# In[70]:


train =  train.drop(['SmartScreen_x'], axis=1)
train


# ## Census_MDC2FormFactor

# In[71]:


train['Census_MDC2FormFactor'].unique()


# In[72]:


ce_leave = ce.LeaveOneOutEncoder(cols = ['Census_MDC2FormFactor'])
y, X = train['HasDetections'], train['Census_MDC2FormFactor']
ce_leave.fit(X, y)        
ce_leave.transform(X, y)
result = ce_leave.transform(X, y)
result


# In[73]:


train = pd.merge(train, result, left_index=True, right_index=True)
train.head(10)


# In[74]:


train = train.rename(columns={"Census_MDC2FormFactor_y": "Census_MDC2FormFactor"})
train


# In[75]:


train.sort_values(by='Census_MDC2FormFactor', ascending=False)


# In[76]:


train.sort_values(by='Census_MDC2FormFactor', ascending=False)


# In[77]:


train.sort_values(by='Census_MDC2FormFactor', ascending=True)


# In[78]:


train =  train.drop(['Census_MDC2FormFactor_x'], axis=1)
train


# ## Census_DeviceFamily

# In[79]:


train['Census_DeviceFamily'].unique()


# In[80]:


pr_df = train.groupby('Census_DeviceFamily')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[81]:


train.loc[:, 'Census_DeviceFamily'] = train['Census_DeviceFamily'].map(pr_df['PR'])
train


# ## Census_ProcessorClass

# In[82]:


train['Census_ProcessorClass'].unique()


# In[83]:


pr_df = train.groupby('Census_ProcessorClass')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[84]:


train.loc[:, 'Census_ProcessorClass'] = train['Census_ProcessorClass'].map(pr_df['PR'])
train


# In[85]:


train['Census_ProcessorClass'].unique()


# ## Census_PrimaryDiskTypeName

# In[86]:


train['Census_PrimaryDiskTypeName'].unique()


# In[87]:


pr_df = train.groupby('Census_PrimaryDiskTypeName')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[88]:


train.loc[:, 'Census_PrimaryDiskTypeName'] = train['Census_PrimaryDiskTypeName'].map(pr_df['PR'])
train


# In[89]:


train['Census_PrimaryDiskTypeName'].unique()


# ## Census_ChassisTypeName

# In[90]:


train['Census_ChassisTypeName'].unique()


# In[91]:


ce_leave = ce.LeaveOneOutEncoder(cols = ['Census_ChassisTypeName'])
y, X = train['HasDetections'], train['Census_ChassisTypeName']
ce_leave.fit(X, y)        
ce_leave.transform(X, y)
result = ce_leave.transform(X, y)
result


# In[92]:


train = pd.merge(train, result, left_index=True, right_index=True)
train.head(10)


# In[93]:


train = train.rename(columns={"Census_ChassisTypeName_y": "Census_ChassisTypeName"})
train


# In[94]:


train =  train.drop(['Census_ChassisTypeName_x'], axis=1)
train


# ## Census_PowerPlatformRoleName 

# In[95]:


train['Census_PowerPlatformRoleName'].unique()


# In[96]:


pr_df = train.groupby('Census_PowerPlatformRoleName')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[97]:


train.loc[:, 'Census_PowerPlatformRoleName'] = train['Census_PowerPlatformRoleName'].map(pr_df['PR'])
train


# In[98]:


train['Census_PowerPlatformRoleName'].unique()


# ## Census_InternalBatteryType

# In[99]:


train['Census_InternalBatteryType'].unique()


# In[100]:


pr_df = train.groupby('Census_InternalBatteryType')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[101]:


train.loc[:, 'Census_InternalBatteryType'] = train['Census_InternalBatteryType'].map(pr_df['PR'])
train


# In[102]:


train['Census_InternalBatteryType'].unique()


# ## Census_OSVersion

# In[103]:


train['Census_OSVersion'].unique()


# In[104]:


ce_leave = ce.LeaveOneOutEncoder(cols = ['Census_OSVersion'])
y, X = train['HasDetections'], train['Census_OSVersion']
ce_leave.fit(X, y)        
ce_leave.transform(X, y)
result = ce_leave.transform(X, y)
result


# In[105]:


train = pd.merge(train, result, left_index=True, right_index=True)
train.head(10)


# In[106]:


train = train.rename(columns={"Census_OSVersion_y": "Census_OSVersion"})
train


# In[107]:


train =  train.drop(['Census_OSVersion_x'], axis=1)
train


# In[108]:


train['Census_OSVersion'].unique()


# ## Census_OSArchitecture

# In[109]:


train['Census_OSArchitecture'].unique()


# In[110]:


pr_df = train.groupby('Census_OSArchitecture')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[111]:


train.loc[:, 'Census_OSArchitecture'] = train['Census_OSArchitecture'].map(pr_df['PR'])
train


# In[112]:


train['Census_OSArchitecture'].unique()


# ## Census_OSBranch

# In[113]:


train['Census_OSBranch'].unique()


# In[114]:


pr_df = train.groupby('Census_OSBranch')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[115]:


train.loc[:, 'Census_OSBranch'] = train['Census_OSBranch'].map(pr_df['PR'])
train


# In[116]:


train['Census_OSBranch'].unique()


# ### Census_OSEdition

# In[117]:


train['Census_OSEdition'].unique()


# In[118]:


pr_df = train.groupby('Census_OSEdition')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[119]:


train.loc[:, 'Census_OSEdition'] = train['Census_OSEdition'].map(pr_df['PR'])
train


# In[120]:


train['Census_OSEdition'].unique()


# ### Census_OSSkuName 

# In[121]:


train['Census_OSSkuName'].unique()


# In[122]:


pr_df = train.groupby('Census_OSSkuName')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[123]:


train.loc[:, 'Census_OSSkuName'] = train['Census_OSSkuName'].map(pr_df['PR'])
train


# In[124]:


train['Census_OSSkuName'].unique()


# ## Census_OSInstallTypeName

# In[125]:


train['Census_OSInstallTypeName'].unique()


# In[126]:


pr_df = train.groupby('Census_OSInstallTypeName')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[127]:


train.loc[:, 'Census_OSInstallTypeName'] = train['Census_OSInstallTypeName'].map(pr_df['PR'])
train


# In[128]:


train['Census_OSInstallTypeName'].unique()


# ## Census_OSWUAutoUpdateOptionsName

# In[129]:


train['Census_OSWUAutoUpdateOptionsName'].unique()


# In[130]:


pr_df = train.groupby('Census_OSWUAutoUpdateOptionsName')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[131]:


train.loc[:, 'Census_OSWUAutoUpdateOptionsName'] = train['Census_OSWUAutoUpdateOptionsName'].map(pr_df['PR'])
train


# In[132]:


train['Census_OSWUAutoUpdateOptionsName'].unique()


# ## Census_GenuineStateName

# In[133]:


train['Census_GenuineStateName'].unique()


# In[134]:


pr_df = train.groupby('Census_GenuineStateName')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[135]:


train.loc[:, 'Census_GenuineStateName'] = train['Census_GenuineStateName'].map(pr_df['PR'])
train


# In[136]:


train['Census_GenuineStateName'].unique()


# ## Census_ActivationChannel 

# In[137]:


train['Census_ActivationChannel'].unique()


# In[138]:


pr_df = train.groupby('Census_ActivationChannel')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[139]:


train.loc[:, 'Census_ActivationChannel'] = train['Census_ActivationChannel'].map(pr_df['PR'])
train


# In[140]:


train['Census_ActivationChannel'].unique()


# ## Census_FlightRing

# In[141]:


train['Census_FlightRing'].unique()


# In[142]:


pr_df = train.groupby('Census_FlightRing')['HasDetections'].mean()
pr_df = pd.DataFrame(pr_df)
pr_df = pr_df.rename(columns = {'HasDetections': 'Good'})
pr_df['Bad']= 1 - pr_df.Good
pr_df['Bad']= np.where(pr_df['Bad']==0, 0.000001, pr_df['Bad'])
pr_df['PR']= (pr_df.Good/pr_df.Bad)
pr_df


# In[143]:


train.loc[:, 'Census_FlightRing'] = train['Census_FlightRing'].map(pr_df['PR'])
train


# In[144]:


train['Census_FlightRing'].unique()


# In[145]:


print(train.info())


# In[146]:


train['PuaMode'] = train['PuaMode'].astype(float).fillna(0)
train['Census_ProcessorClass'] = train['Census_ProcessorClass'].astype(float).fillna(0)
train['Census_PrimaryDiskTypeName'] = train['Census_PrimaryDiskTypeName'].astype(float).fillna(0)
train['Census_PowerPlatformRoleName'] = train['Census_PowerPlatformRoleName'].astype(float).fillna(0)
train['Census_InternalBatteryType'] = train['Census_InternalBatteryType'].astype(float).fillna(0)
train['Census_IsWIMBootEnabled'] = train['Census_IsWIMBootEnabled'].astype(float).fillna(0)
print(train['PuaMode'].isnull().values.any())
print(train['Census_ProcessorClass'].isnull().values.any())
print(train['Census_PrimaryDiskTypeName'].isnull().values.any())
print(train['Census_PowerPlatformRoleName'].isnull().values.any())
print(train['Census_InternalBatteryType'].isnull().values.any())
print(train['Census_IsWIMBootEnabled'].isnull().values.any())


# ## Principales contrastes de hipótesis
# 1. bondad de fi -> verificar normalidad
# 2. contraste de hipótesis para contrastes de medias-paramétricas (hay normalidad) -> las varianzas en ambas poblaciones no difieren significativamente
# 3. Pruebas de hipótesis no paramétricas: (no hay normalidad)
# 4. Pruebas de correlación
# 5. Contrastes de ANOVA -> evalúa la importancia de uno o más factores (variables dependientes) comparando las medias con respecto a una variable en los diferentes niveles de los factores (cada uno de los valores de la variable indp.).

# # Normality test
# 
# ### Shapiro 
# La prueba de Shapiro-Wilk evalúa una muestra de datos y cuantifica la probabilidad de que los datos se hayan extraído de una distribución gaussiana. La estadística de prueba de Shapiro-Wilk (Calc W) es básicamente una medida de qué tan bien los cuantiles de muestra ordenados y estandarizados se ajustan a los cuantiles normales estándar. La estadística tomará un valor entre 0 y 1, siendo 1 una coincidencia perfecta.

# In[147]:


train.columns


# In[148]:


from scipy.stats import shapiro

#perform Shapiro-Wilk test for each variable and for general model
print("ProductName")
print(shapiro(train["ProductName"]))
print("IsBeta")
print(shapiro(train["IsBeta"]))
print("RtpStateBitfield")
print(shapiro(train["RtpStateBitfield"]))
print("IsSxsPassiveMode")
print(shapiro(train["IsSxsPassiveMode"]))
print("DefaultBrowsersIdentifier")
print(shapiro(train["DefaultBrowsersIdentifier"]))
print("AVProductStatesIdentifier")
print(shapiro(train["AVProductStatesIdentifier"]))
print("AVProductsInstalled")
print(shapiro(train["AVProductsInstalled"]))
print("AVProductsEnabled")
print(shapiro(train["AVProductsEnabled"]))
print("HasTpm")
print(shapiro(train["HasTpm"]))
print("CountryIdentifier")
print(shapiro(train["CountryIdentifier"]))
print("CityIdentifier")
print(shapiro(train["CityIdentifier"]))
print("OrganizationIdentifier")
print(shapiro(train["OrganizationIdentifier"]))
print("GeoNameIdentifier")
print(shapiro(train["GeoNameIdentifier"]))
print("LocaleEnglishNameIdentifier")
print(shapiro(train["LocaleEnglishNameIdentifier"]))
print("Platform")
print(shapiro(train["Platform"]))
print("Processor")
print(shapiro(train["Processor"]))
print("OsBuild")
print(shapiro(train["OsBuild"]))
print("OsSuite")
print(shapiro(train["OsSuite"]))
print("OsPlatformSubRelease")
print(shapiro(train["OsPlatformSubRelease"]))
print("SkuEdition")
print(shapiro(train["SkuEdition"]))
print("IsProtected")
print(shapiro(train["IsProtected"]))
print("AutoSampleOptIn")
print(shapiro(train["AutoSampleOptIn"]))
print("PuaMode")
print(shapiro(train["PuaMode"]))
print("SMode")
print(shapiro(train["SMode"]))
print("IeVerIdentifier")
print(shapiro(train["IeVerIdentifier"]))
print("Firewall")
print(shapiro(train["Firewall"]))
print("UacLuaenable")
print(shapiro(train["UacLuaenable"]))
print("Census_DeviceFamily")
print(shapiro(train["Census_DeviceFamily"]))
print("Census_OEMNameIdentifier")
print(shapiro(train["Census_OEMNameIdentifier"]))
print("Census_OEMModelIdentifier")
print(shapiro(train["Census_OEMModelIdentifier"]))
print("Census_ProcessorCoreCount")
print(shapiro(train["Census_ProcessorCoreCount"]))
print("Census_ProcessorManufacturerIdentifier")
print(shapiro(train["Census_ProcessorManufacturerIdentifier"]))
print("Census_ProcessorModelIdentifier")
print(shapiro(train["Census_ProcessorModelIdentifier"]))
print("Census_ProcessorClass")
print(shapiro(train["Census_ProcessorClass"]))
print("Census_PrimaryDiskTotalCapacity")
print(shapiro(train["Census_PrimaryDiskTotalCapacity"]))
print("Census_PrimaryDiskTypeName")
print(shapiro(train["Census_PrimaryDiskTypeName"]))              
print("Census_SystemVolumeTotalCapacity")                    
print(shapiro(train["Census_SystemVolumeTotalCapacity"]))
print("Census_HasOpticalDiskDrive")
print(shapiro(train["Census_HasOpticalDiskDrive"]))
print("Census_TotalPhysicalRAM")
print(shapiro(train["Census_TotalPhysicalRAM"]))
print("Census_InternalPrimaryDiagonalDisplaySizeInInches")
print(shapiro(train["Census_InternalPrimaryDiagonalDisplaySizeInInches"]))
print("Census_InternalPrimaryDisplayResolutionHorizontal")
print(shapiro(train["Census_InternalPrimaryDisplayResolutionHorizontal"]))
print("Census_InternalPrimaryDisplayResolutionVertical")
print(shapiro(train["Census_InternalPrimaryDisplayResolutionVertical"]))
print("Census_PowerPlatformRoleName")
print(shapiro(train["Census_PowerPlatformRoleName"]))
print("Census_InternalBatteryType")
print(shapiro(train["Census_InternalBatteryType"]))
print("Census_InternalBatteryNumberOfCharges")
print(shapiro(train["Census_InternalBatteryNumberOfCharges"]))
print("Census_OSArchitecture")
print(shapiro(train["Census_OSArchitecture"]))
print("Census_OSBranch")
print(shapiro(train["Census_OSBranch"]))               
print("Census_OSBuildNumber")
print(shapiro(train["Census_OSBuildNumber"]))
print("Census_OSBuildRevision")
print(shapiro(train["Census_OSBuildRevision"]))
print("Census_OSEdition")
print(shapiro(train["Census_OSEdition"]))
print("Census_OSSkuName")
print(shapiro(train["Census_OSSkuName"]))
print("Census_OSInstallTypeName")
print(shapiro(train["Census_OSInstallTypeName"]))
print("Census_OSInstallLanguageIdentifier")
print(shapiro(train["Census_OSInstallLanguageIdentifier"]))
print("Census_OSUILocaleIdentifier")
print(shapiro(train["Census_OSUILocaleIdentifier"]))
print("Census_OSWUAutoUpdateOptionsName")
print(shapiro(train["Census_OSWUAutoUpdateOptionsName"]))
print("Census_IsPortableOperatingSystem")
print(shapiro(train["Census_IsPortableOperatingSystem"]))
print("Census_GenuineStateName")
print(shapiro(train["Census_GenuineStateName"]))
print("Census_ActivationChannel")
print(shapiro(train["Census_ActivationChannel"]))
print("Census_IsFlightingInternal")
print(shapiro(train["Census_IsFlightingInternal"]))
print("Census_IsFlightsDisabled")
print(shapiro(train["Census_IsFlightsDisabled"]))
                    
print("Census_FlightRing")
print(shapiro(train["Census_FlightRing"]))
print("Census_ThresholdOptIn")
print(shapiro(train["Census_ThresholdOptIn"]))
print("Census_FirmwareManufacturerIdentifier")
print(shapiro(train["Census_FirmwareManufacturerIdentifier"]))
print("Census_FirmwareVersionIdentifier")
print(shapiro(train["Census_FirmwareVersionIdentifier"]))
print("Census_IsSecureBootEnabled")
print(shapiro(train["Census_IsSecureBootEnabled"]))
print("Census_IsWIMBootEnabled")
print(shapiro(train["Census_IsWIMBootEnabled"]))
print("Census_IsVirtualDevice")
print(shapiro(train["Census_IsVirtualDevice"]))
print("Census_IsTouchEnabled")
print(shapiro(train["Census_IsTouchEnabled"]))
print("Census_IsPenCapable")
print(shapiro(train["Census_IsPenCapable"]))
print("Census_IsAlwaysOnAlwaysConnectedCapable")
print(shapiro(train["Census_IsAlwaysOnAlwaysConnectedCapable"]))
print("Wdft_IsGamer")
print(shapiro(train["Wdft_IsGamer"]))
print("Wdft_RegionIdentifier")
print(shapiro(train["Wdft_RegionIdentifier"]))
print("HasDetections")
print(shapiro(train["HasDetections"]))
print("EngineVersion")
print(shapiro(train["EngineVersion"]))
print("AppVersion")
print(shapiro(train["AppVersion"]))
print("AvSigVersion")
print(shapiro(train["AvSigVersion"]))
print("OsVer")
print(shapiro(train["OsVer"]))
print("OsBuildLab")
print(shapiro(train["OsBuildLab"]))
print("SmartScreen")
print(shapiro(train["SmartScreen"]))
print("Census_MDC2FormFactor")
print(shapiro(train["Census_MDC2FormFactor"]))
print("Census_ChassisTypeName")
print(shapiro(train["Census_ChassisTypeName"]))
print("Census_OSVersion")
print(shapiro(train["Census_OSVersion"]))


# Una prueba de Shapiro-Wilk es la prueba para comprobar la normalidad de los datos. La hipótesis nula para la prueba de Shapiro-Wilk es que sus datos son normales, y si el valor p de la prueba es inferior a 0,05, entonces rechaza la hipótesis nula con un 5 % de significancia y concluye que sus datos no son normales.
# 
# Se puede rechazar la hipótesis nula. Entonces los datos no se distribuyen normalmente.

# In[149]:


train =  train.drop(['Census_IsWIMBootEnabled'], axis=1)
train


# # Contrastes de hipótesis no paramétricos: (no hay normalidad)

# In[150]:


y, X = train['HasDetections'], train[[ 'MachineIdentifier', 'ProductName', 'IsBeta', 'RtpStateBitfield',
       'IsSxsPassiveMode', 'DefaultBrowsersIdentifier',
       'AVProductStatesIdentifier', 'AVProductsInstalled', 'AVProductsEnabled',
       'HasTpm', 'CountryIdentifier', 'CityIdentifier',
       'OrganizationIdentifier', 'GeoNameIdentifier',
       'LocaleEnglishNameIdentifier', 'Platform', 'Processor', 'OsBuild',
       'OsSuite', 'OsPlatformSubRelease', 'SkuEdition', 'IsProtected',
       'AutoSampleOptIn', 'PuaMode', 'SMode', 'IeVerIdentifier', 'Firewall',
       'UacLuaenable', 'Census_DeviceFamily', 'Census_OEMNameIdentifier',
       'Census_OEMModelIdentifier', 'Census_ProcessorCoreCount',
       'Census_ProcessorManufacturerIdentifier',
       'Census_ProcessorModelIdentifier', 'Census_ProcessorClass',
       'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',
       'Census_SystemVolumeTotalCapacity', 'Census_HasOpticalDiskDrive',
       'Census_TotalPhysicalRAM',
       'Census_InternalPrimaryDiagonalDisplaySizeInInches',
       'Census_InternalPrimaryDisplayResolutionHorizontal',
       'Census_InternalPrimaryDisplayResolutionVertical',
       'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',
       'Census_InternalBatteryNumberOfCharges', 'Census_OSArchitecture',
       'Census_OSBranch', 'Census_OSBuildNumber', 'Census_OSBuildRevision',
       'Census_OSEdition', 'Census_OSSkuName', 'Census_OSInstallTypeName',
       'Census_OSInstallLanguageIdentifier', 'Census_OSUILocaleIdentifier',
       'Census_OSWUAutoUpdateOptionsName', 'Census_IsPortableOperatingSystem',
       'Census_GenuineStateName', 'Census_ActivationChannel',
       'Census_IsFlightingInternal', 'Census_IsFlightsDisabled',
       'Census_FlightRing', 'Census_ThresholdOptIn',
       'Census_FirmwareManufacturerIdentifier',
       'Census_FirmwareVersionIdentifier', 'Census_IsSecureBootEnabled',
       'Census_IsVirtualDevice', 'Census_IsTouchEnabled',
       'Census_IsPenCapable', 'Census_IsAlwaysOnAlwaysConnectedCapable',
       'Wdft_IsGamer', 'Wdft_RegionIdentifier',
       'EngineVersion', 'AppVersion', 'AvSigVersion', 'OsVer', 'OsBuildLab',
       'SmartScreen', 'Census_MDC2FormFactor', 'Census_ChassisTypeName',
       'Census_OSVersion']]


# In[151]:


for i in X:
    print(i)


# In[152]:


train['HasDetections'].unique()

from scipy.stats import mannwhitneyu

high = train[train['HasDetections']==1]
low = train[train['HasDetections']==0]

for i in X:
    try:
        print(i)
        stat, p = mannwhitneyu(high[i], low[i])
        print('stat={0:.3g}, p={0:.3g}'.format(stat, p))
        if p > 0.05:
            print('Probably the same distribution')
        else:
            print('Probably different distributions')
    except:
        print("An exception occurred")
# ### columnas a eliminar:
#     IsBeta
#     AutoSampleOptIn
#     Firewall
#     UacLuaenable
#     Census_OEMModelIdentifier
#     Census_IsPortableOperatingSystem
#     Census_IsFlightingInternal
#     Census_ThresholdOptIn
#     Census_IsSecureBootEnabled

# In[151]:


train = train.drop(columns=['IsBeta', 'AutoSampleOptIn', 'Firewall', 'UacLuaenable', 'Census_OEMModelIdentifier', 'Census_IsPortableOperatingSystem', 'Census_IsFlightingInternal', 'Census_ThresholdOptIn', 'Census_IsSecureBootEnabled'])
train


# In[152]:


train.columns


# # ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Ordinary Least Squares (OLS) model
model = ols('HasDetections ~ MachineIdentifier+ ProductName+ RtpStateBitfield+ IsSxsPassiveMode+ DefaultBrowsersIdentifier+ AVProductStatesIdentifier+ AVProductsInstalled+ AVProductsEnabled+ HasTpm+ CountryIdentifier+ CityIdentifier+ OrganizationIdentifier+ GeoNameIdentifier+ LocaleEnglishNameIdentifier+ Platform+ Processor+ OsBuild+ OsSuite+ OsPlatformSubRelease+ SkuEdition+ IsProtected+ PuaMode+ SMode+ IeVerIdentifier+ Census_DeviceFamily+ Census_OEMNameIdentifier+ Census_ProcessorCoreCount+ Census_ProcessorManufacturerIdentifier+ Census_ProcessorModelIdentifier+ Census_ProcessorClass+ Census_PrimaryDiskTotalCapacity+ Census_PrimaryDiskTypeName+ Census_SystemVolumeTotalCapacity+ Census_HasOpticalDiskDrive+ Census_TotalPhysicalRAM+ Census_InternalPrimaryDiagonalDisplaySizeInInches+ Census_InternalPrimaryDisplayResolutionHorizontal+ Census_InternalPrimaryDisplayResolutionVertical+ Census_PowerPlatformRoleName+ Census_InternalBatteryType+ Census_InternalBatteryNumberOfCharges+ Census_OSArchitecture+ Census_OSBranch+ Census_OSBuildNumber+ Census_OSBuildRevision+ Census_OSEdition+ Census_OSSkuName+ Census_OSInstallTypeName+ Census_OSInstallLanguageIdentifier+ Census_OSUILocaleIdentifier+ Census_OSWUAutoUpdateOptionsName+ Census_GenuineStateName+ Census_ActivationChannel+ Census_IsFlightsDisabled+ Census_FlightRing+ Census_FirmwareManufacturerIdentifier+ Census_FirmwareVersionIdentifier+ Census_IsVirtualDevice+ Census_IsTouchEnabled+ Census_IsPenCapable+ Census_IsAlwaysOnAlwaysConnectedCapable+ Wdft_IsGamer+ Wdft_RegionIdentifier+ EngineVersion+ AppVersion+ AvSigVersion+ OsVer+ OsBuildLab+ SmartScreen+ Census_MDC2FormFactor+ Census_ChassisTypeName+ Census_OSVersion', data=train).astype(np.uint8)
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
# # FEATURE ENGINEERING

# ## ExtraTreesClassifier method
# 
# Ayudará a dar la importancia de cada característica independiente con una característica dependiente. La importancia de la característica le dará una puntuación para cada característica en sus datos, cuanto mayor sea la puntuación, más importante o relevante para la característica en relación con su variable de salida.

# In[153]:


train.isna().any()[lambda x: x]


# In[154]:


train.columns


# In[155]:


y, X = train['HasDetections'], train[['ProductName', 'RtpStateBitfield',
       'IsSxsPassiveMode', 'DefaultBrowsersIdentifier',
       'AVProductStatesIdentifier', 'AVProductsInstalled', 'AVProductsEnabled',
       'HasTpm', 'CountryIdentifier', 'CityIdentifier',
       'OrganizationIdentifier', 'GeoNameIdentifier',
       'LocaleEnglishNameIdentifier', 'Platform', 'Processor', 'OsBuild',
       'OsSuite', 'OsPlatformSubRelease', 'SkuEdition', 'IsProtected',
       'PuaMode', 'SMode', 'IeVerIdentifier', 'Census_DeviceFamily',
       'Census_OEMNameIdentifier', 'Census_ProcessorCoreCount',
       'Census_ProcessorManufacturerIdentifier',
       'Census_ProcessorModelIdentifier', 'Census_ProcessorClass',
       'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',
       'Census_SystemVolumeTotalCapacity', 'Census_HasOpticalDiskDrive',
       'Census_TotalPhysicalRAM',
       'Census_InternalPrimaryDiagonalDisplaySizeInInches',
       'Census_InternalPrimaryDisplayResolutionHorizontal',
       'Census_InternalPrimaryDisplayResolutionVertical',
       'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',
       'Census_InternalBatteryNumberOfCharges', 'Census_OSArchitecture',
       'Census_OSBranch', 'Census_OSBuildNumber', 'Census_OSBuildRevision',
       'Census_OSEdition', 'Census_OSSkuName', 'Census_OSInstallTypeName',
       'Census_OSInstallLanguageIdentifier', 'Census_OSUILocaleIdentifier',
       'Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName',
       'Census_ActivationChannel', 'Census_IsFlightsDisabled',
       'Census_FlightRing', 'Census_FirmwareManufacturerIdentifier',
       'Census_FirmwareVersionIdentifier', 'Census_IsVirtualDevice',
       'Census_IsTouchEnabled', 'Census_IsPenCapable',
       'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer',
       'Wdft_RegionIdentifier', 'EngineVersion', 'AppVersion',
       'AvSigVersion', 'OsVer', 'OsBuildLab', 'SmartScreen',
       'Census_MDC2FormFactor', 'Census_ChassisTypeName', 'Census_OSVersion']]

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model =  ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)

fig, ax = plt.subplots(figsize = (15, 18))
feat_importances =pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(130).plot(kind='barh')
plt.show()pd.set_option('display.max_rows', 100)
feat_importances.sort_values(ascending=False)
# ## Pearson Correlation test

# In[156]:


matriz=train.corr(method='pearson')

# Generate the Heatmap
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(matriz,cmap='coolwarm')
plt.show()


# In[157]:


matriz


# In[158]:


#pd.set_option('display.max_rows', None)
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(matriz, 75))


# In[159]:


train = train.drop(columns=['Census_OSUILocaleIdentifier', 'Census_InternalPrimaryDisplayResolutionVertical', 'OsBuild', 'IsSxsPassiveMode', 'Census_PowerPlatformRoleName', 'Census_IsAlwaysOnAlwaysConnectedCapable' , 'AvSigVersion', 'Census_ProcessorManufacturerIdentifier', 'Census_OSBranch', 'Census_IsTouchEnabled', 'Census_ProcessorCoreCount', 'Census_InternalBatteryNumberOfCharges', 'Census_OSVersion', 'HasTpm', 'Census_ChassisTypeName', 'Census_OEMNameIdentifier', 'AvSigVersion' , 'Census_InternalPrimaryDisplayResolutionHorizontal'   , 'Census_OSSkuName' ])
X = X.drop(columns=['Census_OSUILocaleIdentifier', 'Census_InternalPrimaryDisplayResolutionVertical', 'OsBuild', 'IsSxsPassiveMode', 'Census_PowerPlatformRoleName', 'Census_IsAlwaysOnAlwaysConnectedCapable' , 'AvSigVersion', 'Census_ProcessorManufacturerIdentifier', 'Census_OSBranch', 'Census_IsTouchEnabled', 'Census_ProcessorCoreCount', 'Census_InternalBatteryNumberOfCharges', 'Census_OSVersion', 'HasTpm', 'Census_ChassisTypeName', 'Census_OEMNameIdentifier', 'AvSigVersion' , 'Census_InternalPrimaryDisplayResolutionHorizontal'   , 'Census_OSSkuName' ])
X


# In[160]:


# matriz=train.corr(method='pearson')

# Generate the Heatmap
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(matriz,cmap='coolwarm')
plt.show()


# # DBSCAN 
# En base a lo aprendido en clase se utilizara DBSCAN para eliminar valores atipicos o outliers en el modelo.
from sklearn.cluster import DBSCAN
from sklearn import metrics
db = DBSCAN(eps=2, min_samples=10).fit(X)
for i in set(db.labels_):
    print('class {}: number of points {:d}, number of positives {} (fraction: {:.3%})'.format(
        i,  np.sum(db.labels_==i), y[db.labels_==i].sum(), 
        y[db.labels_==i].mean()))core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))train['noise'] = .train.labels_
traindf_filtered = train[train.noise>-1] 
df_filteredtrain.noise.value_counts()df_filtered.noise.value_counts()
# # TEST AND TRAIN SPLIT
# 
# 
# NOTA: algunos problemas de clasificación no tienen un número equilibrado de ejemplos para cada etiqueta de clase. Como tal, es deseable dividir el conjunto de datos en conjuntos de entrenamiento y prueba de una manera que conserve las mismas proporciones de ejemplos en cada clase que se observan en el conjunto de datos original, utilizando estratificar

# In[161]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y, stratify=y, test_size=0.35,random_state=123)

training_base=pd.concat([X_train,y_train],axis=1)

print(training_base['HasDetections'].mean())


# In[162]:


X_train


# ## FINDING OUTLIERS 

# In[163]:


data=pd.concat([X_train,y_train],axis=1)


# In[164]:


desc_df = data.describe()
desc_df


# In[165]:


# finding the inter-quartile range for each column

intqtl_range = {col: desc_df[col].loc['75%'] - desc_df[col].loc['25%'] for col in desc_df.columns}

# Finding the upper and lower bound for each column

upper_bound = {col : desc_df[col].loc['75%'] + (intqtl_range[col]*1.5) for col in desc_df.columns}
lower_bound = {col : desc_df[col].loc['25%'] - (intqtl_range[col]*1.5) for col in desc_df.columns}

# Getting the number of instances with outliers for each column 
outlier_count = {col : len(data[(data[col]>upper_bound[col])|(data[col]<lower_bound[col])].index) for col in desc_df.columns}


# In[166]:


outlier = pd.DataFrame(np.array(list(outlier_count.items())),columns=['column','no_of_outliers'])
total = data.shape[0]
outlier['no_of_outliers'] = outlier['no_of_outliers'].apply(lambda val: int(val))
outlier['percentage_of_outliers'] = outlier['no_of_outliers'].apply(lambda val: (val/total)*100)


# In[167]:


first_20_outlier = outlier.sort_values('percentage_of_outliers',ascending=False).head(20)
first_20_outlier


# In[168]:


plt.figure(figsize=(18,12))
sns.barplot(first_20_outlier.percentage_of_outliers,first_20_outlier.column, orient='h')
plt.tight_layout()


# In[169]:


plt.figure(figsize=(10,8))
sns.distplot(outlier.percentage_of_outliers,bins=10)


# In[170]:


data = data.loc[(X_train["AVProductStatesIdentifier"]<upper_bound["AVProductStatesIdentifier"]) & (X_train["AVProductStatesIdentifier"]>lower_bound["AVProductStatesIdentifier"])]
data = data.loc[(X_train["Census_ProcessorModelIdentifier"]<upper_bound["Census_ProcessorModelIdentifier"]) & (X_train["Census_ProcessorModelIdentifier"]>lower_bound["Census_ProcessorModelIdentifier"])]
data = data.loc[(X_train["SmartScreen"]<upper_bound["SmartScreen"]) & (X_train["SmartScreen"]>lower_bound["SmartScreen"])]
data = data.loc[(X_train["Census_InternalPrimaryDiagonalDisplaySizeInInches"]<upper_bound["Census_InternalPrimaryDiagonalDisplaySizeInInches"]) & (X_train["Census_InternalPrimaryDiagonalDisplaySizeInInches"]>lower_bound["Census_InternalPrimaryDiagonalDisplaySizeInInches"])]
data = data.loc[(X_train["OsBuildLab"]<upper_bound["OsBuildLab"]) & (X_train["OsBuildLab"]>lower_bound["OsBuildLab"])]
data = data.loc[(X_train["Census_OSBuildRevision"]<upper_bound["Census_OSBuildRevision"]) & (X_train["Census_OSBuildRevision"]>lower_bound["Census_OSBuildRevision"])]
data = data.loc[(X_train["Census_TotalPhysicalRAM"]<upper_bound["Census_TotalPhysicalRAM"]) & (X_train["Census_TotalPhysicalRAM"]>lower_bound["Census_TotalPhysicalRAM"])]
#data = data.loc[(X_train["Census_GenuineStateName"]<upper_bound["Census_GenuineStateName"]) & (X_train["Census_GenuineStateName"]>lower_bound["Census_GenuineStateName"])]
#outlier_count = {col : len(X_train[(X_train[col]>upper_bound[col])|(X_train[col]<lower_bound[col])].index) for col in desc_df.columns}


# In[171]:


# finding the inter-quartile range for each column

intqtl_range = {col: desc_df[col].loc['75%'] - desc_df[col].loc['25%'] for col in desc_df.columns}

# Finding the upper and lower bound for each column

upper_bound = {col : desc_df[col].loc['75%'] + (intqtl_range[col]*1.5) for col in desc_df.columns}
lower_bound = {col : desc_df[col].loc['25%'] - (intqtl_range[col]*1.5) for col in desc_df.columns}

# Getting the number of instances with outliers for each column 
outlier_count = {col : len(data[(data[col]>upper_bound[col])|(data[col]<lower_bound[col])].index) for col in desc_df.columns}


# In[172]:


outlier = pd.DataFrame(np.array(list(outlier_count.items())),columns=['column','no_of_outliers'])
total = data.shape[0]
outlier['no_of_outliers'] = outlier['no_of_outliers'].apply(lambda val: int(val))
outlier['percentage_of_outliers'] = outlier['no_of_outliers'].apply(lambda val: (val/total)*100)


# In[173]:


first_20_outlier = outlier.sort_values('no_of_outliers',ascending=False).head(20)
first_20_outlier


# In[174]:


data


# In[175]:


X_test.columns


# In[176]:


y, X = train['HasDetections'], train[['Census_OSBuildNumber' , 'Census_IsPenCapable' , 'Census_OSBuildRevision','ProductName', 'Census_PrimaryDiskTotalCapacity', 'GeoNameIdentifier', 'RtpStateBitfield', 'DefaultBrowsersIdentifier',
       'AVProductStatesIdentifier', 'AVProductsEnabled', 'CountryIdentifier',
       'CityIdentifier', 'OrganizationIdentifier',
       'LocaleEnglishNameIdentifier', 'Platform', 'Processor', 'OsSuite',
       'OsPlatformSubRelease', 'SkuEdition', 'IsProtected', 'PuaMode', 'SMode',
       'IeVerIdentifier', 'Census_DeviceFamily',
       'Census_ProcessorModelIdentifier', 'Census_ProcessorClass',
       'Census_PrimaryDiskTypeName', 'Census_SystemVolumeTotalCapacity',
       'Census_HasOpticalDiskDrive', 'Census_TotalPhysicalRAM', 'Census_InternalPrimaryDiagonalDisplaySizeInInches',
       'Census_InternalBatteryType', 'Census_OSArchitecture',
       'Census_OSEdition', 'Census_OSInstallTypeName',
       'Census_OSInstallLanguageIdentifier',
       'Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName',
       'Census_ActivationChannel', 'Census_IsFlightsDisabled',
       'Census_FlightRing', 'Census_FirmwareManufacturerIdentifier',
       'Census_FirmwareVersionIdentifier', 'Census_IsVirtualDevice',
       'Wdft_IsGamer', 'Wdft_RegionIdentifier', 'EngineVersion', 'AppVersion',
       'OsVer', 'OsBuildLab', 'SmartScreen', 'Census_MDC2FormFactor', 'AVProductsInstalled']]


# In[177]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y, stratify=y, test_size=0.35,random_state=123)

training_base=pd.concat([X_train,y_train],axis=1)

print(training_base['HasDetections'].mean())

## SCALER 
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
from scipy.stats import zscore
X_train = X_train.apply(zscore)
X_test = X_test.apply(zscore)
# In[178]:


X_train


# # Logistic Regression: 
# 
# Es un método de regresión que permite estimar la probabilidad de una variable cualitativa binaria en función de una variable cuantitativa. Una de las principales aplicaciones de la regresión logística es la clasificación binaria, en la que las observaciones se clasifican en un grupo u otro en función del valor de la variable utilizada como predictor.
# 
# El intercepto (a menudo llamado constante) es el valor medio esperado de Y cuando todo X = 0.

# In[179]:


# X_train = X_train.astype(np.float64)
import statsmodels.api as sm

model_without_intercept=sm.Logit(y_train,X_train).fit(method='newton')
print(model_without_intercept.summary())


# In[180]:


X1_train = sm.add_constant(X_train)

modelo_with_intercept=sm.Logit(y_train,X1_train).fit(method='newton')
print(modelo_with_intercept.summary())


# In[181]:


print(model_without_intercept.summary2())


# In[182]:


print(modelo_with_intercept.summary2())


# In[183]:


X_train = X_train.drop(columns=['Census_OSBuildRevision', 'Census_SystemVolumeTotalCapacity' ,'IeVerIdentifier', 'Census_FirmwareManufacturerIdentifier'  , 'GeoNameIdentifier', 'OrganizationIdentifier', 'ProductName' , 'Processor'  , 'CountryIdentifier', 'CityIdentifier', 'LocaleEnglishNameIdentifier','Census_OSArchitecture', 'Census_IsFlightsDisabled', 'Census_FirmwareVersionIdentifier'])
X_train


# In[184]:


X1_train = X1_train.drop(columns=[ 'CityIdentifier', 'OrganizationIdentifier', 'LocaleEnglishNameIdentifier', 'Platform' , 'Processor' , 'SkuEdition', 'Census_DeviceFamily' , 'Census_SystemVolumeTotalCapacity' ,  'Census_OSArchitecture', 'Census_OSWUAutoUpdateOptionsName', 'Census_IsFlightsDisabled', 'Census_FirmwareVersionIdentifier', 'OsVer' ])
X1_train


# # PROBAR CON VARIABLES ELIMINADAS

# In[185]:


import statsmodels.api as sm

model_without_intercept=sm.Logit(y_train,X_train).fit(method='newton')
print(model_without_intercept.summary())


# In[186]:


X1_train = sm.add_constant(X_train)

modelo_with_intercept=sm.Logit(y_train,X1_train).fit(method='newton')
print(modelo_with_intercept.summary())


# In[187]:


print(model_without_intercept.summary2())


# In[188]:


print(modelo_with_intercept.summary2())


# Se utilizara el modelo sin intercepto dado que utiliza mas variables y tiene menor aic y bic

# In[189]:


probabilidades_sin_constante= model_without_intercept.predict(X_train)


# In[190]:


fig, ax = plt.subplots(figsize = (10, 10))
plt.hist(probabilidades_sin_constante,bins=30)
plt.title('Distribución de Probabilidades del modelo sin Constante')
plt.show()


# In[191]:


X_test = X_test.drop(columns=['Census_OSBuildRevision', 'Census_SystemVolumeTotalCapacity' , 'IeVerIdentifier', 'Census_FirmwareManufacturerIdentifier'  , 'GeoNameIdentifier', 'OrganizationIdentifier', 'ProductName' , 'Processor'  , 'CountryIdentifier', 'CityIdentifier', 'LocaleEnglishNameIdentifier','Census_OSArchitecture', 'Census_IsFlightsDisabled', 'Census_FirmwareVersionIdentifier'])
X_test


# In[192]:


probabilidades_sin_constante= model_without_intercept.predict(X_test)
fig, ax = plt.subplots(figsize = (10, 10))
plt.hist(probabilidades_sin_constante,bins=30)
plt.title('Distribución de Probabilidades del modelo sin Constante')
plt.show()


# In[193]:


from sklearn.metrics import (confusion_matrix, accuracy_score)

ypred = model_without_intercept.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[194]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[195]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = model_without_intercept.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[196]:


X_test


# In[197]:


matriz=X_test.corr(method='pearson')

# Generate the Heatmap
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(matriz,cmap='coolwarm')
plt.show()


# In[198]:


sample = pd.read_csv('test_sample.csv',low_memory=True)
sample


# In[199]:


X_test.columns


# ### TESTING SAMPLE

# In[200]:


y_sample, X_sample = sample['HasDetections'], sample[['Census_OSBuildNumber', 'Census_IsPenCapable',
       'Census_PrimaryDiskTotalCapacity', 'RtpStateBitfield',
       'DefaultBrowsersIdentifier', 'AVProductStatesIdentifier',
       'AVProductsEnabled', 'Platform', 'OsSuite', 'OsPlatformSubRelease',
       'SkuEdition', 'IsProtected', 'PuaMode', 'SMode', 'Census_DeviceFamily',
       'Census_ProcessorModelIdentifier', 'Census_ProcessorClass',
       'Census_PrimaryDiskTypeName', 'Census_HasOpticalDiskDrive',
       'Census_TotalPhysicalRAM',
       'Census_InternalPrimaryDiagonalDisplaySizeInInches',
       'Census_InternalBatteryType', 'Census_OSEdition',
       'Census_OSInstallTypeName', 'Census_OSInstallLanguageIdentifier',
       'Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName',
       'Census_ActivationChannel', 'Census_FlightRing',
       'Census_IsVirtualDevice', 'Wdft_IsGamer', 'Wdft_RegionIdentifier',
       'EngineVersion', 'AppVersion', 'OsVer', 'OsBuildLab', 'SmartScreen',
       'Census_MDC2FormFactor', 'AVProductsInstalled']]


# In[201]:


from sklearn.metrics import (confusion_matrix, accuracy_score)

#X_sample = sc_x.fit_transform(X_sample)

ypred = model_without_intercept.predict(X_sample)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_sample, prediction))


# In[202]:


X_sample


# In[203]:


from sklearn.metrics import classification_report
print(classification_report(y_sample, prediction))


# In[204]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = model_without_intercept.predict(X_sample)

fpr, tpr, thresholds = roc_curve(y_sample, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_sample, y_prob)))


# ## LOGISTIC REGRESSION WITH SKLEARN 

# In[205]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
sc_x = MinMaxScaler()

#Variables independientes entrenamiento estandarizadas
X_train_scaled = sc_x.fit_transform(X_train)
X_test_scaled = sc_x.fit_transform(X_test)


# In[206]:


lgr = LogisticRegression(solver='newton-cg' , class_weight="balanced")
lgr.fit(X_train, y_train)


# In[207]:


from sklearn.metrics import (confusion_matrix, accuracy_score)

ypred = lgr.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[208]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[209]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = lgr.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[210]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = lgr.predict(X_sample)

fpr, tpr, thresholds = roc_curve(y_sample, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_sample, y_prob)))


# ## SVC - RBF

# In[211]:


from sklearn import svm


# In[212]:


model = svm.SVC(kernel = 'rbf', C = 1)
model = model.fit(X_train, y_train)


# In[213]:


ypred = model.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[214]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[215]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[216]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = lgr.predict(X_sample)

fpr, tpr, thresholds = roc_curve(y_sample, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_sample, y_prob)))


# ## SVC RBF - C 100 

# In[217]:


from sklearn import svm
model = svm.SVC(kernel = 'rbf', C = 100)
model = model.fit(X_train,y_train)


# In[218]:


ypred = model.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[219]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[220]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[221]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = lgr.predict(X_sample)

fpr, tpr, thresholds = roc_curve(y_sample, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_sample, y_prob)))


# ## SVC linear C 1

# In[ ]:


from sklearn import svm
model = svm.SVC(kernel = 'linear', C = 1)
model = model.fit(X_train,y_train)


# In[ ]:


ypred = model.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = lgr.predict(X_sample)

fpr, tpr, thresholds = roc_curve(y_sample, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_sample, y_prob)))


# ## SVC LINEAR C 100

# In[ ]:


from sklearn import svm
model = svm.SVC(kernel = 'linear', C = 100)
model = model.fit(X_train,y_train)


# In[ ]:


ypred = model.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = lgr.predict(X_sample)

fpr, tpr, thresholds = roc_curve(y_sample, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_sample, y_prob)))


# ## SVC POLY C 100

# In[ ]:


from sklearn import svm
model = svm.SVC(kernel = 'poly', C = 100)
model = model.fit(X_train,y_train)


# In[ ]:


ypred = model.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = lgr.predict(X_sample)

fpr, tpr, thresholds = roc_curve(y_sample, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_sample, y_prob)))


# ## SVC SIGMOID 

# In[ ]:


from sklearn import svm
model = svm.SVC(kernel = 'sigmoid', C = 100)
model = model.fit(X_train,y_train)


# In[ ]:


ypred = model.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = lgr.predict(X_sample)

fpr, tpr, thresholds = roc_curve(y_sample, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_sample, y_prob)))


# ## SVC RBF (El que mejor funciono)

# In[ ]:


# Tunning with Gamma: AUTO
model = svm.SVC(kernel='rbf', C=1, gamma='auto')
model = model.fit(X_train, y_train)


# In[ ]:


ypred = model.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# In[ ]:


# Get the probabilities for each of the two categories
y_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[ ]:


# Tunning with Gamma: 0.1
model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
model = model.fit(X_train, y_train)


# In[ ]:


ypred = model.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# In[ ]:


# Get the probabilities for each of the two categories
y_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[ ]:


# Tunning with Gamma: Auto and C: 1000
model = svm.SVC(kernel='rbf', C=1000, gamma='auto')
model = model.fit(X_train, y_train)


# In[ ]:


ypred = model.predict(X_test)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# In[ ]:


# Get the probabilities for each of the two categories
y_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# ## XGBoost

# In[ ]:


X_test


# In[ ]:


X_train


# In[ ]:


import xgboost as xgb
import time
import gc


# # parametros:
# 
# gamma mientras mas alto menos riesgo
# 
# colsample_bytree valores pequenos
# 
# subsample numeros mas pequenos
# 
# max_depth valores mas pequenos
# 
# gamma  mas alto menos riesgo
# 
# the learning rate mas pequeno
# 
# min_child_weight mas alto menos riesgo

# In[ ]:


X_test = X_test.astype(np.float64)
start_time = time.time()

clf_xgb = xgb.XGBClassifier(learning_rate=0.01, 
                            n_estimators=500, 
                            max_depth=2 ,
                            min_child_weight = 10,
                            gamma=100,
                            objective= 'binary:logistic',
                            nthread=4,
                            seed=42)

clf_xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], 
            early_stopping_rounds=100, eval_metric='auc', verbose=100)

predictions = clf_xgb.predict(X_test)

print(classification_report(y_test, predictions))
print("accuracy_score", accuracy_score(y_test, predictions))
predictions_probas = clf_xgb.predict_proba(X_test)
print("roc-auc score for the class 1, from target 'HasDetections' ", roc_auc_score(y_test, predictions_probas[:,1]))
print("elapsed time in seconds: ", time.time() - start_time)
gc.collect()


# In[ ]:


y_prob = clf_xgb.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, predictions_probas[:,1])

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, predictions_probas[:,1])))


# In[ ]:


xgb.plot_importance(clf_xgb)
plt.figure(figsize = (16, 12))
plt.show()


# In[ ]:


data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)


# In[ ]:


from xgboost import cv

params = {"learning_rate":0.01, "max_depth":3 ,"min_child_weight": 10, "gamma":500, "objective":'binary:logistic', "nthread":4}

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)


# In[ ]:


xgb_cv.head()


# In[ ]:


data_dmatrix = xgb.DMatrix(data=X_test,label=y_test)
from xgboost import cv

params = {"learning_rate":0.01, "max_depth":3 ,"min_child_weight": 10, "gamma":500, "objective":'binary:logistic', "nthread":4}

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)


# In[ ]:


xgb_cv.head()


# In[ ]:


X_test.columns


# In[ ]:


sample = pd.read_csv('test_sample.csv',low_memory=True)
sample


# In[ ]:


y, X = sample['HasDetections'], sample[[
'Census_OSBuildNumber', 'Census_IsPenCapable',
       'Census_PrimaryDiskTotalCapacity', 'RtpStateBitfield',
       'DefaultBrowsersIdentifier', 'AVProductStatesIdentifier',
       'AVProductsEnabled', 'Platform', 'OsSuite', 'OsPlatformSubRelease',
       'SkuEdition', 'IsProtected', 'PuaMode', 'SMode', 'Census_DeviceFamily',
       'Census_ProcessorModelIdentifier', 'Census_ProcessorClass',
       'Census_PrimaryDiskTypeName', 'Census_HasOpticalDiskDrive',
       'Census_TotalPhysicalRAM',
       'Census_InternalPrimaryDiagonalDisplaySizeInInches',
       'Census_InternalBatteryType', 'Census_OSEdition',
       'Census_OSInstallTypeName', 'Census_OSInstallLanguageIdentifier',
       'Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName',
       'Census_ActivationChannel', 'Census_FlightRing',
       'Census_IsVirtualDevice', 'Wdft_IsGamer', 'Wdft_RegionIdentifier',
       'EngineVersion', 'AppVersion', 'OsVer', 'OsBuildLab', 'SmartScreen',
       'Census_MDC2FormFactor', 'AVProductsInstalled']]


# In[ ]:


X = X.astype(np.float64)


# In[ ]:


start_time = time.time()

clf_xgb = xgb.XGBClassifier(learning_rate=0.1, 
                            n_estimators=1000, 
                            max_depth=3,
                            min_child_weight=10,
                            gamma=1000,
                            subsample=0.8,
                            colsample_bytree=0.6,
                            objective= 'binary:logistic',
                            nthread=-1,
                            scale_pos_weight=1,
                            reg_alpha = 0,
                            reg_lambda = 1,
                            seed=42)

clf_xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X, y)], 
            early_stopping_rounds=100, eval_metric='auc', verbose=100)

predictions = clf_xgb.predict(X)

print(classification_report(y, predictions))
print("accuracy_score", accuracy_score(y, predictions))
predictions_probas = clf_xgb.predict_proba(X)
print("roc-auc score for the class 1, from target 'HasDetections' ", roc_auc_score(y, predictions_probas[:,1]))
print("elapsed time in seconds: ", time.time() - start_time)
gc.collect()


# Se esta cayendo en overfitting!

# ## REVISANDO PARA REGRESION LOGISTICA

# In[ ]:


from sklearn.metrics import (confusion_matrix, accuracy_score)

ypred = model_without_intercept.predict(X)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y, prediction))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y, prediction))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = model_without_intercept.predict(X)

fpr, tpr, thresholds = roc_curve(y, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y, y_prob)))


# estimator = xgb.XGBClassifier(
#     objective= 'binary:logistic'
#     nthread=4,
#     seed=42
# )

# parameters = {
#     'max_depth': range (2, 10, 1),
#     'n_estimators': range(60, 220, 40),
#     'learning_rate': [0.1, 0.01, 0.05]
# }

# from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(
#     estimator=estimator,
#     param_grid=parameters,
#     scoring = 'roc_auc',
#     n_jobs = 2,
#     cv = 3,
#     verbose=True
# )

# grid_search.fit(X_train, y_train)
grid_search.best_estimator_
# ## DECISION TREES 

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']
             }
tree_clas = DecisionTreeClassifier(random_state=1024)
grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(X_train, y_train)


# In[ ]:


final_model = grid_search.best_estimator_
final_model


# In[ ]:


tree_clas = DecisionTreeClassifier(ccp_alpha=0.001, max_depth=9, max_features='auto',
                       random_state=1024)
tree_clas.fit(X_train, y_train)
y_predict = tree_clas.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = tree_clas.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[ ]:


y_predict = tree_clas.predict(X)

print(classification_report(y, y_predict))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = tree_clas.predict(X)

fpr, tpr, thresholds = roc_curve(y, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y, y_prob)))


# ## NEURAL NETWORK MLP 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_val_predict, validation_curve

kfold = KFold(n_splits=5,random_state=7, shuffle=True)
clf = MLPClassifier(solver='lbfgs', random_state=1, activation='logistic',  hidden_layer_sizes=(15,))
param_grid = {"alpha":10.0 ** -np.arange(-4, 7)}
grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kfold)
grid.fit(X_train, y_train)
print (grid.best_estimator_)
print (grid.best_score_*100, "%")
# In[ ]:


X_test


# In[ ]:


kfold = KFold(n_splits=5,random_state=7, shuffle=True)
clf = MLPClassifier(solver='lbfgs', random_state=1, alpha=1e-5, activation='logistic',  hidden_layer_sizes=(34,))


# In[ ]:


clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = clf.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# ## POISSON

# In[ ]:


import statsmodels.api as sm

poission_model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
result = poission_model.fit(method="newton")
result.summary()


# In[ ]:


X1_train = X_train.drop(columns=['Platform', 'OsSuite', 'IeVerIdentifier', 'Census_OSEdition', 'Census_OSInstallLanguageIdentifier' , 'Census_FirmwareManufacturerIdentifier'])
X1_train
X1_test = X_test.drop(columns=['Platform', 'OsSuite', 'IeVerIdentifier', 'Census_OSEdition', 'Census_OSInstallLanguageIdentifier' , 'Census_FirmwareManufacturerIdentifier'])
X1_test
X1 = X.drop(columns=['Platform', 'OsSuite', 'IeVerIdentifier', 'Census_OSEdition', 'Census_OSInstallLanguageIdentifier' , 'Census_FirmwareManufacturerIdentifier'])
X1
poission_model = sm.GLM(y_train, X1_train, family=sm.families.Poisson())
result = poission_model.fit(method="newton")
result.summary()


# In[ ]:


probabilidades_sin_constante=  result.predict(X1_train)


# In[ ]:


from sklearn.metrics import (confusion_matrix, accuracy_score)

ypred = result.predict(X1_train)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_train, prediction))


# In[ ]:


ypred


# ## Linear Probability Model 

# In[ ]:


#model_without_intercept=sm.Logit(y_train,X_train).fit(method='newton')
lpm_mod = sm.OLS(y_train, X_train)
lpm_res = lpm_mod.fit_regularized(alpha=2., L1_wt=0.1, refit=True)
lpm_res.summary()


# In[ ]:


X1_train = X_train.drop(columns=['Platform', 'Census_OSEdition'])
X1_train
X1_test = X_test.drop(columns=['Platform', 'Census_OSEdition'])
X1_test
X1 = X.drop(columns=['Platform', 'Census_OSEdition'])
X1
lpm_mod = sm.OLS(y_train, X1_train)
lpm_res = lpm_mod.fit()
lpm_res.summary()


# In[ ]:


from sklearn.metrics import (confusion_matrix, accuracy_score)

ypred = lpm_res.predict(X1_train)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y_train, prediction))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = lpm_res.predict(X1_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob)))


# In[ ]:


from sklearn.metrics import (confusion_matrix, accuracy_score)

ypred = lpm_res.predict(X1)
prediction = list(map(round, ypred))
print('Test accuracy = ', accuracy_score(y, prediction))


# In[ ]:


from sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = lpm_res.predict(X1)

fpr, tpr, thresholds = roc_curve(y, y_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y, y_prob)))


# In[ ]:





# # SMOTE
# 
# SMOTE es una técnica estadística de sobremuestreo de minorías sintéticas para aumentar el número de casos de un conjunto de datos de forma equilibrada. El componente funciona cuando genera nuevas instancias a partir de casos minoritarios existentes que se proporcionan como entrada.
X_train = X_train.astype(np.float64)print(X_train.info())X_train.isna().any()[lambda x: x]np.isinf(X_train).values.sum()y_train.unique()from imblearn.over_sampling import SMOTE

smote=SMOTE(random_state=123,sampling_strategy='minority')
#y_train,X_train
X_res,y_res=smote.fit_resample(X_train,y_train)

#Generar la regresión logística
modelo_con_constante=sm.Logit(y_res,X_res).fit(method='powell')
print(modelo_con_constante.summary())X_res = X_res.drop(columns=['RtpStateBitfield', 'Platform', 'Census_DeviceFamily', 'Census_ProcessorModelIdentifier'])
X_res
# #Generar la regresión logística
# modelo_con_constante=sm.Logit(y_res,X_res).fit(method='powell')
# print(modelo_con_constante.summary())
ypred = modelo_con_constante.predict(X_res)
prediction = list(map(round, ypred))
from sklearn.metrics import (confusion_matrix, accuracy_score)
print('Test accuracy = ', accuracy_score(y_res, prediction))from sklearn.metrics import classification_report
print(classification_report(y_res, prediction))#Generando las predicciones del modelo
probabilidades_con_constante=modelo_con_constante.predict(X_res)

#Mostrando la distribucion de probabilidades de las predicciones
fig, ax = plt.subplots(figsize = (10, 10))
plt.hist(probabilidades_con_constante,bins=30)
plt.title('Distribución de Probabilidades del modelo sin Constante con SMOTE')
plt.show()from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_res, probabilidades_con_constante)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_res, probabilidades_con_constante)))
# In[ ]:





# # KNN 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

k_range = list(range(1,31))

param_grid = dict(n_neighbors = k_range)
#print (param_grid)
knn = KNeighborsClassifier()

Random = RandomizedSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
Random.fit(X_train,y_train)

print (Random.best_score_)
print (Random.best_params_)
print (Random.best_estimator_)X_train
# ## TRAIN
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(Random, X_train,y_train, scoring='accuracy', cv=cv, n_jobs=10)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))ypred = Random.predict(X_train)
from sklearn.metrics import classification_report
print(classification_report(y_train, ypred))from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('talk')
cm = confusion_matrix(y_train , ypred)
ax = sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicho")
plt.ylabel("Valor Real")
### END SOLUTIONfrom sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = Random.predict_proba(X_train)

fpr, tpr, thresholds = roc_curve(y_train, y_prob[:,1])

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_train, y_prob[:,1])))
# ## TEST 
ypred = Random.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, ypred))from sklearn.metrics import confusion_matrix

sns.set_context('talk')
cm = confusion_matrix(y_test , ypred)
ax = sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicho")
plt.ylabel("Valor Real")
### END SOLUTIONfrom sklearn.metrics import roc_curve

# Get the probabilities for each of the two categories
y_prob = Random.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score
print("AUC con Constante: {}".format(roc_auc_score(y_test, y_prob[:,1])))
# ## RNA
from keras import callbacks
from sklearn.metrics import roc_auc_score

class printAUC(callbacks.Callback):
    def __init__(self, X_train, y_train):
        super(printAUC, self).__init__()
        self.bestAUC = 0
        self.X_train = X_train
        self.y_train = y_train
        
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(np.array(self.X_train))
        auc = roc_auc_score(self.y_train, pred)
        print("Train AUC: " + str(auc))
        return# Tenemos que importar Keras y funciones adicionales de sublibrerías de Keras
import keras # a su vez importa TensorFlow
from keras.models import Sequential # necesitamos la función de Sequential para inicializar los valores de los parámetros de las RNA
from keras.layers import Dense # y la función Dense que sirve para crear cada una de las capas intermedias (ocultas) de las RNA
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import LearningRateSchedulerfrom sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

#Variables independientes entrenamiento estandarizadas
X_train = sc_x.fit_transform(X_train)rna = Sequential()X_train.shapemodel = Sequential()
model.add(Dense(100,input_dim=33))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))model.compile(optimizer=Adam(lr=0.01), loss="binary_crossentropy", metrics=["accuracy"])annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)

# TRAIN MODEL
model.fit(X_train,y_train, batch_size=32, epochs = 20, callbacks=[annealer,
          printAUC(X_train, y_train)], validation_data = (X_test, y_test), verbose=2)
# In[ ]:





# In[ ]:




