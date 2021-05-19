import pandas as pd
import numpy as np
from sklearn import ensemble as ske

df = pd.read_csv("immo_data.csv")

df.head(3).T

df.columns

df.dtypes


def get_nan_stats(df):
    
    nan_stats_cat = df.select_dtypes(include='object').isna().sum()
    
    with pd.option_context('mode.use_inf_as_na', True):
        nan_stats_num = df.select_dtypes(exclude='object').isna().sum()
        
    nan_stats_cat = nan_stats_cat[nan_stats_cat > 0]
    nan_stats_num = nan_stats_num[nan_stats_num > 0]
    
    return nan_stats_cat, nan_stats_num


# +
nan_stat_cat, nan_stat_num = get_nan_stats(df)

drop_it = [cat for (cat,nans) in nan_stat_num.iteritems() if nans*2 > len(df)]
drop_it.extend([cat for (cat,nans) in nan_stat_cat.iteritems() if nans*2 > len(df)])
drop_it.extend(['scoutId','regio1','picturecount','street','streetPlain','description','facilities','date'])
df = df.drop(columns=drop_it)
# -

classifier = ske.RandomForestClassifier(n_estimators=100,
                           criterion='gini', max_depth=None, min_samples_split=2, 
                           min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                           max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                           bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, 
                           warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

train, validate, test = np.split(df.sample(frac=1, random_state=42), 
                           [int(.6*len(df)), int(.8*len(df))])

data, label = df.drop(columns=['totalRent'], inplace=False), pd.qcut(df['totalRent'], q=10)

for column, dtype in data.dtypes.iteritems():
    if dtype != 'float64':
        data[column] = pd.Categorical(df[column])
        df['code'] = df.cc.cat.codes
data.head(2).T


