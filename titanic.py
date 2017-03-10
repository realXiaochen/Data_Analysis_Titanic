# Titanic data
# source: kaggle.com
# xiaochen zhuo

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.cross_validation import KFold


# Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test.head(5)

full_data = [train, test]


for dataset in full_data:
	# Had a cabin or not
	dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

	# Add name length
	dataset['Name_length'] = dataset['Name'].apply(len)

	# Add FamilySize
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

	# Add IsAlone
	dataset['IsAlone'] = 0
	dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

	# Remove all NULLS in the Embarked column
	dataset['Embarked'] = dataset['Embarked'].fillna('S')

	# Remove all NULLS in the Fare column and create a new feature CategoricalFare
	dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# Fill empty in Age
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)


for dataset in full_data:

	# Mapping Age
	dataset.loc[ dataset['Age'] <= 15, 'Age'] 					       = 0
	dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 32), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 31) & (dataset['Age'] <= 48), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 47) & (dataset['Age'] <= 64), 'Age'] = 3
	dataset.loc[ dataset['Age'] > 63, 'Age']  = 4

	# Map Sex
	dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

	# Mapping Embarked
	dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

	# Mapping Fare
	dataset.loc[ dataset['Fare'] <= 7.8, 'Fare'] 						        = 0
	dataset.loc[(dataset['Fare'] > 7.8) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
	dataset.loc[(dataset['Fare'] > 14.4) & (dataset['Fare'] <= 31), 'Fare']   = 2
	dataset.loc[ dataset['Fare'] > 30, 'Fare'] 							        = 3
	dataset['Fare'] = dataset['Fare'].astype(int)


# Drop unnecessary features
to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(to_drop, axis = 1)
test  = test.drop(to_drop, axis = 1)



# Correlation visualization
plt.matshow(train.corr())



n_train = train.shape[0]
n_test = test.shape[0]
s = 0


# create a svm object
class Classifier(object):
	def __init__(self,clf,seed=0,params=None):
		params['random_state'] = seed
		self.clf = clf(**params)

	def train(self, x_train, y_train):
		self.clf.fit(x_train,y_train)
	
	def predict(self,x):
		return self.clf.predict(x)
	
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

svc = Classifier(SVC,seed=s,params=svc_params)


# Get x_train (feature), y_train(label)
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data


# Cross validation
ntrain = train.shape[0]
SEED = 0
N_FOLD = 5
kf =  KFold(ntrain, n_folds= N_FOLD, random_state=SEED)

result = []

for i, (train_index, test_index) in enumerate(kf):
    x_tr = x_train[train_index]
    y_tr = y_train[train_index]
    svc.train(x_tr, y_tr)

    x_predict = svc.predict(x_train[test_index])
    x_label = y_train[test_index]

    result.append( np.sum(x_predict == x_label)/float(len(x_predict)))

# Average accuracy
print float(sum(result))/len(result)









