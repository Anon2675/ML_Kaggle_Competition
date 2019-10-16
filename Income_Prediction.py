import pandas as pd  #dataset and reading csv
import numpy as np    #for computing of RMSE
#from sklearn.linear_model import LinearRegression   #former regressor
from sklearn import metrics     #for use in RMSE
from sklearn.compose import ColumnTransformer  #column transformer for differing data combination
from sklearn.pipeline import Pipeline      #pipeline in sklearn
from sklearn.impute import SimpleImputer    #imputer for filling in missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder    #preprocessing scaler and encoder
from sklearn.model_selection import train_test_split     #data splitter
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor  #Regressors
from category_encoders import * #encoder library to access Target Encoder
import plotly.express as px     #graphical representation

#Reading in the files
training_data = pd.read_csv(r'C:\Users\Megan\Desktop\tcdml1920-income-ind\training.csv')
testing_data = pd.read_csv(r'C:\Users\Megan\Desktop\tcdml1920-income-ind\testing.csv')
#auxiliary data
testing_data1=testing_data

#Assigning variables of Income and others to y and x respectively
training_y = training_data['Income in EUR']
training_x = training_data.drop(['Income in EUR'],axis=1)
#Assigning variables of Income and others to y and x repectively for the testing file
testing_y = testing_data['Income']
testing_x = testing_data.drop(['Income'],axis=1)

#splitting traiing data file to 80 20 for testing
x_train,x_test,y_train,y_test = train_test_split(training_x,training_y,test_size=.2)

#Imputation and Scaling of numerical features to remove NaN values and reduce outliers
numeric_features = ['Year of Record','Age','Size of City','Wears Glasses','Body Height [cm]']
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])

#Encoding of categorical values followed by an imputation using the mean strategy
categorical_features = ['Gender','Country','Profession','Hair Color', 'University Degree']
categorical_transformer = Pipeline(steps=[ ('target', TargetEncoder()), ('imputer', SimpleImputer(strategy='mean'))])
    #('onehot', OneHotEncoder(handle_unknown='ignore'))

#Combining the categorical and numerical features using a columnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

#Regressor, AdaBoost using a baseline RandomForestRegressor, setting a single random state for continuity
clf = Pipeline(steps=[('preprocessor', preprocessor),
                  ('regressor', AdaBoostRegressor(base_estimator=RandomForestRegressor(random_state =0),random_state =0))]) 

#Fitting the data                  
clf.fit(training_x,training_y)
#Predicting the new Income values for the testing file
testing_y=clf.predict(testing_x)
#to view predicted data in terminal
#print(testing_y)

#Testing the model on the 20% of the training data file to produce an RMSE
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#Filling the empty values in the test file with the new predicted values and returning it to a new file 
testing_data1['Income'] =testing_y
testing_data1.to_csv(r'C:\Users\Megan\Desktop\tcdml1920-income-ind\new.csv',index=False)

#Used for graphical representation of features, x value was interchanged depending on the feature in question
#fig = px.scatter(training_data, x = 'Age' , y = 'Income in EUR', title='Graph')
#fig.show()