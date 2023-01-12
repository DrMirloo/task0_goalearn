#Linear Regression With given data
import numpy as np
from sklearn.linear_model import LinearRegression
# I assume x as math scores and y as cs score
x = [ [92], [56], [88], [70], [80], [49], [65], [35], [66], [67] ]
y = [98, 68, 81, 80, 83, 52, 66, 30, 68, 73]
x, y = np.array(x), np.array(y)
print(x)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"coefficients: {model.coef_}")
#save the model using pickle module then load the model using pickle
# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
# save the model to disk
filename = 'reg_model_pickle.sav'
pickle.dump(model, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = model.score(x, y)
print(result)
#save the model using joblib module then load the model using joblib

# Save Model Using joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
# save the model to disk
filename = 'reg_model_joblib.sav'
joblib.dump(model, filename)
# load the model from disk
loaded_model = joblib.load(filename)
result2 = model.score(x, y)
print(result)