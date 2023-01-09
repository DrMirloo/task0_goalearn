#Multiple Linear Regression With given data
import numpy as np
from sklearn.linear_model import LinearRegression

x = [ [5, 6, 7], [2, 10, 10], [7, 9, 6], [3, 7, 10], [11, 7, 8] ]
y = [60000, 65000, 70000, 62000, 80000]
x, y = np.array(x), np.array(y)
print(x)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"coefficients: {model.coef_}")

#calculate missing data
experience1=(50000-(model.intercept_+model.coef_[1]*8+model.coef_[2]*9))//model.coef_[0]
if experience1<0:
    experience1=0
print(f"experience1: {experience1}")
experience2=(45000-(model.intercept_+model.coef_[1]*8+model.coef_[2]*6))//model.coef_[0]
if experience2<0:
    experience2=0
print(f"experience2: {experience2}")
test_score7=(72000-(model.intercept_+model.coef_[0]*10+model.coef_[2]*7))//model.coef_[1]
if test_score7>10:
    test_score7=10
print(f"test_score7: {test_score7}")
#Multiple Linear Regression With complete data
x = [ [0, 8, 9], [0, 8, 6], [5, 6, 7], [2, 10, 10], [7, 9, 6], [3, 7, 10], [10, 5, 7], [11, 7, 8] ]
y = [50000, 45000, 60000, 65000, 70000, 62000, 72000, 80000]
x, y = np.array(x), np.array(y)
print(x)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"coefficients: {model.coef_}")

y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")
x_new = [[2, 9, 6], [12, 10, 10]]
y_new = model.predict(x_new)
print(f"expected salary in task:\n{y_new}")

file1 = open("task0.txt","w")
file1.write("Hello \n")
file1.writelines("coefficient of determination:")
file1.writelines(str(r_sq))
file1.write("\n")
file1.writelines("intercept:")
file1.writelines(str(model.intercept_))
file1.write("\n")
file1.writelines("coefficients:")
file1.writelines(str(model.coef_))
file1.write("\n")
file1.writelines("expected salary in task:")
file1.writelines(str(y_new))
file1.close()