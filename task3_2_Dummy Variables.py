"""
use pandas to create dummy 
variables and save it into dummies.
now tell me what is Dummy
 Variable Trap and how to avoid that?
concatinate dummies to orginal data 
 frame and drop extra column
save csv file
"""

# import required modules
import pandas as pd
import numpy as np
 
# create dataset
df = pd.DataFrame({'Car Model': ['BMW X5', 'Audi A5', 'Mercedez Benz C class']})
 
# display dataset
print(df)
 
# create dummy variables
pd.get_dummies(df)