import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('coords.csv')

# df.head()
# df.tail()

print(df[df['class'] == 'Sleepy'])

'''
defide target and feature so we can see the relationship between target and their feature 
target = the objec that will be predict
feature =  coordinate value that will be use to predict the object
'''
x= df.drop('class', axis=1)#feature
y=df['class']#target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)