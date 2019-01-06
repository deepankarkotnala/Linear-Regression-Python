# Simple Linear Regression Using Python

### Importing the required libraries
```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
```
### Reading Data
```python
data = pd.read_csv('D:\iiitB\Python\Linear Regression Python\headbrain.csv')
print(data.shape)
data.head()
```python
![data](https://github.com/deepankarkotnala/LinearRegressionPython/blob/master/data.PNG)


### Collecting X and Y items. By using the values of X, we will predict the value of Y.
```python
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
```
# Equation of a line is given as: Y = mX + c
# In order to find the values of m and c, we first need to calculate the mean of X and Y
```python
mean_x = np.mean(X)
mean_y = np.mean(Y)
```
### Total number of values
```python
n = len(X)
```
### Calculating the value of m and c using the formula
```txt
# Formula : m = [(x - x_bar)* (y - y_bar)] / [ (x - x_bar)^2 ]
# y_bar = (m * x_bar) + c 
# From above equation, we can calculate the value of c as:
# c = y_bar - (m * x_bar)
# where x_bar = mean_x, and y_bar = mean_y
```

```python
numerator = 0
denominator = 0
for i in range(n):
    numerator   += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2
m = numerator / denominator
c = mean_y - (m * mean_x)
```

### Print the coefficients
```python
print('Value of m: ',m)
print('Value of c: ',c)
```
### The value of m and c obtained here will be added to the following equation:
#### BrainWeight = c + m * HeadSize


### Plotting the Linear Regression Line
Now we have the equation of the line. Using this equation, we will find the predicted values of y.
Once we get all the points, we can plot them and create the Linear Regression Line

### Plotting Values and Linear Regression Line
```python
max_x = np.max(X) + 100
min_x = np.min(X) - 100
```
### Calculating line values x and y
```python
x = np.linspace(min_x, max_x, 1000)
y = c + m * x 
``` 
### Ploting Line
```python
plt.plot(x, y, color='#52b920', label='Regression Line')
```
### Ploting Scatter Points
```python
plt.scatter(X, Y, c='#ef4423', label='Scatter Plot') 
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()
```

![graph](https://github.com/deepankarkotnala/LinearRegressionPython/blob/master/graph.PNG)


### tot_sumofsq is the total sum of squares and res_sumofsq is the total sum of squares of residuals(relate them to the formula).
```python
tot_sumofsq = 0
res_sumofsq = 0
for i in range(n): #n is the total number of values
    y_pred = c + m * X[i]
    tot_sumofsq += (Y[i] - mean_y) ** 2
    res_sumofsq += (Y[i] - y_pred) ** 2
    r2 = 1 - (res_sumofsq/tot_sumofsq)
print(r2)
```

