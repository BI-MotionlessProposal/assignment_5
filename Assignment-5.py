import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# part 1, split the data
df = pd.read_json('users.json')
df = df[np.isfinite(df['karma'])]
df = df[['created', 'karma']]
today = pd.to_datetime('today')
df['created'] = (today - pd.to_datetime(df['created'],
                                        unit='s')).dt.days  # converting unix time to human readable format and counting how many days passed til today
df['created'] = df['created'].values.reshape(-1, 1)
df['karma'] = df['karma'].values.reshape(-1, 1)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

trainX = train[['created']]
trainY = train[['karma']]
trainX = np.array(trainX)
trainY = np.array(trainY)

testX = test[['created']]
testY = test[['karma']]
testX = np.array(testX)
testY = np.array(testY)

model = linear_model.LinearRegression()
model.fit(trainX, trainY)

predictTest = model.predict(testX)
predictTrain = model.predict(trainX)

print(predictTest)
plt.xlabel('created before x days')
plt.ylabel('Karma')
plt.scatter(testX, testY)
plt.plot(testX, predictTest, color='black', linewidth=1)
plt.show()
# part 1
# For the linear regression y=ax+b we calculate the a in the formula with model.coef_ and the b with model.intercept_
# For the training data we get the following values for a and b in the formula y=ax +b:

print('coef: ', str(model.coef_))
print('intercept', str(model.intercept_))

# part 2
print("mean_absolute_error training data: ", str(mean_absolute_error(trainY, predictTrain)))
print("mean_absolute_error testing data: ", str(mean_absolute_error(testY, predictTest)))
# The results are different because the values we feed to the method are different and not the same volume.
# The result is approximately the same that means the data have uniformly scattered values.
# but in case we had to choose we would get the one from the actual train data as the more the samples the more accurate results.

# part 3
print("mean_squared_error training data: ", str(mean_squared_error(trainY, predictTrain)))
print("mean_squared_error testing data: ", str(mean_squared_error(testX, predictTest)))
# The MSE shows the the quality of an estimator, in our case our trained model
# the closer  it is to 0.0 the better.
# In our case we can see the MSE is pretty high that means our data set is not
# the best for a linear regression. Maybe a Gradient Boosting regression would
# fit our model better. Never the less we can see that the trainning data and the
# test give about the same MSE that means that our test data and train data are
# homogeneous.

# part 4
# Pearson r tells us how good our model perform where 1 is the max and 0 is the min value.
print('pearson for training data: ', str(model.score(trainX, trainY)))
print('pearson for testing data: ', str(model.score(testX, testY)))
# Conclusion the Pearson r is very low which tells us the is not a good linear regression for predicting accurately
# The main difference between Pearson r and MAE and MSE is that it calculates the error without the x.
# It calculates error from the y value to the linear regression.

# part 5
# According to our statistics we can see that our data model is not very consistent,
# mainly because it includes the input data from inactive users as we can see on our scatter plott where most of the users are close to 0 karma points.
# If we were able to select the active users who actually achieved the 1000 points, or more,
# we can assume that our data model would be much more precise and our models would be much more accurate.
