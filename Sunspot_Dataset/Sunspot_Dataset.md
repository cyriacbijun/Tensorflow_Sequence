# Predicting on Real world Sunspot Dataset

This blog is to demonstrate how to use tensorflow to predict probability of occurence of sunspot

You can use the [.ipnyb notebook](https://github.com/cyriacbijun/Tensorflow_NLP/blob/master/Sunspot_Dataset/Sunspot_Dataset.ipynb) that is given with the repo by downloading and starting a kernel.


*   For local computer, use [jupyter notebook](https://jupyter.org/install)
*   For cloud usage, checkout [Google colab](https://colab.research.google.com/notebooks/intro.ipynb)

```python
import tensorflow as tf
print(tf.__version__)
```

    2.2.0-rc3



```python
import numpy as np
import matplotlib.pyplot as plt
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
```

First, Downloading the data..


```python
!wget --no-check-certificate \
    https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv \
    -O daily-min-temperatures.csv
```

    --2020-05-01 11:07:07--  https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 67921 (66K) [text/plain]
    Saving to: ‘daily-min-temperatures.csv’

    daily-min-temperatu 100%[===================>]  66.33K  --.-KB/s    in 0.008s  

    2020-05-01 11:07:08 (7.69 MB/s) - ‘daily-min-temperatures.csv’ saved [67921/67921]



Now that we downloaded the sunspot data, let us understand the structure of the .csv file using pandas. This step is not really necessary.


```python
import pandas as pd
data = pd.read_csv('daily-min-temperatures.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1981-01-01</td>
      <td>20.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981-01-02</td>
      <td>17.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1981-01-03</td>
      <td>18.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981-01-04</td>
      <td>14.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981-01-05</td>
      <td>15.8</td>
    </tr>
  </tbody>
</table>
</div>



We do not need the first row,that's why we have `next(reader)`. In the for loop we are appending the temp values only, as float. After that we plot to see the series.


```python
import csv
time_step = []
temps = []

with open('daily-min-temperatures.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  next(reader)
  step=0
  for row in reader:
    temps.append(float(row[1]))
    time_step.append(step)
    step = step + 1

series = np.array(temps)
time = np.array(time_step)
plt.figure(figsize=(10, 6))
plot_series(time, series)
```


![png](output_7_0.png)



```python
len(temps)
```




    3650



We are splitting the train and validation data as 2500 and 1150.


```python
split_time = 2500
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer_size = 1000

```

Below function `windowed_dataset` is used to create dataset usin `tf.data.Dataset` format. The `model_forecast` function is used to predict the values from the series provided. It also accepts the trained model as an arguement.


```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
```


```python
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
```

As we did in the previous blogs,after defining model structure, we are using `tf.keras.callbacks.LearningRateScheduler` to dynamically change the learning rate upon epochs. This will enable us to find the optimum learning rate.

Our model is a combination of Convolution and 2 LSTMs, both of which are bidirectional and having `return_sequences` as true. This improves the model `mae` but we have to check hoe accurate it will be after prediction and checking with validation set.


```python
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
window_size = 64
batch_size = 256
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(train_set)
print(x_train.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

```

    <PrefetchDataset shapes: ((None, None, 1), (None, None, 1)), types: (tf.float64, tf.float64)>
    (2500,)
    Epoch 1/100
    10/10 [==============================] - 0s 31ms/step - loss: 31.1706 - mae: 31.6550 - lr: 1.0000e-08
    Epoch 2/100
    10/10 [==============================] - 0s 23ms/step - loss: 30.5228 - mae: 31.0756 - lr: 1.1220e-08
    Epoch 3/100
    10/10 [==============================] - 0s 23ms/step - loss: 29.6681 - mae: 30.1801 - lr: 1.2589e-08
    Epoch 4/100
    10/10 [==============================] - 0s 24ms/step - loss: 28.6311 - mae: 29.0586 - lr: 1.4125e-08
    Epoch 5/100
    10/10 [==============================] - 0s 24ms/step - loss: 27.1851 - mae: 27.6945 - lr: 1.5849e-08
    Epoch 6/100
    10/10 [==============================] - 0s 24ms/step - loss: 25.5238 - mae: 25.9986 - lr: 1.7783e-08
    Epoch 7/100
    10/10 [==============================] - 0s 24ms/step - loss: 23.2871 - mae: 23.8429 - lr: 1.9953e-08
    Epoch 8/100
    10/10 [==============================] - 0s 24ms/step - loss: 20.5082 - mae: 21.1108 - lr: 2.2387e-08
    Epoch 9/100
    10/10 [==============================] - 0s 25ms/step - loss: 17.1935 - mae: 17.8091 - lr: 2.5119e-08
    Epoch 10/100
    10/10 [==============================] - 0s 23ms/step - loss: 13.4642 - mae: 14.1371 - lr: 2.8184e-08
    Epoch 11/100
    10/10 [==============================] - 0s 25ms/step - loss: 10.0364 - mae: 10.6152 - lr: 3.1623e-08
    Epoch 12/100
    10/10 [==============================] - 0s 23ms/step - loss: 7.5764 - mae: 8.1025 - lr: 3.5481e-08
    Epoch 13/100
    10/10 [==============================] - 0s 24ms/step - loss: 6.2482 - mae: 6.7711 - lr: 3.9811e-08
    Epoch 14/100
    10/10 [==============================] - 0s 24ms/step - loss: 5.6777 - mae: 6.1856 - lr: 4.4668e-08
    Epoch 15/100
    10/10 [==============================] - 0s 24ms/step - loss: 5.3116 - mae: 5.8166 - lr: 5.0119e-08
    Epoch 16/100
    10/10 [==============================] - 0s 25ms/step - loss: 4.9275 - mae: 5.4206 - lr: 5.6234e-08
    Epoch 17/100
    10/10 [==============================] - 0s 24ms/step - loss: 4.5450 - mae: 5.0338 - lr: 6.3096e-08
    Epoch 18/100
    10/10 [==============================] - 0s 24ms/step - loss: 4.2185 - mae: 4.7085 - lr: 7.0795e-08
    Epoch 19/100
    10/10 [==============================] - 0s 24ms/step - loss: 3.9399 - mae: 4.4360 - lr: 7.9433e-08
    Epoch 20/100
    10/10 [==============================] - 0s 23ms/step - loss: 3.7303 - mae: 4.2177 - lr: 8.9125e-08
    Epoch 21/100
    10/10 [==============================] - 0s 23ms/step - loss: 3.5704 - mae: 4.0566 - lr: 1.0000e-07
    Epoch 22/100
    10/10 [==============================] - 0s 24ms/step - loss: 3.4603 - mae: 3.9344 - lr: 1.1220e-07
    Epoch 23/100
    10/10 [==============================] - 0s 25ms/step - loss: 3.3656 - mae: 3.8414 - lr: 1.2589e-07
    Epoch 24/100
    10/10 [==============================] - 0s 25ms/step - loss: 3.2869 - mae: 3.7645 - lr: 1.4125e-07
    Epoch 25/100
    10/10 [==============================] - 0s 24ms/step - loss: 3.2232 - mae: 3.6978 - lr: 1.5849e-07
    Epoch 26/100
    10/10 [==============================] - 0s 23ms/step - loss: 3.1540 - mae: 3.6346 - lr: 1.7783e-07
    Epoch 27/100
    10/10 [==============================] - 0s 26ms/step - loss: 3.1000 - mae: 3.5693 - lr: 1.9953e-07
    Epoch 28/100
    10/10 [==============================] - 0s 25ms/step - loss: 3.0355 - mae: 3.5053 - lr: 2.2387e-07
    Epoch 29/100
    10/10 [==============================] - 0s 25ms/step - loss: 2.9662 - mae: 3.4379 - lr: 2.5119e-07
    Epoch 30/100
    10/10 [==============================] - 0s 23ms/step - loss: 2.9018 - mae: 3.3712 - lr: 2.8184e-07
    Epoch 31/100
    10/10 [==============================] - 0s 25ms/step - loss: 2.8406 - mae: 3.3098 - lr: 3.1623e-07
    Epoch 32/100
    10/10 [==============================] - 0s 25ms/step - loss: 2.7814 - mae: 3.2479 - lr: 3.5481e-07
    Epoch 33/100
    10/10 [==============================] - 0s 23ms/step - loss: 2.7205 - mae: 3.1891 - lr: 3.9811e-07
    Epoch 34/100
    10/10 [==============================] - 0s 23ms/step - loss: 2.6695 - mae: 3.1379 - lr: 4.4668e-07
    Epoch 35/100
    10/10 [==============================] - 0s 24ms/step - loss: 2.6166 - mae: 3.0848 - lr: 5.0119e-07
    Epoch 36/100
    10/10 [==============================] - 0s 23ms/step - loss: 2.5696 - mae: 3.0357 - lr: 5.6234e-07
    Epoch 37/100
    10/10 [==============================] - 0s 27ms/step - loss: 2.5223 - mae: 2.9877 - lr: 6.3096e-07
    Epoch 38/100
    10/10 [==============================] - 0s 24ms/step - loss: 2.4715 - mae: 2.9415 - lr: 7.0795e-07
    Epoch 39/100
    10/10 [==============================] - 0s 26ms/step - loss: 2.4295 - mae: 2.8968 - lr: 7.9433e-07
    Epoch 40/100
    10/10 [==============================] - 0s 25ms/step - loss: 2.3863 - mae: 2.8541 - lr: 8.9125e-07
    Epoch 41/100
    10/10 [==============================] - 0s 24ms/step - loss: 2.3447 - mae: 2.8124 - lr: 1.0000e-06
    Epoch 42/100
    10/10 [==============================] - 0s 25ms/step - loss: 2.3051 - mae: 2.7718 - lr: 1.1220e-06
    Epoch 43/100
    10/10 [==============================] - 0s 24ms/step - loss: 2.2710 - mae: 2.7315 - lr: 1.2589e-06
    Epoch 44/100
    10/10 [==============================] - 0s 24ms/step - loss: 2.2294 - mae: 2.6913 - lr: 1.4125e-06
    Epoch 45/100
    10/10 [==============================] - 0s 23ms/step - loss: 2.1910 - mae: 2.6523 - lr: 1.5849e-06
    Epoch 46/100
    10/10 [==============================] - 0s 23ms/step - loss: 2.1624 - mae: 2.6190 - lr: 1.7783e-06
    Epoch 47/100
    10/10 [==============================] - 0s 23ms/step - loss: 2.1291 - mae: 2.5877 - lr: 1.9953e-06
    Epoch 48/100
    10/10 [==============================] - 0s 24ms/step - loss: 2.0990 - mae: 2.5607 - lr: 2.2387e-06
    Epoch 49/100
    10/10 [==============================] - 0s 23ms/step - loss: 2.0698 - mae: 2.5300 - lr: 2.5119e-06
    Epoch 50/100
    10/10 [==============================] - 0s 24ms/step - loss: 2.0455 - mae: 2.5040 - lr: 2.8184e-06
    Epoch 51/100
    10/10 [==============================] - 0s 24ms/step - loss: 2.0226 - mae: 2.4818 - lr: 3.1623e-06
    Epoch 52/100
    10/10 [==============================] - 0s 23ms/step - loss: 2.0002 - mae: 2.4598 - lr: 3.5481e-06
    Epoch 53/100
    10/10 [==============================] - 0s 23ms/step - loss: 1.9872 - mae: 2.4443 - lr: 3.9811e-06
    Epoch 54/100
    10/10 [==============================] - 0s 23ms/step - loss: 1.9620 - mae: 2.4193 - lr: 4.4668e-06
    Epoch 55/100
    10/10 [==============================] - 0s 23ms/step - loss: 1.9474 - mae: 2.4008 - lr: 5.0119e-06
    Epoch 56/100
    10/10 [==============================] - 0s 24ms/step - loss: 1.9220 - mae: 2.3783 - lr: 5.6234e-06
    Epoch 57/100
    10/10 [==============================] - 0s 24ms/step - loss: 1.9025 - mae: 2.3620 - lr: 6.3096e-06
    Epoch 58/100
    10/10 [==============================] - 0s 23ms/step - loss: 1.8872 - mae: 2.3394 - lr: 7.0795e-06
    Epoch 59/100
    10/10 [==============================] - 0s 26ms/step - loss: 1.8568 - mae: 2.3121 - lr: 7.9433e-06
    Epoch 60/100
    10/10 [==============================] - 0s 26ms/step - loss: 2.1811 - mae: 2.6562 - lr: 8.9125e-06
    Epoch 61/100
    10/10 [==============================] - 0s 25ms/step - loss: 2.6798 - mae: 3.1553 - lr: 1.0000e-05
    Epoch 62/100
    10/10 [==============================] - 0s 23ms/step - loss: 3.0776 - mae: 3.5613 - lr: 1.1220e-05
    Epoch 63/100
    10/10 [==============================] - 0s 22ms/step - loss: 3.5627 - mae: 3.9954 - lr: 1.2589e-05
    Epoch 64/100
    10/10 [==============================] - 0s 24ms/step - loss: 3.6635 - mae: 4.1405 - lr: 1.4125e-05
    Epoch 65/100
    10/10 [==============================] - 0s 23ms/step - loss: 4.1839 - mae: 4.6922 - lr: 1.5849e-05
    Epoch 66/100
    10/10 [==============================] - 0s 23ms/step - loss: 4.3617 - mae: 4.8699 - lr: 1.7783e-05
    Epoch 67/100
    10/10 [==============================] - 0s 26ms/step - loss: 4.6143 - mae: 5.1154 - lr: 1.9953e-05
    Epoch 68/100
    10/10 [==============================] - 0s 23ms/step - loss: 4.6684 - mae: 5.1861 - lr: 2.2387e-05
    Epoch 69/100
    10/10 [==============================] - 0s 24ms/step - loss: 4.8741 - mae: 5.4719 - lr: 2.5119e-05
    Epoch 70/100
    10/10 [==============================] - 0s 24ms/step - loss: 5.0190 - mae: 5.5396 - lr: 2.8184e-05
    Epoch 71/100
    10/10 [==============================] - 0s 23ms/step - loss: 5.6000 - mae: 6.0736 - lr: 3.1623e-05
    Epoch 72/100
    10/10 [==============================] - 0s 24ms/step - loss: 5.4061 - mae: 5.9323 - lr: 3.5481e-05
    Epoch 73/100
    10/10 [==============================] - 0s 24ms/step - loss: 4.5663 - mae: 5.0131 - lr: 3.9811e-05
    Epoch 74/100
    10/10 [==============================] - 0s 24ms/step - loss: 4.2385 - mae: 4.7400 - lr: 4.4668e-05
    Epoch 75/100
    10/10 [==============================] - 0s 26ms/step - loss: 3.8110 - mae: 4.2483 - lr: 5.0119e-05
    Epoch 76/100
    10/10 [==============================] - 0s 25ms/step - loss: 4.0372 - mae: 4.5317 - lr: 5.6234e-05
    Epoch 77/100
    10/10 [==============================] - 0s 25ms/step - loss: 4.0612 - mae: 4.5477 - lr: 6.3096e-05
    Epoch 78/100
    10/10 [==============================] - 0s 24ms/step - loss: 4.0974 - mae: 4.6127 - lr: 7.0795e-05
    Epoch 79/100
    10/10 [==============================] - 0s 25ms/step - loss: 2.9855 - mae: 3.4656 - lr: 7.9433e-05
    Epoch 80/100
    10/10 [==============================] - 0s 25ms/step - loss: 3.3014 - mae: 3.8295 - lr: 8.9125e-05
    Epoch 81/100
    10/10 [==============================] - 0s 26ms/step - loss: 3.2526 - mae: 3.7488 - lr: 1.0000e-04
    Epoch 82/100
    10/10 [==============================] - 0s 25ms/step - loss: 3.9145 - mae: 4.4089 - lr: 1.1220e-04
    Epoch 83/100
    10/10 [==============================] - 0s 24ms/step - loss: 4.4860 - mae: 4.9016 - lr: 1.2589e-04
    Epoch 84/100
    10/10 [==============================] - 0s 23ms/step - loss: 4.9954 - mae: 5.4146 - lr: 1.4125e-04
    Epoch 85/100
    10/10 [==============================] - 0s 27ms/step - loss: 5.8608 - mae: 6.4323 - lr: 1.5849e-04
    Epoch 86/100
    10/10 [==============================] - 0s 24ms/step - loss: 6.1167 - mae: 6.7497 - lr: 1.7783e-04
    Epoch 87/100
    10/10 [==============================] - 0s 26ms/step - loss: 5.7634 - mae: 6.1832 - lr: 1.9953e-04
    Epoch 88/100
    10/10 [==============================] - 0s 25ms/step - loss: 18.4549 - mae: 18.5721 - lr: 2.2387e-04
    Epoch 89/100
    10/10 [==============================] - 0s 25ms/step - loss: 19.1328 - mae: 19.9009 - lr: 2.5119e-04
    Epoch 90/100
    10/10 [==============================] - 0s 26ms/step - loss: 26.0024 - mae: 25.3163 - lr: 2.8184e-04
    Epoch 91/100
    10/10 [==============================] - 0s 26ms/step - loss: 39.1242 - mae: 41.0090 - lr: 3.1623e-04
    Epoch 92/100
    10/10 [==============================] - 0s 24ms/step - loss: 29.0936 - mae: 29.5980 - lr: 3.5481e-04
    Epoch 93/100
    10/10 [==============================] - 0s 24ms/step - loss: 33.0701 - mae: 32.9219 - lr: 3.9811e-04
    Epoch 94/100
    10/10 [==============================] - 0s 26ms/step - loss: 36.9112 - mae: 37.8804 - lr: 4.4668e-04
    Epoch 95/100
    10/10 [==============================] - 0s 27ms/step - loss: 41.6311 - mae: 41.2765 - lr: 5.0119e-04
    Epoch 96/100
    10/10 [==============================] - 0s 25ms/step - loss: 46.5597 - mae: 47.5416 - lr: 5.6234e-04
    Epoch 97/100
    10/10 [==============================] - 0s 24ms/step - loss: 52.5361 - mae: 52.1260 - lr: 6.3096e-04
    Epoch 98/100
    10/10 [==============================] - 0s 25ms/step - loss: 58.8236 - mae: 59.7652 - lr: 7.0795e-04
    Epoch 99/100
    10/10 [==============================] - 0s 25ms/step - loss: 66.3937 - mae: 65.8950 - lr: 7.9433e-04
    Epoch 100/100
    10/10 [==============================] - 0s 26ms/step - loss: 74.3771 - mae: 75.2363 - lr: 8.9125e-04



```python
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 60])
```




    (1e-08, 0.0001, 0.0, 60.0)




![png](output_16_1.png)


After plotting the graph we see that the loss is really low and stable for the learning_rate = $10^{-5}$. Now,we train using that learning rate.


```python
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
train_set = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])


optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set,epochs=150)
```

    Epoch 1/150
    25/25 [==============================] - 0s 14ms/step - loss: 9.8211 - mae: 10.4694
    Epoch 2/150
    25/25 [==============================] - 0s 13ms/step - loss: 2.5191 - mae: 2.9923
    Epoch 3/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.9513 - mae: 2.4047
    Epoch 4/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.8610 - mae: 2.3151
    Epoch 5/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.8209 - mae: 2.2733
    Epoch 6/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.7905 - mae: 2.2418
    Epoch 7/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.7667 - mae: 2.2186
    Epoch 8/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.7387 - mae: 2.1906
    Epoch 9/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.7173 - mae: 2.1681
    Epoch 10/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.6987 - mae: 2.1482
    Epoch 11/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.6815 - mae: 2.1287
    Epoch 12/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.6685 - mae: 2.1159
    Epoch 13/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.6576 - mae: 2.1030
    Epoch 14/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.6430 - mae: 2.0891
    Epoch 15/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.6353 - mae: 2.0803
    Epoch 16/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.6261 - mae: 2.0710
    Epoch 17/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.6115 - mae: 2.0581
    Epoch 18/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.6095 - mae: 2.0553
    Epoch 19/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5999 - mae: 2.0444
    Epoch 20/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5926 - mae: 2.0365
    Epoch 21/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5908 - mae: 2.0348
    Epoch 22/150
    25/25 [==============================] - 0s 12ms/step - loss: 1.5844 - mae: 2.0276
    Epoch 23/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5807 - mae: 2.0232
    Epoch 24/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5768 - mae: 2.0222
    Epoch 25/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5698 - mae: 2.0120
    Epoch 26/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5637 - mae: 2.0085
    Epoch 27/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5641 - mae: 2.0057
    Epoch 28/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5579 - mae: 2.0014
    Epoch 29/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5549 - mae: 1.9991
    Epoch 30/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5540 - mae: 1.9972
    Epoch 31/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5538 - mae: 1.9967
    Epoch 32/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5524 - mae: 1.9937
    Epoch 33/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5457 - mae: 1.9897
    Epoch 34/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5513 - mae: 1.9939
    Epoch 35/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5474 - mae: 1.9909
    Epoch 36/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5412 - mae: 1.9846
    Epoch 37/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5417 - mae: 1.9836
    Epoch 38/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5404 - mae: 1.9816
    Epoch 39/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5388 - mae: 1.9822
    Epoch 40/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5403 - mae: 1.9829
    Epoch 41/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5337 - mae: 1.9768
    Epoch 42/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5355 - mae: 1.9770
    Epoch 43/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5337 - mae: 1.9768
    Epoch 44/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5366 - mae: 1.9780
    Epoch 45/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5292 - mae: 1.9723
    Epoch 46/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5328 - mae: 1.9748
    Epoch 47/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5282 - mae: 1.9699
    Epoch 48/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5311 - mae: 1.9730
    Epoch 49/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5290 - mae: 1.9724
    Epoch 50/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5247 - mae: 1.9678
    Epoch 51/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5281 - mae: 1.9708
    Epoch 52/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5263 - mae: 1.9692
    Epoch 53/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5287 - mae: 1.9708
    Epoch 54/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5250 - mae: 1.9656
    Epoch 55/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5232 - mae: 1.9654
    Epoch 56/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5211 - mae: 1.9629
    Epoch 57/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5191 - mae: 1.9635
    Epoch 58/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5340 - mae: 1.9776
    Epoch 59/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5233 - mae: 1.9669
    Epoch 60/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5237 - mae: 1.9638
    Epoch 61/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5177 - mae: 1.9600
    Epoch 62/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5201 - mae: 1.9612
    Epoch 63/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5145 - mae: 1.9584
    Epoch 64/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5180 - mae: 1.9589
    Epoch 65/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5152 - mae: 1.9581
    Epoch 66/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5176 - mae: 1.9585
    Epoch 67/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5227 - mae: 1.9635
    Epoch 68/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5158 - mae: 1.9579
    Epoch 69/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5119 - mae: 1.9561
    Epoch 70/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5140 - mae: 1.9559
    Epoch 71/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5121 - mae: 1.9542
    Epoch 72/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5133 - mae: 1.9537
    Epoch 73/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5139 - mae: 1.9569
    Epoch 74/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5125 - mae: 1.9527
    Epoch 75/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5114 - mae: 1.9542
    Epoch 76/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5116 - mae: 1.9527
    Epoch 77/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5096 - mae: 1.9520
    Epoch 78/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5137 - mae: 1.9559
    Epoch 79/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5082 - mae: 1.9505
    Epoch 80/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5103 - mae: 1.9520
    Epoch 81/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5143 - mae: 1.9565
    Epoch 82/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5063 - mae: 1.9490
    Epoch 83/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5078 - mae: 1.9494
    Epoch 84/150
    25/25 [==============================] - 0s 12ms/step - loss: 1.5068 - mae: 1.9501
    Epoch 85/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5061 - mae: 1.9484
    Epoch 86/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5057 - mae: 1.9475
    Epoch 87/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5075 - mae: 1.9489
    Epoch 88/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5035 - mae: 1.9467
    Epoch 89/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5054 - mae: 1.9463
    Epoch 90/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5052 - mae: 1.9479
    Epoch 91/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5045 - mae: 1.9468
    Epoch 92/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5034 - mae: 1.9459
    Epoch 93/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5025 - mae: 1.9445
    Epoch 94/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5010 - mae: 1.9437
    Epoch 95/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5092 - mae: 1.9529
    Epoch 96/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5012 - mae: 1.9448
    Epoch 97/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5042 - mae: 1.9469
    Epoch 98/150
    25/25 [==============================] - 0s 12ms/step - loss: 1.5051 - mae: 1.9469
    Epoch 99/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5012 - mae: 1.9426
    Epoch 100/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4997 - mae: 1.9425
    Epoch 101/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5004 - mae: 1.9427
    Epoch 102/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4979 - mae: 1.9410
    Epoch 103/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4986 - mae: 1.9412
    Epoch 104/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4992 - mae: 1.9418
    Epoch 105/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4987 - mae: 1.9410
    Epoch 106/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4996 - mae: 1.9411
    Epoch 107/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5003 - mae: 1.9429
    Epoch 108/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.5015 - mae: 1.9414
    Epoch 109/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4990 - mae: 1.9413
    Epoch 110/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5006 - mae: 1.9427
    Epoch 111/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4968 - mae: 1.9401
    Epoch 112/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4972 - mae: 1.9388
    Epoch 113/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4967 - mae: 1.9378
    Epoch 114/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4955 - mae: 1.9380
    Epoch 115/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4941 - mae: 1.9377
    Epoch 116/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4978 - mae: 1.9391
    Epoch 117/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4982 - mae: 1.9377
    Epoch 118/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.4965 - mae: 1.9383
    Epoch 119/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4951 - mae: 1.9370
    Epoch 120/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4952 - mae: 1.9381
    Epoch 121/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5003 - mae: 1.9426
    Epoch 122/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.4956 - mae: 1.9375
    Epoch 123/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.5006 - mae: 1.9424
    Epoch 124/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.4941 - mae: 1.9369
    Epoch 125/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4950 - mae: 1.9366
    Epoch 126/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4938 - mae: 1.9359
    Epoch 127/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4936 - mae: 1.9353
    Epoch 128/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.4932 - mae: 1.9349
    Epoch 129/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4938 - mae: 1.9336
    Epoch 130/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.4965 - mae: 1.9373
    Epoch 131/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.4919 - mae: 1.9335
    Epoch 132/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4985 - mae: 1.9414
    Epoch 133/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4955 - mae: 1.9379
    Epoch 134/150
    25/25 [==============================] - 0s 12ms/step - loss: 1.4956 - mae: 1.9382
    Epoch 135/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4920 - mae: 1.9322
    Epoch 136/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4917 - mae: 1.9333
    Epoch 137/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4946 - mae: 1.9363
    Epoch 138/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4904 - mae: 1.9329
    Epoch 139/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4885 - mae: 1.9321
    Epoch 140/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4897 - mae: 1.9325
    Epoch 141/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4902 - mae: 1.9311
    Epoch 142/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4900 - mae: 1.9304
    Epoch 143/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4916 - mae: 1.9333
    Epoch 144/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.4909 - mae: 1.9326
    Epoch 145/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4887 - mae: 1.9299
    Epoch 146/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4943 - mae: 1.9364
    Epoch 147/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4889 - mae: 1.9306
    Epoch 148/150
    25/25 [==============================] - 0s 14ms/step - loss: 1.4882 - mae: 1.9300
    Epoch 149/150
    25/25 [==============================] - 0s 16ms/step - loss: 1.4900 - mae: 1.9307
    Epoch 150/150
    25/25 [==============================] - 0s 13ms/step - loss: 1.4886 - mae: 1.9292



```python
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
```

We are removing the series data at time before split_time - window_size and then taking the data all the way to the end. After that, we plot to see the validation vs prediction.


```python
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
```


![png](output_21_0.png)



```python
tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
```




    1.7796258



The mean absolute error is really low compared to the other models we have tried out before. The next statement simply prints the forecasted values.


```python
print(rnn_forecast)
```

    [11.329356 10.705612 12.124963 ... 13.604561 13.796917 15.009445]
