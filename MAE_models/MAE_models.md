# Comparison of Mean Average Errors of different types of Models

In this blog,we compare the MAEs of different models, when we change the layers.

You can use the [.ipnyb notebook](https://github.com/cyriacbijun/Tensorflow_NLP/blob/master/MAE_models/MAE_models.ipynb) that is given with the repo by downloading and starting a kernel.


*   For local computer, use [jupyter notebook](https://jupyter.org/install)
*   For cloud usage, checkout [Google colab](https://colab.research.google.com/notebooks/intro.ipynb)


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

The following cell is simply used to define functions which enables us to create synthetic series, just like the previous exercise.


```python
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(10 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.005
noise_level = 3

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=51)
```

We split the series into train and validation, where the training data is upto 3000 and validation is rest of the data. Then, we plot it.


```python
split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

plot_series(time, series)
```


![png](output_5_0.png)


As seen in the previous exercise, we can create an instance of tensorflow dataset with below code, which has inbuilt batching and shuffling.


```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset
```

Why did we use a Lambda layer? An RNN expects 3 dimensions:
* batch-size
* the number of timestamps
* series dimensionality
So, since we are providing only 2 dimensions using the dataset, we can use a Lambda Layer to increase the dimension of input Layer, to suit the RNN.

`tf.keras.backend.clear_session()` is used so that the backend is cleared and models does not affect each other. Here, you can also see `tf.keras.callbacks.LearningRateScheduler`. The function is ... as the training progresses, the learning rate is changed via the callback function depending on the current epoch. So, inintial `lr` was $10^{-8}$. The as the epoch number changes, `lr` also changes according to the formula : `1e-8 * 10**(epoch / 20)`


```python
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 10.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
```

    Epoch 1/100
    94/94 [==============================] - 3s 29ms/step - loss: 20.4602 - mae: 20.8828 - lr: 1.0000e-08
    Epoch 2/100
    94/94 [==============================] - 3s 28ms/step - loss: 20.3766 - mae: 20.8543 - lr: 1.1220e-08
    Epoch 3/100
    94/94 [==============================] - 3s 29ms/step - loss: 20.3643 - mae: 20.8218 - lr: 1.2589e-08
    Epoch 4/100
    94/94 [==============================] - 3s 29ms/step - loss: 20.3237 - mae: 20.7851 - lr: 1.4125e-08
    Epoch 5/100
    94/94 [==============================] - 3s 28ms/step - loss: 20.2210 - mae: 20.7436 - lr: 1.5849e-08
    Epoch 6/100
    94/94 [==============================] - 3s 28ms/step - loss: 20.1817 - mae: 20.6964 - lr: 1.7783e-08
    Epoch 7/100
    94/94 [==============================] - 3s 28ms/step - loss: 20.1016 - mae: 20.6429 - lr: 1.9953e-08
    Epoch 8/100
    94/94 [==============================] - 3s 29ms/step - loss: 20.0922 - mae: 20.5823 - lr: 2.2387e-08
    Epoch 9/100
    94/94 [==============================] - 3s 28ms/step - loss: 20.0004 - mae: 20.5140 - lr: 2.5119e-08
    Epoch 10/100
    94/94 [==============================] - 3s 28ms/step - loss: 19.9554 - mae: 20.4385 - lr: 2.8184e-08
    Epoch 11/100
    94/94 [==============================] - 3s 28ms/step - loss: 19.9592 - mae: 20.3585 - lr: 3.1623e-08
    Epoch 12/100
    94/94 [==============================] - 3s 28ms/step - loss: 19.8126 - mae: 20.2777 - lr: 3.5481e-08
    Epoch 13/100
    94/94 [==============================] - 3s 29ms/step - loss: 19.7425 - mae: 20.1968 - lr: 3.9811e-08
    Epoch 14/100
    94/94 [==============================] - 3s 28ms/step - loss: 19.5884 - mae: 20.1109 - lr: 4.4668e-08
    Epoch 15/100
    94/94 [==============================] - 3s 28ms/step - loss: 19.5268 - mae: 20.0163 - lr: 5.0119e-08
    Epoch 16/100
    94/94 [==============================] - 3s 28ms/step - loss: 19.4060 - mae: 19.9109 - lr: 5.6234e-08
    Epoch 17/100
    94/94 [==============================] - 3s 28ms/step - loss: 19.2756 - mae: 19.7932 - lr: 6.3096e-08
    Epoch 18/100
    94/94 [==============================] - 3s 29ms/step - loss: 19.1609 - mae: 19.6618 - lr: 7.0795e-08
    Epoch 19/100
    94/94 [==============================] - 3s 29ms/step - loss: 19.0090 - mae: 19.5148 - lr: 7.9433e-08
    Epoch 20/100
    94/94 [==============================] - 3s 29ms/step - loss: 18.8467 - mae: 19.3502 - lr: 8.9125e-08
    Epoch 21/100
    94/94 [==============================] - 3s 28ms/step - loss: 18.7482 - mae: 19.1656 - lr: 1.0000e-07
    Epoch 22/100
    94/94 [==============================] - 3s 28ms/step - loss: 18.4040 - mae: 18.9583 - lr: 1.1220e-07
    Epoch 23/100
    94/94 [==============================] - 3s 29ms/step - loss: 18.3458 - mae: 18.7249 - lr: 1.2589e-07
    Epoch 24/100
    94/94 [==============================] - 3s 29ms/step - loss: 17.9591 - mae: 18.4621 - lr: 1.4125e-07
    Epoch 25/100
    94/94 [==============================] - 3s 28ms/step - loss: 17.7048 - mae: 18.1655 - lr: 1.5849e-07
    Epoch 26/100
    94/94 [==============================] - 3s 28ms/step - loss: 17.3764 - mae: 17.8312 - lr: 1.7783e-07
    Epoch 27/100
    94/94 [==============================] - 3s 28ms/step - loss: 16.9770 - mae: 17.4554 - lr: 1.9953e-07
    Epoch 28/100
    94/94 [==============================] - 3s 28ms/step - loss: 16.5467 - mae: 17.0350 - lr: 2.2387e-07
    Epoch 29/100
    94/94 [==============================] - 3s 28ms/step - loss: 16.0490 - mae: 16.5714 - lr: 2.5119e-07
    Epoch 30/100
    94/94 [==============================] - 3s 28ms/step - loss: 15.5696 - mae: 16.0630 - lr: 2.8184e-07
    Epoch 31/100
    94/94 [==============================] - 3s 28ms/step - loss: 14.9412 - mae: 15.5113 - lr: 3.1623e-07
    Epoch 32/100
    94/94 [==============================] - 3s 28ms/step - loss: 14.4644 - mae: 14.9140 - lr: 3.5481e-07
    Epoch 33/100
    94/94 [==============================] - 3s 29ms/step - loss: 13.7479 - mae: 14.2730 - lr: 3.9811e-07
    Epoch 34/100
    94/94 [==============================] - 3s 30ms/step - loss: 13.0584 - mae: 13.5900 - lr: 4.4668e-07
    Epoch 35/100
    94/94 [==============================] - 3s 29ms/step - loss: 12.4185 - mae: 12.8770 - lr: 5.0119e-07
    Epoch 36/100
    94/94 [==============================] - 3s 29ms/step - loss: 11.7088 - mae: 12.1503 - lr: 5.6234e-07
    Epoch 37/100
    94/94 [==============================] - 3s 28ms/step - loss: 10.9549 - mae: 11.4246 - lr: 6.3096e-07
    Epoch 38/100
    94/94 [==============================] - 3s 29ms/step - loss: 10.1918 - mae: 10.7178 - lr: 7.0795e-07
    Epoch 39/100
    94/94 [==============================] - 3s 29ms/step - loss: 9.5596 - mae: 10.0572 - lr: 7.9433e-07
    Epoch 40/100
    94/94 [==============================] - 3s 28ms/step - loss: 8.9681 - mae: 9.4767 - lr: 8.9125e-07
    Epoch 41/100
    94/94 [==============================] - 3s 29ms/step - loss: 8.4933 - mae: 8.9762 - lr: 1.0000e-06
    Epoch 42/100
    94/94 [==============================] - 3s 28ms/step - loss: 8.0897 - mae: 8.5559 - lr: 1.1220e-06
    Epoch 43/100
    94/94 [==============================] - 3s 28ms/step - loss: 7.7067 - mae: 8.2186 - lr: 1.2589e-06
    Epoch 44/100
    94/94 [==============================] - 3s 28ms/step - loss: 7.6117 - mae: 7.9611 - lr: 1.4125e-06
    Epoch 45/100
    94/94 [==============================] - 3s 29ms/step - loss: 7.2321 - mae: 7.7611 - lr: 1.5849e-06
    Epoch 46/100
    94/94 [==============================] - 3s 29ms/step - loss: 7.1898 - mae: 7.6089 - lr: 1.7783e-06
    Epoch 47/100
    94/94 [==============================] - 3s 28ms/step - loss: 6.9884 - mae: 7.4794 - lr: 1.9953e-06
    Epoch 48/100
    94/94 [==============================] - 3s 28ms/step - loss: 6.9385 - mae: 7.3684 - lr: 2.2387e-06
    Epoch 49/100
    94/94 [==============================] - 3s 28ms/step - loss: 6.7348 - mae: 7.2586 - lr: 2.5119e-06
    Epoch 50/100
    94/94 [==============================] - 3s 29ms/step - loss: 6.6561 - mae: 7.1513 - lr: 2.8184e-06
    Epoch 51/100
    94/94 [==============================] - 3s 28ms/step - loss: 6.5186 - mae: 7.0292 - lr: 3.1623e-06
    Epoch 52/100
    94/94 [==============================] - 3s 29ms/step - loss: 6.2941 - mae: 6.8203 - lr: 3.5481e-06
    Epoch 53/100
    94/94 [==============================] - 3s 28ms/step - loss: 5.9168 - mae: 6.3736 - lr: 3.9811e-06
    Epoch 54/100
    94/94 [==============================] - 3s 29ms/step - loss: 5.7016 - mae: 6.1635 - lr: 4.4668e-06
    Epoch 55/100
    94/94 [==============================] - 3s 29ms/step - loss: 5.5810 - mae: 5.9845 - lr: 5.0119e-06
    Epoch 56/100
    94/94 [==============================] - 3s 28ms/step - loss: 5.2817 - mae: 5.7677 - lr: 5.6234e-06
    Epoch 57/100
    94/94 [==============================] - 3s 28ms/step - loss: 5.1438 - mae: 5.6485 - lr: 6.3096e-06
    Epoch 58/100
    94/94 [==============================] - 3s 29ms/step - loss: 5.1344 - mae: 5.5023 - lr: 7.0795e-06
    Epoch 59/100
    94/94 [==============================] - 3s 29ms/step - loss: 4.8486 - mae: 5.3295 - lr: 7.9433e-06
    Epoch 60/100
    94/94 [==============================] - 3s 29ms/step - loss: 4.7960 - mae: 5.2320 - lr: 8.9125e-06
    Epoch 61/100
    94/94 [==============================] - 3s 29ms/step - loss: 4.6901 - mae: 5.1469 - lr: 1.0000e-05
    Epoch 62/100
    94/94 [==============================] - 3s 28ms/step - loss: 4.5663 - mae: 5.0500 - lr: 1.1220e-05
    Epoch 63/100
    94/94 [==============================] - 3s 29ms/step - loss: 4.5622 - mae: 5.0578 - lr: 1.2589e-05
    Epoch 64/100
    94/94 [==============================] - 3s 28ms/step - loss: 4.5000 - mae: 5.0037 - lr: 1.4125e-05
    Epoch 65/100
    94/94 [==============================] - 3s 29ms/step - loss: 4.5189 - mae: 5.0130 - lr: 1.5849e-05
    Epoch 66/100
    94/94 [==============================] - 3s 28ms/step - loss: 4.3868 - mae: 4.8267 - lr: 1.7783e-05
    Epoch 67/100
    94/94 [==============================] - 3s 29ms/step - loss: 4.1656 - mae: 4.6171 - lr: 1.9953e-05
    Epoch 68/100
    94/94 [==============================] - 3s 29ms/step - loss: 4.0647 - mae: 4.5466 - lr: 2.2387e-05
    Epoch 69/100
    94/94 [==============================] - 3s 29ms/step - loss: 4.1426 - mae: 4.6046 - lr: 2.5119e-05
    Epoch 70/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.9723 - mae: 4.4540 - lr: 2.8184e-05
    Epoch 71/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.8489 - mae: 4.3447 - lr: 3.1623e-05
    Epoch 72/100
    94/94 [==============================] - 3s 29ms/step - loss: 3.8480 - mae: 4.3093 - lr: 3.5481e-05
    Epoch 73/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.8283 - mae: 4.3224 - lr: 3.9811e-05
    Epoch 74/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.7423 - mae: 4.1808 - lr: 4.4668e-05
    Epoch 75/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.6687 - mae: 4.1301 - lr: 5.0119e-05
    Epoch 76/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.6071 - mae: 4.0604 - lr: 5.6234e-05
    Epoch 77/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.4510 - mae: 3.9372 - lr: 6.3096e-05
    Epoch 78/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.3652 - mae: 3.8122 - lr: 7.0795e-05
    Epoch 79/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.2506 - mae: 3.7256 - lr: 7.9433e-05
    Epoch 80/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.3137 - mae: 3.7916 - lr: 8.9125e-05
    Epoch 81/100
    94/94 [==============================] - 3s 29ms/step - loss: 3.0446 - mae: 3.5303 - lr: 1.0000e-04
    Epoch 82/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.0396 - mae: 3.5044 - lr: 1.1220e-04
    Epoch 83/100
    94/94 [==============================] - 3s 29ms/step - loss: 3.1091 - mae: 3.5500 - lr: 1.2589e-04
    Epoch 84/100
    94/94 [==============================] - 3s 29ms/step - loss: 3.0488 - mae: 3.5263 - lr: 1.4125e-04
    Epoch 85/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.0680 - mae: 3.5280 - lr: 1.5849e-04
    Epoch 86/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.0208 - mae: 3.4821 - lr: 1.7783e-04
    Epoch 87/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.0580 - mae: 3.5221 - lr: 1.9953e-04
    Epoch 88/100
    94/94 [==============================] - 3s 28ms/step - loss: 2.9100 - mae: 3.3843 - lr: 2.2387e-04
    Epoch 89/100
    94/94 [==============================] - 3s 29ms/step - loss: 3.2118 - mae: 3.6719 - lr: 2.5119e-04
    Epoch 90/100
    94/94 [==============================] - 3s 28ms/step - loss: 3.1488 - mae: 3.6115 - lr: 2.8184e-04
    Epoch 91/100
    94/94 [==============================] - 3s 28ms/step - loss: 2.9512 - mae: 3.4205 - lr: 3.1623e-04
    Epoch 92/100
    94/94 [==============================] - 3s 29ms/step - loss: 3.0505 - mae: 3.5465 - lr: 3.5481e-04
    Epoch 93/100
    94/94 [==============================] - 3s 28ms/step - loss: 2.8636 - mae: 3.3496 - lr: 3.9811e-04
    Epoch 94/100
    94/94 [==============================] - 3s 29ms/step - loss: 2.9379 - mae: 3.3702 - lr: 4.4668e-04
    Epoch 95/100
    94/94 [==============================] - 3s 28ms/step - loss: 2.9892 - mae: 3.4724 - lr: 5.0119e-04
    Epoch 96/100
    94/94 [==============================] - 3s 29ms/step - loss: 2.9214 - mae: 3.4063 - lr: 5.6234e-04
    Epoch 97/100
    94/94 [==============================] - 3s 28ms/step - loss: 2.8961 - mae: 3.3721 - lr: 6.3096e-04
    Epoch 98/100
    94/94 [==============================] - 3s 28ms/step - loss: 2.9407 - mae: 3.3576 - lr: 7.0795e-04
    Epoch 99/100
    94/94 [==============================] - 3s 28ms/step - loss: 2.8388 - mae: 3.2893 - lr: 7.9433e-04
    Epoch 100/100
    94/94 [==============================] - 3s 29ms/step - loss: 3.2231 - mae: 3.6984 - lr: 8.9125e-04
    


```python
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
```




    (1e-08, 0.0001, 0.0, 30.0)




![png](output_10_1.png)


We, then plot the loss vs learning rate and find out that loss was less and quite stable at $10^{-5}$. Using this as our learning rate, we train for 500 epochs.


```python
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])


model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),metrics=["mae"])
history = model.fit(dataset,epochs=500,verbose=1)
```

    Epoch 1/500
    94/94 [==============================] - 3s 29ms/step - loss: 260.9243 - mae: 10.1378
    Epoch 2/500
    94/94 [==============================] - 3s 29ms/step - loss: 33.4073 - mae: 3.9017
    Epoch 3/500
    94/94 [==============================] - 3s 29ms/step - loss: 27.5539 - mae: 3.5518
    Epoch 4/500
    94/94 [==============================] - 3s 28ms/step - loss: 31.6568 - mae: 3.9759
    Epoch 5/500
    94/94 [==============================] - 3s 29ms/step - loss: 27.0354 - mae: 3.5515
    Epoch 6/500
    94/94 [==============================] - 3s 30ms/step - loss: 25.6458 - mae: 3.4461
    Epoch 7/500
    94/94 [==============================] - 3s 29ms/step - loss: 31.8559 - mae: 3.9959
    Epoch 8/500
    94/94 [==============================] - 3s 29ms/step - loss: 27.8153 - mae: 3.6121
    Epoch 9/500
    94/94 [==============================] - 3s 31ms/step - loss: 29.5963 - mae: 3.8187
    Epoch 10/500
    94/94 [==============================] - 3s 28ms/step - loss: 27.9760 - mae: 3.6757
    Epoch 11/500
    94/94 [==============================] - 3s 29ms/step - loss: 27.4339 - mae: 3.6672
    Epoch 12/500
    94/94 [==============================] - 3s 29ms/step - loss: 25.1888 - mae: 3.4160
    Epoch 13/500
    94/94 [==============================] - 3s 29ms/step - loss: 28.0933 - mae: 3.7027
    Epoch 14/500
    94/94 [==============================] - 3s 29ms/step - loss: 26.9125 - mae: 3.5711
    Epoch 15/500
    94/94 [==============================] - 3s 29ms/step - loss: 25.7826 - mae: 3.4787
    Epoch 16/500
    94/94 [==============================] - 3s 29ms/step - loss: 29.2301 - mae: 3.8158
    Epoch 17/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.0560 - mae: 3.3191
    Epoch 18/500
    94/94 [==============================] - 3s 29ms/step - loss: 26.5630 - mae: 3.5473
    Epoch 19/500
    94/94 [==============================] - 3s 29ms/step - loss: 25.6564 - mae: 3.4825
    Epoch 20/500
    94/94 [==============================] - 3s 29ms/step - loss: 25.0230 - mae: 3.4211
    Epoch 21/500
    94/94 [==============================] - 3s 28ms/step - loss: 26.9035 - mae: 3.5829
    Epoch 22/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.9078 - mae: 3.3448
    Epoch 23/500
    94/94 [==============================] - 3s 29ms/step - loss: 27.8602 - mae: 3.7013
    Epoch 24/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.9127 - mae: 3.4339
    Epoch 25/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.1448 - mae: 3.3488
    Epoch 26/500
    94/94 [==============================] - 3s 29ms/step - loss: 25.8940 - mae: 3.5183
    Epoch 27/500
    94/94 [==============================] - 3s 28ms/step - loss: 29.4972 - mae: 3.7865
    Epoch 28/500
    94/94 [==============================] - 3s 29ms/step - loss: 27.9105 - mae: 3.7044
    Epoch 29/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.8223 - mae: 3.3226
    Epoch 30/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.4867 - mae: 3.2755
    Epoch 31/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.9789 - mae: 3.3859
    Epoch 32/500
    94/94 [==============================] - 3s 29ms/step - loss: 27.9910 - mae: 3.7034
    Epoch 33/500
    94/94 [==============================] - 3s 28ms/step - loss: 24.4712 - mae: 3.3983
    Epoch 34/500
    94/94 [==============================] - 3s 29ms/step - loss: 26.9075 - mae: 3.6388
    Epoch 35/500
    94/94 [==============================] - 3s 29ms/step - loss: 27.7890 - mae: 3.3517
    Epoch 36/500
    94/94 [==============================] - 3s 28ms/step - loss: 25.9258 - mae: 3.5010
    Epoch 37/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.1371 - mae: 3.3698
    Epoch 38/500
    94/94 [==============================] - 3s 29ms/step - loss: 25.3325 - mae: 3.4867
    Epoch 39/500
    94/94 [==============================] - 3s 29ms/step - loss: 29.4570 - mae: 3.8720
    Epoch 40/500
    94/94 [==============================] - 3s 29ms/step - loss: 27.7061 - mae: 3.6383
    Epoch 41/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.6731 - mae: 3.4319
    Epoch 42/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.0372 - mae: 3.3383
    Epoch 43/500
    94/94 [==============================] - 3s 28ms/step - loss: 25.0614 - mae: 3.4564
    Epoch 44/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.6508 - mae: 3.2177
    Epoch 45/500
    94/94 [==============================] - 3s 29ms/step - loss: 26.1932 - mae: 3.5584
    Epoch 46/500
    94/94 [==============================] - 3s 29ms/step - loss: 28.8187 - mae: 3.5451
    Epoch 47/500
    94/94 [==============================] - 3s 28ms/step - loss: 27.2578 - mae: 3.6459
    Epoch 48/500
    94/94 [==============================] - 3s 29ms/step - loss: 25.0742 - mae: 3.4406
    Epoch 49/500
    94/94 [==============================] - 3s 28ms/step - loss: 24.5512 - mae: 3.4041
    Epoch 50/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.9825 - mae: 3.3799
    Epoch 51/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.6023 - mae: 3.2937
    Epoch 52/500
    94/94 [==============================] - 3s 29ms/step - loss: 26.3073 - mae: 3.5666
    Epoch 53/500
    94/94 [==============================] - 3s 30ms/step - loss: 24.7269 - mae: 3.3710
    Epoch 54/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.6298 - mae: 3.2934
    Epoch 55/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.1335 - mae: 3.2379
    Epoch 56/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.7154 - mae: 3.3371
    Epoch 57/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.9116 - mae: 3.3308
    Epoch 58/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.8936 - mae: 3.4204
    Epoch 59/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.6073 - mae: 3.3415
    Epoch 60/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.5089 - mae: 3.2789
    Epoch 61/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.7189 - mae: 3.4363
    Epoch 62/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.6546 - mae: 3.3026
    Epoch 63/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.7552 - mae: 3.3240
    Epoch 64/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.9041 - mae: 3.3732
    Epoch 65/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.9327 - mae: 3.2762
    Epoch 66/500
    94/94 [==============================] - 3s 29ms/step - loss: 25.0834 - mae: 3.4505
    Epoch 67/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.5338 - mae: 3.2022
    Epoch 68/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.3399 - mae: 3.2604
    Epoch 69/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.3025 - mae: 3.4066
    Epoch 70/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.7113 - mae: 3.2219
    Epoch 71/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.7327 - mae: 3.2379
    Epoch 72/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.7573 - mae: 3.2511
    Epoch 73/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.4239 - mae: 3.3078
    Epoch 74/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.2532 - mae: 3.2647
    Epoch 75/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.4609 - mae: 3.3068
    Epoch 76/500
    94/94 [==============================] - 3s 30ms/step - loss: 22.9219 - mae: 3.2700
    Epoch 77/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.9490 - mae: 3.3795
    Epoch 78/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.0954 - mae: 3.1299
    Epoch 79/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.4475 - mae: 3.1946
    Epoch 80/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.6905 - mae: 3.2368
    Epoch 81/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.2462 - mae: 3.2895
    Epoch 82/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.4814 - mae: 3.3587
    Epoch 83/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.3126 - mae: 3.4740
    Epoch 84/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.7365 - mae: 3.2698
    Epoch 85/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.4842 - mae: 3.1069
    Epoch 86/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.5229 - mae: 3.2007
    Epoch 87/500
    94/94 [==============================] - 3s 28ms/step - loss: 24.1722 - mae: 3.4214
    Epoch 88/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.6527 - mae: 3.1197
    Epoch 89/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.4167 - mae: 3.2804
    Epoch 90/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.5472 - mae: 3.2255
    Epoch 91/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.8491 - mae: 3.2678
    Epoch 92/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.7938 - mae: 3.2492
    Epoch 93/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.7464 - mae: 3.1521
    Epoch 94/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.8238 - mae: 3.2445
    Epoch 95/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.0789 - mae: 3.1707
    Epoch 96/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.3846 - mae: 3.2424
    Epoch 97/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.4396 - mae: 3.1395
    Epoch 98/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.8234 - mae: 3.2505
    Epoch 99/500
    94/94 [==============================] - 3s 29ms/step - loss: 24.9576 - mae: 3.4763
    Epoch 100/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.1408 - mae: 3.2191
    Epoch 101/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.2857 - mae: 3.2249
    Epoch 102/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.8496 - mae: 3.2552
    Epoch 103/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.8038 - mae: 3.1619
    Epoch 104/500
    94/94 [==============================] - 3s 28ms/step - loss: 26.8354 - mae: 3.2110
    Epoch 105/500
    94/94 [==============================] - 3s 29ms/step - loss: 29.7648 - mae: 3.9280
    Epoch 106/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.3701 - mae: 3.3321
    Epoch 107/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.9000 - mae: 3.1658
    Epoch 108/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.3959 - mae: 3.1116
    Epoch 109/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.8241 - mae: 3.2621
    Epoch 110/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.0320 - mae: 3.3056
    Epoch 111/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.1043 - mae: 3.0915
    Epoch 112/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.3396 - mae: 3.2165
    Epoch 113/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.2771 - mae: 3.1083
    Epoch 114/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.3585 - mae: 3.1475
    Epoch 115/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.5683 - mae: 3.1260
    Epoch 116/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.1277 - mae: 3.1667
    Epoch 117/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.3421 - mae: 3.1136
    Epoch 118/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.8514 - mae: 3.1652
    Epoch 119/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.5553 - mae: 3.1390
    Epoch 120/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.6411 - mae: 3.1431
    Epoch 121/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.6718 - mae: 3.1721
    Epoch 122/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.0239 - mae: 3.0954
    Epoch 123/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.5544 - mae: 3.1588
    Epoch 124/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.8165 - mae: 3.1725
    Epoch 125/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.5042 - mae: 3.2897
    Epoch 126/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.9986 - mae: 3.1933
    Epoch 127/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.8257 - mae: 3.1896
    Epoch 128/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.0957 - mae: 3.2698
    Epoch 129/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.2831 - mae: 3.2214
    Epoch 130/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.4674 - mae: 3.2470
    Epoch 131/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.7978 - mae: 3.0797
    Epoch 132/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4392 - mae: 3.0165
    Epoch 133/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.5255 - mae: 3.0416
    Epoch 134/500
    94/94 [==============================] - 3s 28ms/step - loss: 26.0501 - mae: 3.1241
    Epoch 135/500
    94/94 [==============================] - 3s 29ms/step - loss: 25.0075 - mae: 3.4524
    Epoch 136/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.6876 - mae: 3.1170
    Epoch 137/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.0236 - mae: 3.1106
    Epoch 138/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.9527 - mae: 3.1476
    Epoch 139/500
    94/94 [==============================] - 3s 28ms/step - loss: 24.5650 - mae: 3.0783
    Epoch 140/500
    94/94 [==============================] - 3s 28ms/step - loss: 25.6310 - mae: 3.5320
    Epoch 141/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.6603 - mae: 3.0338
    Epoch 142/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1360 - mae: 3.0033
    Epoch 143/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.7506 - mae: 3.1638
    Epoch 144/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.0185 - mae: 3.0857
    Epoch 145/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.2586 - mae: 3.2245
    Epoch 146/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.5515 - mae: 3.1214
    Epoch 147/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.8862 - mae: 3.0714
    Epoch 148/500
    94/94 [==============================] - 3s 28ms/step - loss: 25.3068 - mae: 3.2410
    Epoch 149/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.4525 - mae: 3.3757
    Epoch 150/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.0707 - mae: 3.2086
    Epoch 151/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.3614 - mae: 3.2051
    Epoch 152/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.0342 - mae: 3.0776
    Epoch 153/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.8891 - mae: 3.0829
    Epoch 154/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.2346 - mae: 3.3304
    Epoch 155/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3567 - mae: 3.0309
    Epoch 156/500
    94/94 [==============================] - 3s 29ms/step - loss: 23.2377 - mae: 3.3332
    Epoch 157/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3291 - mae: 3.0087
    Epoch 158/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.2275 - mae: 3.1344
    Epoch 159/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.4989 - mae: 3.1906
    Epoch 160/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5767 - mae: 3.0115
    Epoch 161/500
    94/94 [==============================] - 3s 29ms/step - loss: 22.1617 - mae: 3.1934
    Epoch 162/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.0533 - mae: 3.0469
    Epoch 163/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.8568 - mae: 3.0612
    Epoch 164/500
    94/94 [==============================] - 3s 29ms/step - loss: 25.4129 - mae: 3.1015
    Epoch 165/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.1789 - mae: 3.2423
    Epoch 166/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.9662 - mae: 3.0750
    Epoch 167/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7865 - mae: 3.0631
    Epoch 168/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.1888 - mae: 3.1401
    Epoch 169/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.9257 - mae: 3.0777
    Epoch 170/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1197 - mae: 3.0056
    Epoch 171/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.8592 - mae: 3.0901
    Epoch 172/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.1259 - mae: 2.9919
    Epoch 173/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5221 - mae: 3.0338
    Epoch 174/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.6374 - mae: 3.0725
    Epoch 175/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.4938 - mae: 3.0553
    Epoch 176/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.7650 - mae: 3.0842
    Epoch 177/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.3270 - mae: 3.0239
    Epoch 178/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.3860 - mae: 3.1252
    Epoch 179/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.4867 - mae: 3.0162
    Epoch 180/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.4717 - mae: 3.1437
    Epoch 181/500
    94/94 [==============================] - 3s 30ms/step - loss: 20.1010 - mae: 2.9943
    Epoch 182/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.4011 - mae: 3.0970
    Epoch 183/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.6700 - mae: 3.0468
    Epoch 184/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.5746 - mae: 3.0206
    Epoch 185/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.5573 - mae: 3.1434
    Epoch 186/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.4949 - mae: 3.1443
    Epoch 187/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.6804 - mae: 3.0351
    Epoch 188/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.3363 - mae: 3.1160
    Epoch 189/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7864 - mae: 3.0719
    Epoch 190/500
    94/94 [==============================] - 3s 30ms/step - loss: 20.4138 - mae: 3.0243
    Epoch 191/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.2353 - mae: 3.0453
    Epoch 192/500
    94/94 [==============================] - 3s 30ms/step - loss: 20.4726 - mae: 3.0428
    Epoch 193/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.4023 - mae: 3.1381
    Epoch 194/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.9073 - mae: 3.2032
    Epoch 195/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5996 - mae: 3.0399
    Epoch 196/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.9243 - mae: 3.0960
    Epoch 197/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.8131 - mae: 3.0891
    Epoch 198/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2999 - mae: 3.0097
    Epoch 199/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.4425 - mae: 3.1447
    Epoch 200/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.3553 - mae: 3.0988
    Epoch 201/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.2961 - mae: 3.1056
    Epoch 202/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.9185 - mae: 3.0833
    Epoch 203/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7674 - mae: 3.0589
    Epoch 204/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.8820 - mae: 3.0975
    Epoch 205/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0250 - mae: 2.9756
    Epoch 206/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.0128 - mae: 2.9890
    Epoch 207/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4610 - mae: 3.0584
    Epoch 208/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9634 - mae: 2.9780
    Epoch 209/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.0590 - mae: 2.9819
    Epoch 210/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.5749 - mae: 3.0732
    Epoch 211/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.5734 - mae: 3.1931
    Epoch 212/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8305 - mae: 3.0070
    Epoch 213/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.0804 - mae: 2.9622
    Epoch 214/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.5961 - mae: 2.9234
    Epoch 215/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.6859 - mae: 2.9764
    Epoch 216/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.4786 - mae: 3.3028
    Epoch 217/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.2870 - mae: 3.0085
    Epoch 218/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.8524 - mae: 3.1605
    Epoch 219/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.2863 - mae: 3.0916
    Epoch 220/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1230 - mae: 3.0016
    Epoch 221/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4097 - mae: 3.0381
    Epoch 222/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2801 - mae: 2.9885
    Epoch 223/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8932 - mae: 2.9865
    Epoch 224/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.4441 - mae: 3.1199
    Epoch 225/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3803 - mae: 3.0233
    Epoch 226/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.5219 - mae: 3.1389
    Epoch 227/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.9942 - mae: 3.0664
    Epoch 228/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.0498 - mae: 3.0862
    Epoch 229/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5129 - mae: 3.0407
    Epoch 230/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.4228 - mae: 3.1512
    Epoch 231/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6899 - mae: 2.9490
    Epoch 232/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.6142 - mae: 3.0615
    Epoch 233/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.8695 - mae: 3.0976
    Epoch 234/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.7748 - mae: 3.1707
    Epoch 235/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7125 - mae: 3.0017
    Epoch 236/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.7429 - mae: 3.0433
    Epoch 237/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.7762 - mae: 3.2219
    Epoch 238/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2822 - mae: 3.0038
    Epoch 239/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0583 - mae: 2.9885
    Epoch 240/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2285 - mae: 3.0319
    Epoch 241/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.8370 - mae: 3.0825
    Epoch 242/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.5701 - mae: 3.1984
    Epoch 243/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.5086 - mae: 3.1192
    Epoch 244/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.0289 - mae: 3.1037
    Epoch 245/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.6227 - mae: 3.0662
    Epoch 246/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.9999 - mae: 3.1024
    Epoch 247/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7712 - mae: 2.9639
    Epoch 248/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.2839 - mae: 3.1007
    Epoch 249/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9272 - mae: 2.9622
    Epoch 250/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1893 - mae: 3.0003
    Epoch 251/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.6455 - mae: 3.0829
    Epoch 252/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.9825 - mae: 3.0600
    Epoch 253/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2173 - mae: 2.9927
    Epoch 254/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9564 - mae: 2.9835
    Epoch 255/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.6334 - mae: 3.0846
    Epoch 256/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5229 - mae: 3.0273
    Epoch 257/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3960 - mae: 3.0393
    Epoch 258/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1027 - mae: 2.9595
    Epoch 259/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.1440 - mae: 3.1025
    Epoch 260/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5783 - mae: 3.0398
    Epoch 261/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.9595 - mae: 2.9952
    Epoch 262/500
    94/94 [==============================] - 3s 30ms/step - loss: 20.6480 - mae: 3.0839
    Epoch 263/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.0884 - mae: 3.0135
    Epoch 264/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4509 - mae: 3.0605
    Epoch 265/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.7556 - mae: 3.0748
    Epoch 266/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1365 - mae: 3.0167
    Epoch 267/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.9347 - mae: 2.9731
    Epoch 268/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.9222 - mae: 3.0871
    Epoch 269/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7224 - mae: 3.0450
    Epoch 270/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.7339 - mae: 3.1746
    Epoch 271/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3750 - mae: 3.0432
    Epoch 272/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.5896 - mae: 3.1757
    Epoch 273/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.9395 - mae: 3.0835
    Epoch 274/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.8885 - mae: 3.0842
    Epoch 275/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.9252 - mae: 2.9863
    Epoch 276/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.9370 - mae: 2.9702
    Epoch 277/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.8639 - mae: 3.0712
    Epoch 278/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7260 - mae: 2.9564
    Epoch 279/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8492 - mae: 2.9696
    Epoch 280/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5338 - mae: 2.9248
    Epoch 281/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.6389 - mae: 3.1607
    Epoch 282/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0440 - mae: 2.9583
    Epoch 283/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.1353 - mae: 3.1102
    Epoch 284/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.2481 - mae: 3.1398
    Epoch 285/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.0125 - mae: 2.9680
    Epoch 286/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4069 - mae: 3.0271
    Epoch 287/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5899 - mae: 3.0502
    Epoch 288/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4308 - mae: 3.0328
    Epoch 289/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.9048 - mae: 3.0680
    Epoch 290/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5563 - mae: 2.9487
    Epoch 291/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1250 - mae: 2.9706
    Epoch 292/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3264 - mae: 3.0356
    Epoch 293/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.3822 - mae: 3.0093
    Epoch 294/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9896 - mae: 2.9921
    Epoch 295/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4534 - mae: 2.9353
    Epoch 296/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.7924 - mae: 2.9765
    Epoch 297/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.6534 - mae: 3.1188
    Epoch 298/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.8756 - mae: 2.9898
    Epoch 299/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5606 - mae: 3.0834
    Epoch 300/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9835 - mae: 3.0018
    Epoch 301/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7249 - mae: 2.9682
    Epoch 302/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2354 - mae: 2.9935
    Epoch 303/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.2089 - mae: 3.0097
    Epoch 304/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.6306 - mae: 3.0655
    Epoch 305/500
    94/94 [==============================] - 3s 29ms/step - loss: 21.6688 - mae: 3.1606
    Epoch 306/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.2002 - mae: 2.9708
    Epoch 307/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3726 - mae: 3.0184
    Epoch 308/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1405 - mae: 3.0025
    Epoch 309/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.6192 - mae: 3.0132
    Epoch 310/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.9798 - mae: 3.0957
    Epoch 311/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.9118 - mae: 2.9649
    Epoch 312/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.9878 - mae: 3.1284
    Epoch 313/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5964 - mae: 2.9476
    Epoch 314/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5805 - mae: 3.0278
    Epoch 315/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7011 - mae: 2.9867
    Epoch 316/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7744 - mae: 3.0784
    Epoch 317/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7310 - mae: 3.1065
    Epoch 318/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.9247 - mae: 2.9876
    Epoch 319/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2988 - mae: 3.0285
    Epoch 320/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.0208 - mae: 3.3163
    Epoch 321/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8889 - mae: 2.9703
    Epoch 322/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7205 - mae: 2.9576
    Epoch 323/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.5104 - mae: 3.0458
    Epoch 324/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.2333 - mae: 3.2227
    Epoch 325/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9029 - mae: 2.9907
    Epoch 326/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7106 - mae: 2.9481
    Epoch 327/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5818 - mae: 2.9529
    Epoch 328/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4747 - mae: 3.0604
    Epoch 329/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.6147 - mae: 3.2417
    Epoch 330/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1789 - mae: 3.0164
    Epoch 331/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.2355 - mae: 3.0043
    Epoch 332/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8299 - mae: 2.9729
    Epoch 333/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4152 - mae: 3.0252
    Epoch 334/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.6522 - mae: 3.0693
    Epoch 335/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9685 - mae: 3.0137
    Epoch 336/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9232 - mae: 2.9979
    Epoch 337/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9415 - mae: 2.9655
    Epoch 338/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.3307 - mae: 2.9085
    Epoch 339/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.1259 - mae: 2.9986
    Epoch 340/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3671 - mae: 3.0326
    Epoch 341/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5547 - mae: 2.9451
    Epoch 342/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4791 - mae: 2.9364
    Epoch 343/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7233 - mae: 3.0912
    Epoch 344/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9924 - mae: 2.9748
    Epoch 345/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1990 - mae: 3.0003
    Epoch 346/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.2729 - mae: 3.0250
    Epoch 347/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8781 - mae: 2.9702
    Epoch 348/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.6821 - mae: 2.9660
    Epoch 349/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9880 - mae: 2.9782
    Epoch 350/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.0633 - mae: 2.9843
    Epoch 351/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9726 - mae: 2.9860
    Epoch 352/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5669 - mae: 3.0467
    Epoch 353/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0424 - mae: 2.9854
    Epoch 354/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7629 - mae: 2.9550
    Epoch 355/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0486 - mae: 3.0038
    Epoch 356/500
    94/94 [==============================] - 3s 28ms/step - loss: 24.4919 - mae: 2.9800
    Epoch 357/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.5382 - mae: 3.1732
    Epoch 358/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3099 - mae: 3.0384
    Epoch 359/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.7121 - mae: 2.9785
    Epoch 360/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.8718 - mae: 2.9744
    Epoch 361/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2973 - mae: 3.0360
    Epoch 362/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1177 - mae: 3.0239
    Epoch 363/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.6168 - mae: 3.0530
    Epoch 364/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9341 - mae: 2.9943
    Epoch 365/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8721 - mae: 2.9775
    Epoch 366/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3081 - mae: 3.0387
    Epoch 367/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.2033 - mae: 3.1301
    Epoch 368/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4507 - mae: 3.0484
    Epoch 369/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4414 - mae: 2.9516
    Epoch 370/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0979 - mae: 2.9776
    Epoch 371/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8027 - mae: 2.9579
    Epoch 372/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5537 - mae: 2.9397
    Epoch 373/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0116 - mae: 2.9710
    Epoch 374/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4849 - mae: 2.9142
    Epoch 375/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3556 - mae: 3.0319
    Epoch 376/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2366 - mae: 3.0104
    Epoch 377/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.5409 - mae: 3.0482
    Epoch 378/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.9237 - mae: 3.1172
    Epoch 379/500
    94/94 [==============================] - 3s 28ms/step - loss: 24.5019 - mae: 3.0412
    Epoch 380/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.3485 - mae: 3.2265
    Epoch 381/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.5157 - mae: 2.9642
    Epoch 382/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.1478 - mae: 2.8969
    Epoch 383/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7225 - mae: 2.9904
    Epoch 384/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6986 - mae: 2.9628
    Epoch 385/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.5711 - mae: 3.0534
    Epoch 386/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.3419 - mae: 2.9372
    Epoch 387/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3766 - mae: 3.0649
    Epoch 388/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.3269 - mae: 2.9904
    Epoch 389/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7586 - mae: 3.0988
    Epoch 390/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2858 - mae: 3.0506
    Epoch 391/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9343 - mae: 2.9881
    Epoch 392/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8017 - mae: 2.9564
    Epoch 393/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6002 - mae: 2.9364
    Epoch 394/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8686 - mae: 2.9773
    Epoch 395/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.3826 - mae: 2.9269
    Epoch 396/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4468 - mae: 2.9314
    Epoch 397/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3508 - mae: 3.0681
    Epoch 398/500
    94/94 [==============================] - 3s 28ms/step - loss: 23.4464 - mae: 2.9458
    Epoch 399/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.6094 - mae: 3.1401
    Epoch 400/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.0033 - mae: 2.9976
    Epoch 401/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8765 - mae: 2.9905
    Epoch 402/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.1166 - mae: 2.9212
    Epoch 403/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6491 - mae: 2.9564
    Epoch 404/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.1933 - mae: 2.8962
    Epoch 405/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5048 - mae: 2.9403
    Epoch 406/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8629 - mae: 2.9827
    Epoch 407/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.4246 - mae: 2.9466
    Epoch 408/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6301 - mae: 2.9484
    Epoch 409/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7920 - mae: 3.0941
    Epoch 410/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8492 - mae: 3.0077
    Epoch 411/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.3994 - mae: 2.9467
    Epoch 412/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.2020 - mae: 2.9957
    Epoch 413/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.1697 - mae: 3.0288
    Epoch 414/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7121 - mae: 3.0405
    Epoch 415/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.8321 - mae: 3.1987
    Epoch 416/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9067 - mae: 2.9493
    Epoch 417/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9746 - mae: 2.9991
    Epoch 418/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.3654 - mae: 2.9231
    Epoch 419/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.6961 - mae: 2.9576
    Epoch 420/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5969 - mae: 2.9661
    Epoch 421/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.4324 - mae: 3.0387
    Epoch 422/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4641 - mae: 2.9423
    Epoch 423/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.6405 - mae: 2.9603
    Epoch 424/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4215 - mae: 3.0556
    Epoch 425/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0044 - mae: 2.9971
    Epoch 426/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.2013 - mae: 2.9056
    Epoch 427/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.7588 - mae: 3.0853
    Epoch 428/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9475 - mae: 3.0083
    Epoch 429/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.3400 - mae: 2.9370
    Epoch 430/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.3521 - mae: 3.0320
    Epoch 431/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7940 - mae: 2.9730
    Epoch 432/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4040 - mae: 2.9152
    Epoch 433/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2492 - mae: 3.0083
    Epoch 434/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.2445 - mae: 3.0328
    Epoch 435/500
    94/94 [==============================] - 3s 28ms/step - loss: 22.5854 - mae: 3.2971
    Epoch 436/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4650 - mae: 2.9350
    Epoch 437/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.3986 - mae: 2.9403
    Epoch 438/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2143 - mae: 3.0013
    Epoch 439/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7453 - mae: 2.9389
    Epoch 440/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.1448 - mae: 2.8906
    Epoch 441/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2495 - mae: 2.9946
    Epoch 442/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6684 - mae: 2.9551
    Epoch 443/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.4236 - mae: 3.0504
    Epoch 444/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.9483 - mae: 2.9844
    Epoch 445/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4991 - mae: 3.0487
    Epoch 446/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.1692 - mae: 3.1109
    Epoch 447/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1750 - mae: 3.0305
    Epoch 448/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.6375 - mae: 3.0713
    Epoch 449/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4053 - mae: 2.9326
    Epoch 450/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5284 - mae: 2.9465
    Epoch 451/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.2753 - mae: 2.9045
    Epoch 452/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.9824 - mae: 2.9894
    Epoch 453/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4559 - mae: 2.9476
    Epoch 454/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.6194 - mae: 3.0946
    Epoch 455/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4958 - mae: 3.0377
    Epoch 456/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8091 - mae: 2.9660
    Epoch 457/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0414 - mae: 3.0012
    Epoch 458/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6858 - mae: 2.9725
    Epoch 459/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6553 - mae: 2.9663
    Epoch 460/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.5338 - mae: 3.0577
    Epoch 461/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.4824 - mae: 3.0651
    Epoch 462/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1164 - mae: 3.0242
    Epoch 463/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.4856 - mae: 2.9270
    Epoch 464/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.3254 - mae: 3.0071
    Epoch 465/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.1918 - mae: 2.9131
    Epoch 466/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.9244 - mae: 2.9924
    Epoch 467/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6275 - mae: 2.9369
    Epoch 468/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.0700 - mae: 2.8899
    Epoch 469/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0381 - mae: 2.9885
    Epoch 470/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7748 - mae: 2.9412
    Epoch 471/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.0686 - mae: 2.9147
    Epoch 472/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.1384 - mae: 2.8979
    Epoch 473/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.1481 - mae: 3.0104
    Epoch 474/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.0052 - mae: 3.0159
    Epoch 475/500
    94/94 [==============================] - 3s 29ms/step - loss: 18.8902 - mae: 2.8694
    Epoch 476/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.1823 - mae: 2.8951
    Epoch 477/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.3896 - mae: 3.1443
    Epoch 478/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7183 - mae: 2.9628
    Epoch 479/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.8160 - mae: 2.9698
    Epoch 480/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.2361 - mae: 3.0457
    Epoch 481/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5367 - mae: 2.9773
    Epoch 482/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6291 - mae: 2.9588
    Epoch 483/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.4482 - mae: 2.9261
    Epoch 484/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.6441 - mae: 3.0734
    Epoch 485/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.0524 - mae: 2.9070
    Epoch 486/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.2163 - mae: 2.9132
    Epoch 487/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.7663 - mae: 2.9852
    Epoch 488/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.1162 - mae: 2.9086
    Epoch 489/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.5258 - mae: 2.9212
    Epoch 490/500
    94/94 [==============================] - 3s 29ms/step - loss: 19.7166 - mae: 2.9476
    Epoch 491/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.2588 - mae: 2.9383
    Epoch 492/500
    94/94 [==============================] - 3s 28ms/step - loss: 21.1324 - mae: 3.1124
    Epoch 493/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6550 - mae: 2.9468
    Epoch 494/500
    94/94 [==============================] - 3s 29ms/step - loss: 20.0007 - mae: 2.9948
    Epoch 495/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.0840 - mae: 2.8932
    Epoch 496/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6511 - mae: 2.9502
    Epoch 497/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.2156 - mae: 2.9267
    Epoch 498/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.6160 - mae: 2.9396
    Epoch 499/500
    94/94 [==============================] - 3s 28ms/step - loss: 20.4251 - mae: 3.0397
    Epoch 500/500
    94/94 [==============================] - 3s 28ms/step - loss: 19.3343 - mae: 2.9032
    

Using what our model learnt, we are going to predict the validation set and plot them side by side                                                                                         


```python
forecast = []
results = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))
 
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
```


![png](output_14_0.png)


Next, we calculate the MAE of predicted validation set.


```python
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
```




    2.9205346



The next code is to plot the mae and loss vs epochs to see how well the model has fared.


```python
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
mae=history.history['mae']
loss=history.history['loss']

epochs=range(len(loss)) # Get number of epochs

#------------------------------------------------
# Plot MAE and Loss
#------------------------------------------------
plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()

epochs_zoom = epochs[200:]
mae_zoom = mae[200:]
loss_zoom = loss[200:]

#------------------------------------------------
# Plot Zoomed MAE and Loss
#------------------------------------------------
plt.plot(epochs_zoom, mae_zoom, 'r')
plt.plot(epochs_zoom, loss_zoom, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()
```




    <Figure size 432x288 with 0 Axes>




![png](output_18_1.png)



![png](output_18_2.png)



    <Figure size 432x288 with 0 Axes>


Next, we are going to try a model with only 1 LSTM.


```python
tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  tf.keras.layers.LSTM(32),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])


model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),metrics=["mae"])
history = model.fit(dataset,epochs=500,verbose=1)
```

    Epoch 1/500
    94/94 [==============================] - 1s 10ms/step - loss: 420.0227 - mae: 11.4754
    Epoch 2/500
    94/94 [==============================] - 1s 10ms/step - loss: 46.2899 - mae: 4.9492
    Epoch 3/500
    94/94 [==============================] - 1s 10ms/step - loss: 45.5109 - mae: 4.9844
    Epoch 4/500
    94/94 [==============================] - 1s 10ms/step - loss: 35.8790 - mae: 4.3274
    Epoch 5/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.1591 - mae: 3.8956
    Epoch 6/500
    94/94 [==============================] - 1s 10ms/step - loss: 40.4956 - mae: 4.6770
    Epoch 7/500
    94/94 [==============================] - 1s 10ms/step - loss: 37.0422 - mae: 4.2873
    Epoch 8/500
    94/94 [==============================] - 1s 11ms/step - loss: 56.7832 - mae: 5.7634
    Epoch 9/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.7145 - mae: 4.0724
    Epoch 10/500
    94/94 [==============================] - 1s 10ms/step - loss: 35.5975 - mae: 4.3444
    Epoch 11/500
    94/94 [==============================] - 1s 10ms/step - loss: 32.8992 - mae: 4.1317
    Epoch 12/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.7249 - mae: 3.8881
    Epoch 13/500
    94/94 [==============================] - 1s 10ms/step - loss: 44.0893 - mae: 4.9510
    Epoch 14/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.9967 - mae: 3.7647
    Epoch 15/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.1039 - mae: 3.7495
    Epoch 16/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.3236 - mae: 4.0196
    Epoch 17/500
    94/94 [==============================] - 1s 10ms/step - loss: 36.3006 - mae: 4.2143
    Epoch 18/500
    94/94 [==============================] - 1s 11ms/step - loss: 31.6467 - mae: 4.0461
    Epoch 19/500
    94/94 [==============================] - 1s 10ms/step - loss: 30.5163 - mae: 3.9626
    Epoch 20/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.6787 - mae: 3.4657
    Epoch 21/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.7513 - mae: 3.8333
    Epoch 22/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.4697 - mae: 3.9998
    Epoch 23/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.4395 - mae: 3.7176
    Epoch 24/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.2855 - mae: 3.7348
    Epoch 25/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.6966 - mae: 3.9268
    Epoch 26/500
    94/94 [==============================] - 1s 10ms/step - loss: 32.6525 - mae: 4.1553
    Epoch 27/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.1877 - mae: 4.0860
    Epoch 28/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.6748 - mae: 3.6007
    Epoch 29/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.1945 - mae: 3.7915
    Epoch 30/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.2063 - mae: 3.8052
    Epoch 31/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.3194 - mae: 4.0922
    Epoch 32/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.4610 - mae: 3.8823
    Epoch 33/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.4174 - mae: 3.8080
    Epoch 34/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.2047 - mae: 4.0306
    Epoch 35/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.0772 - mae: 3.5612
    Epoch 36/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.6391 - mae: 4.0726
    Epoch 37/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.8159 - mae: 3.9256
    Epoch 38/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.3274 - mae: 4.0229
    Epoch 39/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.3659 - mae: 3.6320
    Epoch 40/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.8694 - mae: 4.0739
    Epoch 41/500
    94/94 [==============================] - 1s 11ms/step - loss: 29.1054 - mae: 3.8648
    Epoch 42/500
    94/94 [==============================] - 1s 10ms/step - loss: 35.4346 - mae: 4.2802
    Epoch 43/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.2185 - mae: 3.5876
    Epoch 44/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.5454 - mae: 3.6458
    Epoch 45/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.0653 - mae: 3.8690
    Epoch 46/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.0870 - mae: 3.5676
    Epoch 47/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.9778 - mae: 3.6952
    Epoch 48/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.9565 - mae: 3.8003
    Epoch 49/500
    94/94 [==============================] - 1s 11ms/step - loss: 35.8894 - mae: 4.3494
    Epoch 50/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.6399 - mae: 3.9136
    Epoch 51/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.0489 - mae: 3.6628
    Epoch 52/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3536 - mae: 3.5306
    Epoch 53/500
    94/94 [==============================] - 1s 11ms/step - loss: 28.3303 - mae: 3.5084
    Epoch 54/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.0236 - mae: 4.0358
    Epoch 55/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.5448 - mae: 3.7246
    Epoch 56/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.1416 - mae: 3.6358
    Epoch 57/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.5822 - mae: 3.7144
    Epoch 58/500
    94/94 [==============================] - 1s 10ms/step - loss: 31.8170 - mae: 3.7276
    Epoch 59/500
    94/94 [==============================] - 1s 10ms/step - loss: 44.1734 - mae: 4.8174
    Epoch 60/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.1900 - mae: 3.7638
    Epoch 61/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.6856 - mae: 3.6242
    Epoch 62/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.8266 - mae: 3.5577
    Epoch 63/500
    94/94 [==============================] - 1s 10ms/step - loss: 30.8594 - mae: 4.0069
    Epoch 64/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.1727 - mae: 3.8912
    Epoch 65/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.2834 - mae: 3.6229
    Epoch 66/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.7279 - mae: 3.4759
    Epoch 67/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.2037 - mae: 3.5843
    Epoch 68/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.3815 - mae: 3.6590
    Epoch 69/500
    94/94 [==============================] - 1s 10ms/step - loss: 32.3514 - mae: 4.0542
    Epoch 70/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.8738 - mae: 3.8036
    Epoch 71/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.3470 - mae: 3.6157
    Epoch 72/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.9908 - mae: 3.7797
    Epoch 73/500
    94/94 [==============================] - 1s 11ms/step - loss: 25.6883 - mae: 3.5351
    Epoch 74/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.4613 - mae: 3.5088
    Epoch 75/500
    94/94 [==============================] - 1s 11ms/step - loss: 25.0104 - mae: 3.5349
    Epoch 76/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.6715 - mae: 3.7890
    Epoch 77/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.5041 - mae: 3.9115
    Epoch 78/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3441 - mae: 3.5096
    Epoch 79/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.7460 - mae: 3.7475
    Epoch 80/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.4730 - mae: 3.8065
    Epoch 81/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.9580 - mae: 3.4360
    Epoch 82/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.5398 - mae: 3.5438
    Epoch 83/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.0618 - mae: 3.5712
    Epoch 84/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.6603 - mae: 3.9027
    Epoch 85/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.5829 - mae: 3.6362
    Epoch 86/500
    94/94 [==============================] - 1s 10ms/step - loss: 30.4043 - mae: 3.9836
    Epoch 87/500
    94/94 [==============================] - 1s 10ms/step - loss: 33.4316 - mae: 3.8901
    Epoch 88/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.5667 - mae: 3.8875
    Epoch 89/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.4233 - mae: 3.4802
    Epoch 90/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.0640 - mae: 3.6442
    Epoch 91/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.9487 - mae: 3.5712
    Epoch 92/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.7191 - mae: 3.6299
    Epoch 93/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.3439 - mae: 3.5891
    Epoch 94/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.9372 - mae: 3.6893
    Epoch 95/500
    94/94 [==============================] - 1s 10ms/step - loss: 30.6830 - mae: 3.9992
    Epoch 96/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.6603 - mae: 3.2475
    Epoch 97/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.7314 - mae: 3.7135
    Epoch 98/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.5180 - mae: 3.6029
    Epoch 99/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8237 - mae: 3.4820
    Epoch 100/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.3656 - mae: 3.4228
    Epoch 101/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.4804 - mae: 3.4972
    Epoch 102/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8536 - mae: 3.3876
    Epoch 103/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.8391 - mae: 3.6790
    Epoch 104/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.7390 - mae: 3.4894
    Epoch 105/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.2574 - mae: 3.4268
    Epoch 106/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.3111 - mae: 3.7530
    Epoch 107/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.2348 - mae: 3.7815
    Epoch 108/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.7371 - mae: 3.4877
    Epoch 109/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.7250 - mae: 3.3386
    Epoch 110/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.7722 - mae: 3.3242
    Epoch 111/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.6818 - mae: 3.4598
    Epoch 112/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.3512 - mae: 3.7083
    Epoch 113/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.8678 - mae: 3.6295
    Epoch 114/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.1006 - mae: 3.6594
    Epoch 115/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8503 - mae: 3.4288
    Epoch 116/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.9598 - mae: 3.4064
    Epoch 117/500
    94/94 [==============================] - 1s 10ms/step - loss: 34.2036 - mae: 4.2366
    Epoch 118/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.5216 - mae: 3.4321
    Epoch 119/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.0339 - mae: 3.5999
    Epoch 120/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.3885 - mae: 3.3630
    Epoch 121/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.6091 - mae: 3.8250
    Epoch 122/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8405 - mae: 3.4908
    Epoch 123/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.3710 - mae: 3.5863
    Epoch 124/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.3185 - mae: 3.4333
    Epoch 125/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.5256 - mae: 3.4602
    Epoch 126/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.1493 - mae: 3.7011
    Epoch 127/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.3298 - mae: 3.4219
    Epoch 128/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.4493 - mae: 3.4581
    Epoch 129/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.4722 - mae: 3.7335
    Epoch 130/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.1702 - mae: 3.3830
    Epoch 131/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.7490 - mae: 3.5981
    Epoch 132/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.5575 - mae: 3.6297
    Epoch 133/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8442 - mae: 3.4644
    Epoch 134/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.0603 - mae: 3.8015
    Epoch 135/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.7879 - mae: 3.5697
    Epoch 136/500
    94/94 [==============================] - 1s 10ms/step - loss: 34.1208 - mae: 4.2512
    Epoch 137/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8590 - mae: 3.3860
    Epoch 138/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.5440 - mae: 3.4742
    Epoch 139/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.7436 - mae: 3.6305
    Epoch 140/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8759 - mae: 3.4509
    Epoch 141/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.1879 - mae: 3.8179
    Epoch 142/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.7787 - mae: 3.8260
    Epoch 143/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.4212 - mae: 3.3314
    Epoch 144/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.6372 - mae: 3.4596
    Epoch 145/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.8331 - mae: 3.7653
    Epoch 146/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.4085 - mae: 3.4718
    Epoch 147/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8753 - mae: 3.3954
    Epoch 148/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.6394 - mae: 3.5915
    Epoch 149/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.4298 - mae: 3.6569
    Epoch 150/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.9903 - mae: 3.4993
    Epoch 151/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.9397 - mae: 3.5611
    Epoch 152/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.7871 - mae: 3.4507
    Epoch 153/500
    94/94 [==============================] - 1s 11ms/step - loss: 22.9068 - mae: 3.2723
    Epoch 154/500
    94/94 [==============================] - 1s 10ms/step - loss: 32.4519 - mae: 4.1377
    Epoch 155/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.4658 - mae: 3.5059
    Epoch 156/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.9858 - mae: 3.5521
    Epoch 157/500
    94/94 [==============================] - 1s 11ms/step - loss: 28.2411 - mae: 3.7696
    Epoch 158/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.0365 - mae: 3.7541
    Epoch 159/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.9762 - mae: 3.6673
    Epoch 160/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.9244 - mae: 3.4997
    Epoch 161/500
    94/94 [==============================] - 1s 11ms/step - loss: 28.1464 - mae: 3.7795
    Epoch 162/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.6106 - mae: 3.6423
    Epoch 163/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.1072 - mae: 3.6541
    Epoch 164/500
    94/94 [==============================] - 1s 11ms/step - loss: 27.0120 - mae: 3.6517
    Epoch 165/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.7637 - mae: 3.5666
    Epoch 166/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.5175 - mae: 3.5178
    Epoch 167/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.5091 - mae: 3.5463
    Epoch 168/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.6773 - mae: 3.6401
    Epoch 169/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.4005 - mae: 3.4574
    Epoch 170/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.7860 - mae: 3.6384
    Epoch 171/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.5025 - mae: 3.6622
    Epoch 172/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.7154 - mae: 3.5551
    Epoch 173/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.6108 - mae: 3.8724
    Epoch 174/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.3341 - mae: 3.6128
    Epoch 175/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.9326 - mae: 3.7036
    Epoch 176/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3261 - mae: 3.5004
    Epoch 177/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.9694 - mae: 3.6665
    Epoch 178/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.8343 - mae: 3.5584
    Epoch 179/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.2989 - mae: 3.3465
    Epoch 180/500
    94/94 [==============================] - 1s 11ms/step - loss: 28.4905 - mae: 3.8205
    Epoch 181/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.5862 - mae: 3.7473
    Epoch 182/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.4832 - mae: 3.6500
    Epoch 183/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.1626 - mae: 3.7268
    Epoch 184/500
    94/94 [==============================] - 1s 11ms/step - loss: 29.1907 - mae: 3.8472
    Epoch 185/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.4058 - mae: 3.7901
    Epoch 186/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.6734 - mae: 3.5477
    Epoch 187/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.1436 - mae: 3.3165
    Epoch 188/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.5784 - mae: 3.4814
    Epoch 189/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.9335 - mae: 3.6432
    Epoch 190/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8952 - mae: 3.3730
    Epoch 191/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.2245 - mae: 3.4999
    Epoch 192/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.1765 - mae: 3.5421
    Epoch 193/500
    94/94 [==============================] - 1s 11ms/step - loss: 25.8653 - mae: 3.5292
    Epoch 194/500
    94/94 [==============================] - 1s 11ms/step - loss: 27.5760 - mae: 3.7509
    Epoch 195/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.6377 - mae: 3.8378
    Epoch 196/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.6161 - mae: 3.5772
    Epoch 197/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.2271 - mae: 3.4154
    Epoch 198/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.5101 - mae: 3.6476
    Epoch 199/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.6039 - mae: 3.5499
    Epoch 200/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.1890 - mae: 3.5408
    Epoch 201/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.9331 - mae: 3.5689
    Epoch 202/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.5511 - mae: 3.5106
    Epoch 203/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.2184 - mae: 3.6775
    Epoch 204/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.7156 - mae: 3.6040
    Epoch 205/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.6870 - mae: 3.6461
    Epoch 206/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.5303 - mae: 3.5396
    Epoch 207/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.0843 - mae: 3.4445
    Epoch 208/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.2227 - mae: 3.5107
    Epoch 209/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8154 - mae: 3.3721
    Epoch 210/500
    94/94 [==============================] - 1s 11ms/step - loss: 31.5425 - mae: 4.0568
    Epoch 211/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.0173 - mae: 3.8541
    Epoch 212/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.1425 - mae: 3.5126
    Epoch 213/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.7151 - mae: 3.5312
    Epoch 214/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.1801 - mae: 3.4254
    Epoch 215/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.3274 - mae: 3.4314
    Epoch 216/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.3823 - mae: 3.4334
    Epoch 217/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.5895 - mae: 3.5194
    Epoch 218/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.4853 - mae: 3.5200
    Epoch 219/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.0931 - mae: 3.6700
    Epoch 220/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.6468 - mae: 3.7367
    Epoch 221/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.1429 - mae: 3.6595
    Epoch 222/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.5264 - mae: 3.5623
    Epoch 223/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0070 - mae: 3.3904
    Epoch 224/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.2294 - mae: 3.5674
    Epoch 225/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.4757 - mae: 3.4090
    Epoch 226/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8791 - mae: 3.4775
    Epoch 227/500
    94/94 [==============================] - 1s 10ms/step - loss: 32.0469 - mae: 4.1074
    Epoch 228/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.1500 - mae: 3.7042
    Epoch 229/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.9433 - mae: 3.9616
    Epoch 230/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.1930 - mae: 3.8052
    Epoch 231/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.0952 - mae: 3.5116
    Epoch 232/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.7851 - mae: 3.4832
    Epoch 233/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.2499 - mae: 3.4987
    Epoch 234/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0767 - mae: 3.3815
    Epoch 235/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.4504 - mae: 3.3771
    Epoch 236/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.4582 - mae: 3.5794
    Epoch 237/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8938 - mae: 3.3968
    Epoch 238/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.9608 - mae: 3.5094
    Epoch 239/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.1838 - mae: 3.7474
    Epoch 240/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.7131 - mae: 3.5112
    Epoch 241/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8639 - mae: 3.4666
    Epoch 242/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.0079 - mae: 3.3002
    Epoch 243/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.2510 - mae: 3.4772
    Epoch 244/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.2855 - mae: 3.6271
    Epoch 245/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.8857 - mae: 3.5820
    Epoch 246/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.1518 - mae: 3.6039
    Epoch 247/500
    94/94 [==============================] - 1s 11ms/step - loss: 27.5877 - mae: 3.7420
    Epoch 248/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0418 - mae: 3.4170
    Epoch 249/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.1976 - mae: 3.5965
    Epoch 250/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.4313 - mae: 3.3216
    Epoch 251/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0899 - mae: 3.3962
    Epoch 252/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.5341 - mae: 3.3987
    Epoch 253/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.5665 - mae: 3.5400
    Epoch 254/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.8027 - mae: 3.7816
    Epoch 255/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.7872 - mae: 3.6439
    Epoch 256/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.9763 - mae: 3.4634
    Epoch 257/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.6356 - mae: 3.5347
    Epoch 258/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.1808 - mae: 3.4856
    Epoch 259/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8171 - mae: 3.3711
    Epoch 260/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0356 - mae: 3.4229
    Epoch 261/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.0883 - mae: 3.5008
    Epoch 262/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.1340 - mae: 3.2835
    Epoch 263/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.9700 - mae: 3.4402
    Epoch 264/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.9003 - mae: 3.2565
    Epoch 265/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.0449 - mae: 3.5240
    Epoch 266/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.8021 - mae: 3.5818
    Epoch 267/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.9553 - mae: 3.3282
    Epoch 268/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.2451 - mae: 3.3122
    Epoch 269/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.5062 - mae: 3.4275
    Epoch 270/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.4082 - mae: 3.4580
    Epoch 271/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.8292 - mae: 3.7772
    Epoch 272/500
    94/94 [==============================] - 1s 11ms/step - loss: 28.5629 - mae: 3.7147
    Epoch 273/500
    94/94 [==============================] - 1s 11ms/step - loss: 31.3994 - mae: 4.0349
    Epoch 274/500
    94/94 [==============================] - 1s 11ms/step - loss: 30.5468 - mae: 3.9832
    Epoch 275/500
    94/94 [==============================] - 1s 11ms/step - loss: 25.1862 - mae: 3.5302
    Epoch 276/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.4588 - mae: 3.4657
    Epoch 277/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.1127 - mae: 3.5819
    Epoch 278/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.2470 - mae: 3.4298
    Epoch 279/500
    94/94 [==============================] - 1s 11ms/step - loss: 28.5142 - mae: 3.8234
    Epoch 280/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.3703 - mae: 3.5473
    Epoch 281/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.9542 - mae: 3.4926
    Epoch 282/500
    94/94 [==============================] - 1s 11ms/step - loss: 25.0561 - mae: 3.4476
    Epoch 283/500
    94/94 [==============================] - 1s 11ms/step - loss: 27.4683 - mae: 3.6880
    Epoch 284/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.8598 - mae: 3.6950
    Epoch 285/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.4637 - mae: 3.4042
    Epoch 286/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.6170 - mae: 3.2355
    Epoch 287/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3889 - mae: 3.5317
    Epoch 288/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0072 - mae: 3.4139
    Epoch 289/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.0659 - mae: 3.3809
    Epoch 290/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.9903 - mae: 3.4842
    Epoch 291/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.7329 - mae: 3.4592
    Epoch 292/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.9452 - mae: 3.9829
    Epoch 293/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.2681 - mae: 3.3759
    Epoch 294/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3634 - mae: 3.4866
    Epoch 295/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.4334 - mae: 3.3341
    Epoch 296/500
    94/94 [==============================] - 1s 12ms/step - loss: 25.6551 - mae: 3.5204
    Epoch 297/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.1545 - mae: 3.4777
    Epoch 298/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.1007 - mae: 3.4308
    Epoch 299/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.9339 - mae: 3.7028
    Epoch 300/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.9320 - mae: 3.3780
    Epoch 301/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.9036 - mae: 3.6628
    Epoch 302/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.8731 - mae: 3.2922
    Epoch 303/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.7785 - mae: 3.3844
    Epoch 304/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.2424 - mae: 3.3761
    Epoch 305/500
    94/94 [==============================] - 1s 10ms/step - loss: 30.1701 - mae: 3.9631
    Epoch 306/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.4530 - mae: 3.5568
    Epoch 307/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8632 - mae: 3.5296
    Epoch 308/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.2410 - mae: 3.7389
    Epoch 309/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.2681 - mae: 3.6516
    Epoch 310/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.6078 - mae: 3.6939
    Epoch 311/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.3590 - mae: 3.4296
    Epoch 312/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.2714 - mae: 3.3800
    Epoch 313/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.8082 - mae: 3.7605
    Epoch 314/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.4174 - mae: 3.2680
    Epoch 315/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.5426 - mae: 3.4455
    Epoch 316/500
    94/94 [==============================] - 1s 11ms/step - loss: 25.3144 - mae: 3.5892
    Epoch 317/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.3016 - mae: 3.3341
    Epoch 318/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.3412 - mae: 3.3764
    Epoch 319/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.0915 - mae: 3.7189
    Epoch 320/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.6231 - mae: 3.2579
    Epoch 321/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.9605 - mae: 3.4571
    Epoch 322/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3497 - mae: 3.5212
    Epoch 323/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.1570 - mae: 3.6424
    Epoch 324/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.2786 - mae: 3.3178
    Epoch 325/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.6773 - mae: 3.4654
    Epoch 326/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.9609 - mae: 3.4651
    Epoch 327/500
    94/94 [==============================] - 1s 10ms/step - loss: 30.0262 - mae: 3.9739
    Epoch 328/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3900 - mae: 3.5331
    Epoch 329/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.3178 - mae: 3.5590
    Epoch 330/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.5973 - mae: 3.4849
    Epoch 331/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.7383 - mae: 3.4708
    Epoch 332/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8580 - mae: 3.4851
    Epoch 333/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.5242 - mae: 3.7578
    Epoch 334/500
    94/94 [==============================] - 1s 11ms/step - loss: 25.7438 - mae: 3.5894
    Epoch 335/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.6005 - mae: 3.5210
    Epoch 336/500
    94/94 [==============================] - 1s 11ms/step - loss: 23.3751 - mae: 3.3511
    Epoch 337/500
    94/94 [==============================] - 1s 11ms/step - loss: 22.7100 - mae: 3.2518
    Epoch 338/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8441 - mae: 3.4908
    Epoch 339/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.7366 - mae: 3.2501
    Epoch 340/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.3018 - mae: 3.4422
    Epoch 341/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.8729 - mae: 3.5917
    Epoch 342/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.3061 - mae: 3.4611
    Epoch 343/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8205 - mae: 3.3804
    Epoch 344/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.4362 - mae: 3.7582
    Epoch 345/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.9525 - mae: 3.4029
    Epoch 346/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.4287 - mae: 3.4321
    Epoch 347/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.6987 - mae: 3.7046
    Epoch 348/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.4435 - mae: 3.4070
    Epoch 349/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.1999 - mae: 3.6060
    Epoch 350/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.2569 - mae: 3.5939
    Epoch 351/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.5750 - mae: 3.3275
    Epoch 352/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.8002 - mae: 3.6830
    Epoch 353/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.9110 - mae: 3.7449
    Epoch 354/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0468 - mae: 3.3797
    Epoch 355/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0615 - mae: 3.4310
    Epoch 356/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.0577 - mae: 3.5292
    Epoch 357/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.1762 - mae: 3.5660
    Epoch 358/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.5239 - mae: 3.3296
    Epoch 359/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.4102 - mae: 3.3410
    Epoch 360/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.1689 - mae: 3.2038
    Epoch 361/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.7310 - mae: 3.6928
    Epoch 362/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.7822 - mae: 3.7850
    Epoch 363/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.9571 - mae: 3.4583
    Epoch 364/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.7248 - mae: 3.5362
    Epoch 365/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.6281 - mae: 3.2282
    Epoch 366/500
    94/94 [==============================] - 1s 11ms/step - loss: 27.3060 - mae: 3.7224
    Epoch 367/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.9925 - mae: 3.4165
    Epoch 368/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.6613 - mae: 3.3351
    Epoch 369/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.7138 - mae: 3.4297
    Epoch 370/500
    94/94 [==============================] - 1s 11ms/step - loss: 23.6014 - mae: 3.3474
    Epoch 371/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.7249 - mae: 3.5507
    Epoch 372/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.9875 - mae: 3.3868
    Epoch 373/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.4493 - mae: 3.7408
    Epoch 374/500
    94/94 [==============================] - 1s 11ms/step - loss: 23.9704 - mae: 3.3904
    Epoch 375/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3823 - mae: 3.5224
    Epoch 376/500
    94/94 [==============================] - 1s 11ms/step - loss: 29.7485 - mae: 3.9054
    Epoch 377/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3196 - mae: 3.5283
    Epoch 378/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.9980 - mae: 3.7071
    Epoch 379/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.9652 - mae: 3.5116
    Epoch 380/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.0867 - mae: 3.5106
    Epoch 381/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.2087 - mae: 3.4331
    Epoch 382/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.2586 - mae: 3.2243
    Epoch 383/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.1260 - mae: 3.4435
    Epoch 384/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.4422 - mae: 3.3976
    Epoch 385/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3325 - mae: 3.5337
    Epoch 386/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.9660 - mae: 3.3124
    Epoch 387/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8818 - mae: 3.3984
    Epoch 388/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.8762 - mae: 3.5614
    Epoch 389/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.8365 - mae: 3.5383
    Epoch 390/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.6715 - mae: 3.2457
    Epoch 391/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.0843 - mae: 3.2908
    Epoch 392/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.6245 - mae: 3.3829
    Epoch 393/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.2039 - mae: 3.5702
    Epoch 394/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.1106 - mae: 3.6065
    Epoch 395/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.2101 - mae: 3.6326
    Epoch 396/500
    94/94 [==============================] - 1s 11ms/step - loss: 25.4717 - mae: 3.5351
    Epoch 397/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.5863 - mae: 3.3548
    Epoch 398/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.0480 - mae: 3.4315
    Epoch 399/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0932 - mae: 3.3892
    Epoch 400/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.2962 - mae: 3.5019
    Epoch 401/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.1827 - mae: 3.7590
    Epoch 402/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.7546 - mae: 3.3163
    Epoch 403/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8098 - mae: 3.3646
    Epoch 404/500
    94/94 [==============================] - 1s 11ms/step - loss: 23.3752 - mae: 3.3312
    Epoch 405/500
    94/94 [==============================] - 1s 11ms/step - loss: 23.3666 - mae: 3.3658
    Epoch 406/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.7675 - mae: 3.3601
    Epoch 407/500
    94/94 [==============================] - 1s 11ms/step - loss: 22.9637 - mae: 3.3213
    Epoch 408/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.8314 - mae: 3.5618
    Epoch 409/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.8562 - mae: 3.6547
    Epoch 410/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.9298 - mae: 3.3536
    Epoch 411/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.6950 - mae: 3.4682
    Epoch 412/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.4527 - mae: 3.3450
    Epoch 413/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.7152 - mae: 3.2642
    Epoch 414/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.5040 - mae: 3.3272
    Epoch 415/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.6791 - mae: 3.4757
    Epoch 416/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.6522 - mae: 3.4297
    Epoch 417/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.7841 - mae: 3.4111
    Epoch 418/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.6727 - mae: 3.5512
    Epoch 419/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.5013 - mae: 3.6073
    Epoch 420/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.7649 - mae: 3.3514
    Epoch 421/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.2180 - mae: 3.5045
    Epoch 422/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.7328 - mae: 3.2851
    Epoch 423/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.3644 - mae: 3.4391
    Epoch 424/500
    94/94 [==============================] - 1s 11ms/step - loss: 25.8762 - mae: 3.6015
    Epoch 425/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.4277 - mae: 3.4516
    Epoch 426/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.7114 - mae: 3.6459
    Epoch 427/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.4296 - mae: 3.5963
    Epoch 428/500
    94/94 [==============================] - 1s 10ms/step - loss: 29.4461 - mae: 3.8712
    Epoch 429/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.2760 - mae: 3.6066
    Epoch 430/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.4741 - mae: 3.3469
    Epoch 431/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.6877 - mae: 3.2569
    Epoch 432/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0483 - mae: 3.4186
    Epoch 433/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.0580 - mae: 3.6356
    Epoch 434/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.8620 - mae: 3.4712
    Epoch 435/500
    94/94 [==============================] - 1s 11ms/step - loss: 23.5636 - mae: 3.3511
    Epoch 436/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.6054 - mae: 3.7406
    Epoch 437/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.1498 - mae: 3.4013
    Epoch 438/500
    94/94 [==============================] - 1s 10ms/step - loss: 30.5332 - mae: 3.9683
    Epoch 439/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.6025 - mae: 3.4909
    Epoch 440/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.9251 - mae: 3.7064
    Epoch 441/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.5832 - mae: 3.7051
    Epoch 442/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.8372 - mae: 3.6684
    Epoch 443/500
    94/94 [==============================] - 1s 11ms/step - loss: 28.6340 - mae: 3.5021
    Epoch 444/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.0411 - mae: 3.4071
    Epoch 445/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.7245 - mae: 3.6560
    Epoch 446/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.7418 - mae: 3.3931
    Epoch 447/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.7974 - mae: 3.7652
    Epoch 448/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.3396 - mae: 3.3197
    Epoch 449/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.1308 - mae: 3.4114
    Epoch 450/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.5475 - mae: 3.3680
    Epoch 451/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.3147 - mae: 3.4503
    Epoch 452/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.1076 - mae: 3.3518
    Epoch 453/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.7939 - mae: 3.8183
    Epoch 454/500
    94/94 [==============================] - 1s 11ms/step - loss: 23.0427 - mae: 3.2882
    Epoch 455/500
    94/94 [==============================] - 1s 11ms/step - loss: 23.4348 - mae: 3.3122
    Epoch 456/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8034 - mae: 3.3822
    Epoch 457/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.2211 - mae: 3.5549
    Epoch 458/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.1753 - mae: 3.4144
    Epoch 459/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.3820 - mae: 3.4212
    Epoch 460/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.3013 - mae: 3.4645
    Epoch 461/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.0322 - mae: 3.6171
    Epoch 462/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.7803 - mae: 3.4314
    Epoch 463/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.3819 - mae: 3.3504
    Epoch 464/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.1452 - mae: 3.6070
    Epoch 465/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.9790 - mae: 3.5102
    Epoch 466/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.2090 - mae: 3.4459
    Epoch 467/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.7549 - mae: 3.6618
    Epoch 468/500
    94/94 [==============================] - 1s 10ms/step - loss: 28.4150 - mae: 3.8325
    Epoch 469/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.1973 - mae: 3.6180
    Epoch 470/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.0347 - mae: 3.5888
    Epoch 471/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.9313 - mae: 3.3058
    Epoch 472/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.5941 - mae: 3.4478
    Epoch 473/500
    94/94 [==============================] - 1s 11ms/step - loss: 23.1672 - mae: 3.3172
    Epoch 474/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.7679 - mae: 3.2733
    Epoch 475/500
    94/94 [==============================] - 1s 11ms/step - loss: 25.3391 - mae: 3.5157
    Epoch 476/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.1268 - mae: 3.3965
    Epoch 477/500
    94/94 [==============================] - 1s 11ms/step - loss: 26.3488 - mae: 3.5563
    Epoch 478/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.0451 - mae: 3.6158
    Epoch 479/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.4173 - mae: 3.7040
    Epoch 480/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.6763 - mae: 3.4747
    Epoch 481/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.9664 - mae: 3.4155
    Epoch 482/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.4143 - mae: 3.2561
    Epoch 483/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.1216 - mae: 3.3269
    Epoch 484/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.3370 - mae: 3.4934
    Epoch 485/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.9429 - mae: 3.3677
    Epoch 486/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.8685 - mae: 3.5414
    Epoch 487/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8408 - mae: 3.4060
    Epoch 488/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.8756 - mae: 3.6528
    Epoch 489/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.8095 - mae: 3.3624
    Epoch 490/500
    94/94 [==============================] - 1s 10ms/step - loss: 24.6054 - mae: 3.4412
    Epoch 491/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.9528 - mae: 3.3084
    Epoch 492/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.2477 - mae: 3.3044
    Epoch 493/500
    94/94 [==============================] - 1s 10ms/step - loss: 22.7466 - mae: 3.2731
    Epoch 494/500
    94/94 [==============================] - 1s 10ms/step - loss: 26.4276 - mae: 3.6968
    Epoch 495/500
    94/94 [==============================] - 1s 12ms/step - loss: 27.1720 - mae: 3.7213
    Epoch 496/500
    94/94 [==============================] - 1s 10ms/step - loss: 23.7589 - mae: 3.3997
    Epoch 497/500
    94/94 [==============================] - 1s 10ms/step - loss: 27.8912 - mae: 3.7350
    Epoch 498/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.7745 - mae: 3.5177
    Epoch 499/500
    94/94 [==============================] - 1s 10ms/step - loss: 25.0025 - mae: 3.4904
    Epoch 500/500
    94/94 [==============================] - 1s 11ms/step - loss: 24.7565 - mae: 3.4528
    


```python
forecast = []
results = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))
 
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
```


![png](output_21_0.png)



```python
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
```




    3.1546707




```python
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
mae=history.history['mae']
loss=history.history['loss']

epochs=range(len(loss)) # Get number of epochs

#------------------------------------------------
# Plot MAE and Loss
#------------------------------------------------
plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()

epochs_zoom = epochs[200:]
mae_zoom = mae[200:]
loss_zoom = loss[200:]

#------------------------------------------------
# Plot Zoomed MAE and Loss
#------------------------------------------------
plt.plot(epochs_zoom, mae_zoom, 'r')
plt.plot(epochs_zoom, loss_zoom, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()
```




    <Figure size 432x288 with 0 Axes>




![png](output_23_1.png)



![png](output_23_2.png)



    <Figure size 432x288 with 0 Axes>

