import numpy as np
import tensorflow as tf

# Input data
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Create a model with one layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Train the model
history = model.fit(celsius, fahrenheit, epochs=500, verbose=0)

# Predict the conversion for a new value
celsius_value = 100
fahrenheit_value = model.predict([celsius_value])
print(f"The predicted Fahrenheit value for {celsius_value} degrees Celsius is {fahrenheit_value} degrees Fahrenheit.")

