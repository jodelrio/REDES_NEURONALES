import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
historial = model.fit(celsius, fahrenheit, epochs=500, verbose=0)

# Resultado de funcion de perdida
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
