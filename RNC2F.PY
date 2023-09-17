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
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(1))

# Train the model
history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)

# Gráfica de la pérdida durante el entrenamiento
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Predict the conversion for a new value
celsius_value = 100
fahrenheit_value = model.predict([celsius_value])
print("The predicted Fahrenheit value for", \
      f"{celsius_value} degrees Celsius is", \
      f"{fahrenheit_value} degrees Fahrenheit.")

