import pandas as pd
import matplotlib.pyplot as plt

# Example loss data
loss_values = [0.5, 0.6, 0.55, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]

# Convert list to Pandas DataFrame
df = pd.DataFrame(loss_values, columns=['Loss'])
# Apply a moving average; window size can be adjusted as needed
window_size = 3
df['Smoothed'] = df['Loss'].rolling(window=window_size, center=True).mean()
plt.figure(figsize=(10, 6))

# Plot original loss data
plt.plot(df['Loss'], label='Original Loss', color='blue', alpha=0.5)

# Plot smoothed loss data
plt.plot(df['Smoothed'], label='Smoothed Loss', color='red')

# Fill the area under the smoothed curve, with transparency
plt.fill_between(df.index, df['Smoothed'], alpha=0.3, color='red')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve with Smoothed Loss (Moving Average)')
plt.legend()

# Show the plot
plt.show()
