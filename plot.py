import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('inference_times.csv')

plt.figure(figsize=(10, 5))
plt.plot(data['Hidden Layer Size'], data['Time (ms)'], marker='o', linestyle='-')
plt.title('Inference Time vs Hidden Layer Size')
plt.xlabel('Hidden Layer Size')
plt.ylabel('Inference Time (ms)')
plt.grid(True)
plt.savefig('inference_plot.png')
