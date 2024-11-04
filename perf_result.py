"""Plot the profiling result"""

import matplotlib.pyplot as plt
import pandas as pd

# load result from tsv file
file_name = "perf_result.tsv"

execution_times = pd.read_csv(file_name, sep='\t')

# Plot a bar chart
plt.figure(figsize=(6, 6))
# use coolwarm to indicate the time
colors = plt.cm.coolwarm(execution_times['total_duration_ms'] / max(execution_times['total_duration_ms']))
plt.bar(execution_times['name'], execution_times['total_duration_ms'], color=colors, width=0.5)
plt.xlabel('Function')
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time of Functions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()