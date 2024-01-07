import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./FPS_CUDA_FINAL.csv")
frames = df['Frame']
fps_sma = df['SMA']
fps_wma = df['WMA']
print(fps_sma.max(), " ", fps_wma.max())
plt.xlabel("Frames")
plt.ylabel("FPS (frames per second)")
plt.plot(frames, fps_sma, label="SMA")
plt.plot(frames, fps_wma, label="WMA")
plt.legend(title="Result with window size 1000",loc=4)
plt.grid()
plt.savefig("../CUDA_FPS_trend.png")

print("process complete!")