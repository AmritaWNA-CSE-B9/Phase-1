import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./FPS_CPU_FINAL.csv")
frames = df['Frame']
fps_sma = df['SMA']
fps_wma = df['WMA']
plt.xlabel("Frames")
plt.ylabel("FPS (frames per second)")
plt.plot(frames, fps_sma, label="SMA")
plt.plot(frames, fps_wma, label="WMA")
plt.legend(title="Result with window size 1000",loc=4)
plt.grid()
plt.savefig("../CPU_FPS_trend.png")

print("process complete!")