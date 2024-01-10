import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./FPS_CPU_FINAL.csv")
frames = df['Frame']
fps_sma = df['SMA']
fps_wma = df['WMA']
plt.xlabel("Frame no.",fontsize=15)
plt.ylabel("FPS", fontsize=15)
plt.plot(frames, fps_sma, label="SMA")
plt.plot(frames, fps_wma, label="WMA")
plt.legend(title="Result with window size 1000",loc=4, fontsize=14, title_fontsize=15)
plt.grid()
plt.savefig("../CPU_FPS_trend.png")

print("process complete!")
