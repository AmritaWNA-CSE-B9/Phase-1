import pandas as pd
import matplotlib.pyplot as plt


df_sma = pd.read_csv("./SMA/FPS_CUDA_SMA.csv")
df_wma = pd.read_csv("./WMA/FPS_CUDA_WMA.csv")
frames = df_wma['Frame']
fps_sma = df_sma['SMA']
fps_wma = df_wma['WMA']
plt.xlabel("Frames")
plt.ylabel("FPS (frames per second)")
plt.plot(frames, fps_sma, label="SMA")
plt.plot(frames, fps_wma, label="WMA")
plt.legend(title="Result with window size 1000",loc=4)
plt.savefig("../CPU_FPS_trend.png")

final = pd.concat([df_wma, fps_sma], axis=1)
final.to_csv('../final_FPS_CPU.csv')
print("process complete!")