import matplotlib.pyplot as plt
# import pandas as pd


# df = pd.read_csv("./FPS_CPU_FINAL.csv")
# frames = df['Frame']
# fps_sma = df['SMA']
# fps_wma = df['WMA']
# plt.xlabel("Frame no.",fontsize=15)
# plt.ylabel("FPS", fontsize=15)
# plt.plot(frames, fps_sma, label="SMA")
# plt.plot(frames, fps_wma, label="WMA")
# plt.legend(title="Result with window size 1000",loc=4, fontsize=14, title_fontsize=15)
# plt.grid()
# plt.savefig("../CPU_FPS_trend.png")

# print("process complete!")
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./FPS_CUDA_FINAL.csv")
fig, ax = plt.subplots(figsize=(20, 16))
ax.tick_params(axis='both', labelsize=60)
ax.plot(data["Frame"], data["SMA"], marker='o', linestyle='-', label='SMA')
ax.plot(data["Frame"], data["WMA"], marker='o', linestyle='-', label='WMA')
ax.grid(linewidth=4)
ax.set_xlabel("Frames", fontsize=70)
ax.set_ylabel("FPS", fontsize=70)
ax.legend(title='Result with window size 1000', fontsize=45, title_fontsize=60, loc='lower right')
plt.savefig("../CUDA_FPS_trend.png", bbox_inches='tight')
