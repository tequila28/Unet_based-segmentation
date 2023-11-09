import matplotlib.pyplot as plt
import numpy as np

# 定义文件名和标签
filenames = ['road_logs/seunet(road)_val_log.txt', 'road_logs/resunet(road)_val_log.txt',
             'road_logs/seresunet(road)_val_log.txt', 'road_logs/unet(road)_val_log.txt']
labels = ['seunet', 'resunet', 'seresunet', 'unet']

# 初始化存储miou的列表
miou_values = [[] for _ in range(len(filenames))]

# 生成横轴数据：从1000到27000，间隔为1000
x = np.arange(1000, 28000, 1000)

# 读取每个文件中的数据
for i, filename in enumerate(filenames):
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'Epoch' in line:
                parts = line.split(',')
                for part in parts:
                    if 'miou' in part:
                        miou = float(part.split(':')[1])
                        miou_values[i].append(miou)

# 绘制四条miou曲线
for i in range(len(filenames)):
    plt.plot(x[:len(miou_values[i])], miou_values[i], label=labels[i])

# 添加标签和标题
plt.xlabel('Iteration')
plt.ylabel('miou')
plt.title('miou vs. Iteration')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
