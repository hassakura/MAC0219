import numpy as np
import matplotlib.pyplot as plt

fig= plt.subplots()
plt.title('Full Picture pthread')
plt.xlabel('Tamanho da entrada')
plt.ylabel('Tempo gasto (s)')

input_size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

pth_full_1 = [0.007341383, 0.000845960, 0.001391936, 0.003589773, 0.012148041, 0.046139356, 0.181194705, 0.720696178, 2.875685373, 11.492263291]
pth_full_2 = [0.000968612, 0.001047585, 0.001346804, 0.002576172, 0.007600918, 0.026743875, 0.104566251, 0.415520072, 1.603915511, 6.504294881]
pth_full_4 = [0.000882641, 0.000848280, 0.001192225, 0.002280656, 0.006872089, 0.024869913, 0.101082713, 0.409096277, 1.557313322, 6.258010240]
pth_full_8 = [0.001160374, 0.001070056, 0.001120028, 0.002236987, 0.005752060, 0.020134559, 0.079482785, 0.308349219, 1.226518633, 4.873723125]
pth_full_16 = [0.001057673, 0.001065751, 0.001199627, 0.001691212, 0.003972538, 0.012906732, 0.053638852, 0.206835360, 0.814769476, 3.275115899]
pth_full_32 = [0.001423940, 0.001537306, 0.001673173, 0.002031913, 0.003663288, 0.009821873, 0.037002377, 0.146378578, 0.609411606, 2.404019988]

stdd_pth_full_1_pct = [90.03, 0.71, 0.93, 0.33, 0.12, 0.11, 0.04, 0.03, 0.01, 0.02]
stdd_pth_full_2_pct = [2.95, 1.65, 2.12, 1.05, 0.78, 0.45, 0.89, 0.80, 0.18, 0.81]
stdd_pth_full_4_pct = [1.62, 1.66, 1.03, 0.69, 0.28, 0.05, 0.75, 0.31, 0.80, 1.12]
stdd_pth_full_8_pct = [4.09, 2.98, 2.05, 2.36, 0.53, 0.29, 0.61, 0.21, 0.18, 0.16]
stdd_pth_full_16_pct = [2.32, 2.00, 2.12, 1.75, 1.54, 0.63, 2.15, 0.55, 0.54, 0.34]
stdd_pth_full_32_pct = [2.08, 3.17, 2.40, 1.81, 1.09, 2.84, 2.73, 0.68, 0.91, 0.50]

stdd_pth_full_1 = [(pth_full_1[x] * stdd_pth_full_1_pct[x] / 100) for x in range (0, 10)]
stdd_pth_full_2 = [(pth_full_2[x] * stdd_pth_full_2_pct[x] / 100) for x in range (0, 10)]
stdd_pth_full_4 = [(pth_full_4[x] * stdd_pth_full_4_pct[x] / 100) for x in range (0, 10)]
stdd_pth_full_8 = [(pth_full_8[x] * stdd_pth_full_8_pct[x] / 100) for x in range (0, 10)]
stdd_pth_full_16 = [(pth_full_16[x] * stdd_pth_full_16_pct[x] / 100) for x in range (0, 10)]
stdd_pth_full_32 = [(pth_full_32[x] * stdd_pth_full_32_pct[x] / 100) for x in range (0, 10)]

bar_full_width = 1 / 7
index = np.arange(10)

bar_full_1 = plt.bar(index, pth_full_1, bar_full_width, label = '1 thread', yerr = stdd_pth_full_1)
bar_full_2 = plt.bar(index + bar_full_width, pth_full_2, bar_full_width, label = '2 threads', yerr = stdd_pth_full_2)
bar_full_4 = plt.bar(index + 2 * bar_full_width, pth_full_4, bar_full_width, label = '4 threads', yerr = stdd_pth_full_4)
bar_full_8 = plt.bar(index + 3 * bar_full_width, pth_full_8, bar_full_width, label = '8 threads', yerr = stdd_pth_full_8)
bar_full_16 = plt.bar(index + 4 * bar_full_width, pth_full_16, bar_full_width, label = '16 threads', yerr = stdd_pth_full_16)
bar_full_32 = plt.bar(index + 5 * bar_full_width, pth_full_32, bar_full_width, label = '32 threads', yerr = stdd_pth_full_32)

plt.xticks(index + 2.5 * bar_full_width, input_size)
plt.legend(fontsize = 'x-large')
plt.grid()
plt.show()