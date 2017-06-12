import numpy as np
import matplotlib.pyplot as plt

fig= plt.subplots()
plt.title('Seahorse Valley pthread')
plt.xlabel('Tamanho da entrada')
plt.ylabel('Tempo gasto (s)')

input_size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

pth_seahorse_1 = [0.000994672, 0.002015780, 0.005967515, 0.021625448, 0.084258653, 0.334624789, 1.333644055, 5.332049912, 21.315033572, 86.717775737]
pth_seahorse_2 = [0.001084228, 0.001726127, 0.004373411, 0.014021247, 0.053637499, 0.207435491, 0.824392659, 3.467980936, 13.331221560, 53.509416120]
pth_seahorse_4 = [0.000950265, 0.001222456, 0.002421871, 0.007470943, 0.029043276, 0.108233855, 0.429347179, 1.709345779, 7.376534103, 28.753318355]
pth_seahorse_8 = [0.001120474, 0.001234999, 0.001686768, 0.004514717, 0.016155491, 0.057169552, 0.264018623, 1.062742442, 4.157763694, 16.876527407]
pth_seahorse_16 = [0.001033653, 0.001186121, 0.001668335, 0.004258545, 0.014004131, 0.052341472, 0.240568073, 0.968138345, 3.845148994, 15.366701063]
pth_seahorse_32 = [0.001428456, 0.001587117, 0.001995032, 0.004241356, 0.013233294, 0.048918420, 0.238423671, 0.948915343, 3.782724963, 15.069927655]

stdd_pth_seahorse_1_pct = [1.46, 0.74, 0.19, 0.08, 0.04, 0.03, 0.01, 0.03, 0.01, 0.56]
stdd_pth_seahorse_2_pct = [1.78, 0.69, 1.82, 0.38, 0.22, 0.23, 0.11, 0.77, 0.10, 0.28]
stdd_pth_seahorse_4_pct = [1.20, 0.88, 0.30, 0.32, 1.35, 0.17, 0.10, 0.03, 2.42, 1.89]
stdd_pth_seahorse_8_pct = [1.49, 1.87, 2.48, 0.94, 2.37, 1.53, 0.36, 0.75, 1.18, 0.35]
stdd_pth_seahorse_16_pct = [2.87, 1.86, 1.30, 2.21, 0.93, 0.58, 0.17, 0.37, 0.20, 0.20]
stdd_pth_seahorse_32_pct = [2.17, 1.90, 1.19, 1.16, 0.82, 0.47, 0.17, 0.19, 0.15, 0.09]

stdd_pth_seahorse_1 = [(pth_seahorse_1[x] * stdd_pth_seahorse_1_pct[x] / 100) for x in range (0, 10)]
stdd_pth_seahorse_2 = [(pth_seahorse_2[x] * stdd_pth_seahorse_2_pct[x] / 100) for x in range (0, 10)]
stdd_pth_seahorse_4 = [(pth_seahorse_4[x] * stdd_pth_seahorse_4_pct[x] / 100) for x in range (0, 10)]
stdd_pth_seahorse_8 = [(pth_seahorse_8[x] * stdd_pth_seahorse_8_pct[x] / 100) for x in range (0, 10)]
stdd_pth_seahorse_16 = [(pth_seahorse_16[x] * stdd_pth_seahorse_16_pct[x] / 100) for x in range (0, 10)]
stdd_pth_seahorse_32 = [(pth_seahorse_32[x] * stdd_pth_seahorse_32_pct[x] / 100) for x in range (0, 10)]

bar_seahorse_width = 1 / 7
index = np.arange(10)

bar_seahorse_1 = plt.bar(index, pth_seahorse_1, bar_seahorse_width, label = '1 thread', yerr = stdd_pth_seahorse_1)
bar_seahorse_2 = plt.bar(index + bar_seahorse_width, pth_seahorse_2, bar_seahorse_width, label = '2 threads', yerr = stdd_pth_seahorse_2)
bar_seahorse_4 = plt.bar(index + 2 * bar_seahorse_width, pth_seahorse_4, bar_seahorse_width, label = '4 threads', yerr = stdd_pth_seahorse_4)
bar_seahorse_8 = plt.bar(index + 3 * bar_seahorse_width, pth_seahorse_8, bar_seahorse_width, label = '8 threads', yerr = stdd_pth_seahorse_8)
bar_seahorse_16 = plt.bar(index + 4 * bar_seahorse_width, pth_seahorse_16, bar_seahorse_width, label = '16 threads', yerr = stdd_pth_seahorse_16)
bar_seahorse_32 = plt.bar(index + 5 * bar_seahorse_width, pth_seahorse_32, bar_seahorse_width, label = '32 threads', yerr = stdd_pth_seahorse_32)

plt.xticks(index + 2.5 * bar_seahorse_width, input_size)
plt.legend(fontsize = 'x-large')
plt.grid()
plt.show()