import numpy as np
import matplotlib.pyplot as plt

fig= plt.subplots()
plt.title('Elephant Valley pthread')
plt.xlabel('Tamanho da entrada')
plt.ylabel('Tempo gasto (s)')

input_size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

pth_elephant_1 = [0.001005060, 0.001943597, 0.005734699, 0.020489631, 0.079562400, 0.315301019, 1.258185969, 5.026180836, 20.106780819, 81.181665197]
pth_elephant_2 = [0.001086002, 0.001757217, 0.003659727, 0.012258692, 0.045457151, 0.177593562, 0.707015861, 2.806535954, 11.341790358, 45.294231822]
pth_elephant_4 = [0.000983763, 0.001163847, 0.002338220, 0.006992166, 0.025594707, 0.100743140, 0.394875665, 1.646628861, 6.367647440, 26.711731595]
pth_elephant_8 = [0.001105832, 0.001132811, 0.001693929, 0.004247958, 0.014807217, 0.054712422, 0.251525071, 1.006903399, 3.996539720, 16.009348964]
pth_elephant_16 = [0.001041184, 0.001292822, 0.001539835, 0.003861766, 0.013611538, 0.052547608, 0.231009970, 0.910567991, 3.678390737, 14.648509657]
pth_elephant_32 = [0.001417263, 0.001620617, 0.002058744, 0.004235412, 0.012534250, 0.048014617, 0.227304661, 0.896609197, 3.576922553, 14.252888280]

stdd_pth_elephant_1_pct = [1.12, 0.71, 0.23, 0.07, 0.03, 0.03, 0.01, 0.04, 0.02, 0.48]
stdd_pth_elephant_2_pct = [1.98, 1.49, 1.01, 1.37, 0.34, 0.10, 0.92, 0.52, 0.60, 0.62]
stdd_pth_elephant_4_pct = [1.42, 0.60, 0.81, 0.85, 0.12, 0.56, 0.04, 1.99, 0.50, 1.69]
stdd_pth_elephant_8_pct = [1.85, 2.12, 2.09, 1.01, 1.58, 0.96, 0.41, 0.32, 0.37, 0.27]
stdd_pth_elephant_16_pct = [1.52, 2.70, 0.73, 0.39, 0.63, 1.46, 0.43, 0.37, 0.49, 0.26]
stdd_pth_elephant_32_pct = [1.84, 1.17, 1.34, 0.90, 0.69, 5.40, 0.25, 0.13, 0.11, 0.06]

stdd_pth_elephant_1 = [(pth_elephant_1[x] * stdd_pth_elephant_1_pct[x] / 100) for x in range (0, 10)]
stdd_pth_elephant_2 = [(pth_elephant_2[x] * stdd_pth_elephant_2_pct[x] / 100) for x in range (0, 10)]
stdd_pth_elephant_4 = [(pth_elephant_4[x] * stdd_pth_elephant_4_pct[x] / 100) for x in range (0, 10)]
stdd_pth_elephant_8 = [(pth_elephant_8[x] * stdd_pth_elephant_8_pct[x] / 100) for x in range (0, 10)]
stdd_pth_elephant_16 = [(pth_elephant_16[x] * stdd_pth_elephant_16_pct[x] / 100) for x in range (0, 10)]
stdd_pth_elephant_32 = [(pth_elephant_32[x] * stdd_pth_elephant_32_pct[x] / 100) for x in range (0, 10)]

bar_elephant_width = 1 / 7
index = np.arange(10)

bar_elephant_1 = plt.bar(index, pth_elephant_1, bar_elephant_width, label = '1 thread', yerr = stdd_pth_elephant_1)
bar_elephant_2 = plt.bar(index + bar_elephant_width, pth_elephant_2, bar_elephant_width, label = '2 threads', yerr = stdd_pth_elephant_2)
bar_elephant_4 = plt.bar(index + 2 * bar_elephant_width, pth_elephant_4, bar_elephant_width, label = '4 threads', yerr = stdd_pth_elephant_4)
bar_elephant_8 = plt.bar(index + 3 * bar_elephant_width, pth_elephant_8, bar_elephant_width, label = '8 threads', yerr = stdd_pth_elephant_8)
bar_elephant_16 = plt.bar(index + 4 * bar_elephant_width, pth_elephant_16, bar_elephant_width, label = '16 threads', yerr = stdd_pth_elephant_16)
bar_elephant_32 = plt.bar(index + 5 * bar_elephant_width, pth_elephant_32, bar_elephant_width, label = '32 threads', yerr = stdd_pth_elephant_32)

plt.xticks(index + 2.5 * bar_elephant_width, input_size)
plt.legend(fontsize = 'x-large')
plt.grid()
plt.show()