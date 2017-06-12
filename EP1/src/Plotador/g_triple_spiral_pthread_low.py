import numpy as np
import matplotlib.pyplot as plt

fig= plt.subplots()
plt.title('Triple Spiral Valley pthread')
plt.xlabel('Tamanho da entrada = 16')
plt.ylabel('Tempo gasto (s)')

input_size = [1]

pth_triple_spiral_low = [0.001023408,0.001184869,0.000962242,0.001207627,0.001060821,0.001478282]

bar_triple_spiral_width = 1 / 7

b1 = plt.bar(bar_triple_spiral_width, pth_triple_spiral_low[0], bar_triple_spiral_width, label = '1 thread', yerr = 0.78 * pth_triple_spiral_low[0] / 100)
b1 = plt.bar(2 * bar_triple_spiral_width, pth_triple_spiral_low[1], bar_triple_spiral_width, label = '2 threads', yerr = 3.42 * pth_triple_spiral_low[1] / 100)
b1 = plt.bar(3 * bar_triple_spiral_width, pth_triple_spiral_low[2], bar_triple_spiral_width, label = '4 threads', yerr = 1.67 * pth_triple_spiral_low[2] / 100)
b1 = plt.bar(4 * bar_triple_spiral_width, pth_triple_spiral_low[3], bar_triple_spiral_width, label = '8 threads', yerr = 8.92 * pth_triple_spiral_low[3] / 100)
b1 = plt.bar(5 * bar_triple_spiral_width, pth_triple_spiral_low[4], bar_triple_spiral_width, label = '16 threads', yerr = 1.03 * pth_triple_spiral_low[4] / 100)
b1 = plt.bar(6 * bar_triple_spiral_width, pth_triple_spiral_low[5], bar_triple_spiral_width, label = '32 threads', yerr = 2.13 * pth_triple_spiral_low[5] / 100)



plt.xticks([])
plt.legend(fontsize = 'x-large')
plt.grid()
plt.show()

