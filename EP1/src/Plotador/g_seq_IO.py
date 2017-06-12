import numpy as np
import matplotlib.pyplot as plt

input_size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

#Grafico mandelbrot_seq

#fig = plt.figure(1)

#fig.suptitle('1 Thread')
fig = plt.subplots()
plt.title('Mandelbrot Sequencial')
plt.xlabel('Tamanho da entrada')
plt.ylabel('Tempo gasto (s)')


seq_elephant = [0.000911988, 0.001947579, 0.005810300, 0.021446828, 0.084371574, 0.335577493, 1.335835770, 5.505320427, 21.618584923, 88.499974297]
seq_full = [0.004342094, 0.001383223, 0.003042832, 0.004556242, 0.016193319, 0.066050534, 0.262612604, 1.022417810, 4.340282105, 17.951102282]
seq_seahorse = [0.000922286, 0.001980260, 0.006083679, 0.022501505, 0.089204788, 0.353246800, 1.414779058, 5.853969199, 23.400544848, 93.346398082]
seq_triple_spiral = [0.001118846, 0.002167216, 0.006845384, 0.025581389, 0.100892618, 0.400364173, 1.598103541, 6.405259670, 26.707453872, 108.143847179]

stdd_seq_elephant_pct = [0.69, 0.50, 0.07, 0.19, 0.22, 0.12, 0.05, 0.57, 0.44, 0.67]
stdd_seq_full_pct = [84.52, 38.94, 19.96, 1.51, 0.55, 3.24, 2.25, 0.21, 1.99, 1.16]
stdd_seq_seahorse_pct = [0.68, 0.42, 0.13, 0.11, 0.27, 0.13, 0.31, 0.95, 1.52, 1.13]
stdd_seq_triple_spiral_pct = [11.56, 0.40, 0.17, 0.13, 0.21, 0.13, 0.07, 0.13, 1.35, 1.85]

stdd_seq_elephant = [(seq_elephant[x] * stdd_seq_elephant_pct[x] / 100) for x in range (0, 10)]
stdd_seq_full = [(seq_full[x] * stdd_seq_full_pct[x] / 100) for x in range (0, 10)]
stdd_seq_seahorse = [(seq_seahorse[x] * stdd_seq_seahorse_pct[x] / 100) for x in range (0, 10)]
stdd_seq_triple_spiral = [(seq_triple_spiral[x] * stdd_seq_triple_spiral_pct[x] / 100) for x in range (0, 10)]

bar_width = 0.2
index = np.arange(10)



bar_elephant = plt.bar(index, seq_elephant, bar_width, label = 'Elephant Valley', yerr = stdd_seq_elephant)
bar_full = plt.bar(index + bar_width, seq_full, bar_width, label = 'Full Picture', yerr = stdd_seq_full)
bar_seahorse = plt.bar(index + 2 * bar_width, seq_seahorse, bar_width, label = 'Seahorse Valley', yerr = stdd_seq_seahorse)
bar_triple_spiral = plt.bar(index + 3 * bar_width, seq_triple_spiral, bar_width, label = 'Triple Spiral Valley', yerr = stdd_seq_triple_spiral)




plt.xticks(index + 1.5 * bar_width, input_size)
plt.legend(fontsize = 'x-large')
plt.grid()
plt.show()