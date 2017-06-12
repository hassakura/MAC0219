import numpy as np
import matplotlib.pyplot as plt

input_size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

#Grafico mandelbrot_seq

#fig = plt.figure(1)

#fig.suptitle('1 Thread')
fig = plt.subplots()
plt.title('Mandelbrot Sequencial sem I/O e Aloc. Memória')
plt.xlabel('Tamanho da entrada')
plt.ylabel('Tempo gasto (s)')


seq_elephant_noIO = [0.000828969, 0.001730687, 0.005445275, 0.020440713, 0.079885698, 0.315624210, 1.264212856, 5.044261568, 20.142546251, 80.567885132]
seq_full_noIO = [0.000522615, 0.000707057, 0.001205509, 0.003340191, 0.011976808, 0.046012920, 0.181370767, 0.721708696, 2.896251346, 11.512569715]
seq_seahorse_noIO = [0.000846648, 0.001797932, 0.005698106, 0.021556756, 0.084291041, 0.335083699, 1.335688790, 5.340266308, 21.352286005, 85.428445966]
seq_triple_spiral_noIO = [0.000878355, 0.001974630, 0.006459315, 0.024534558, 0.096012499, 0.381568825, 1.522197403, 6.082120414, 24.327146013, 97.428475693]

stdd_seq_elephant_noIO_pct = [1.02, 0.59, 0.19, 0.06, 0.08, 0.05, 0.31, 0.18, 0.11, 0.04]
stdd_seq_full_noIO_pct = [3.41, 1.62, 1.14, 0.39, 0.16, 0.13, 0.13, 0.08, 0.38, 0.02]
stdd_seq_seahorse_noIO_pct = [1.15, 0.67, 0.20, 0.15, 0.06, 0.09, 0.06, 0.01, 0.03, 0.04]
stdd_seq_triple_spiral_noIO_pct = [0.68, 0.77, 0.19, 0.13, 0.05, 0.08, 0.05, 0.02, 0.05, 0.05]

stdd_seq_elephant_noIO = [(seq_elephant_noIO[x] * stdd_seq_elephant_noIO_pct[x] / 100) for x in range (0, 10)]
stdd_seq_full_noIO = [(seq_full_noIO[x] * stdd_seq_full_noIO_pct[x] / 100) for x in range (0, 10)]
stdd_seq_seahorse_noIO = [(seq_seahorse_noIO[x] * stdd_seq_seahorse_noIO_pct[x] / 100) for x in range (0, 10)]
stdd_seq_triple_spiral_noIO = [(seq_triple_spiral_noIO[x] * stdd_seq_triple_spiral_noIO_pct[x] / 100) for x in range (0, 10)]

bar_width = 0.2
index = np.arange(10)



bar_elephant_noIO = plt.bar(index, seq_elephant_noIO, bar_width, label = 'Elephant Valley', yerr = stdd_seq_elephant_noIO)
bar_full_noIO = plt.bar(index + bar_width, seq_full_noIO, bar_width, label = 'Full Picture', yerr = stdd_seq_full_noIO)
bar_seahorse_noIO = plt.bar(index + 2 * bar_width, seq_seahorse_noIO, bar_width, label = 'Seahorse Valley', yerr = stdd_seq_seahorse_noIO)
bar_triple_spiral_noIO = plt.bar(index + 3 * bar_width, seq_triple_spiral_noIO, bar_width, label = 'Triple Spiral Valley', yerr = stdd_seq_triple_spiral_noIO)




plt.xticks(index + 1.5 * bar_width, input_size)
plt.legend(fontsize = 'x-large')
plt.grid()
plt.show()