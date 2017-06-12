import numpy as np
import matplotlib.pyplot as plt

fig= plt.subplots()
plt.title('Algoritmo ROT-13')
plt.xlabel('Tamanho da entrada (milhões de linhas)')
plt.ylabel('Tempo gasto (s)')

input_size = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

rot13_avg_sc = [0.059467528, 0.110140389, 0.216796784, 0.540986542, 1.063779236, 2.117913542, 5.389230445, 10.621004347]
rot13_avg_cuda = [0.074911309, 0.078914708, 0.086681030, 0.109220427, 0.148574880, 0.215440677, 0.413337468, 0.742699101]

rot13_stdd_sc_pct = [2.91, 0.26, 0.69, 0.53, 0.31, 1.00, 1.07, 1.14]
rot13_stdd_cuda_pct = [0.77, 0.71, 0.35, 0.27, 0.19, 0.10, 0.11, 0.14]

rot13_stdd_sc = [(rot13_avg_sc[x] * rot13_stdd_sc_pct[x] / 100) for x in range (0, 8)]
rot13_stdd_cuda = [(rot13_avg_cuda[x] * rot13_stdd_cuda_pct[x] / 100) for x in range (0, 8)]

bar_full_width = 1 / 3
index = np.arange(8)

bar_rot13_sc = plt.bar(index, rot13_avg_sc, bar_full_width, label = 'Sequencial', yerr = rot13_stdd_sc)
bar_rot13_cuda = plt.bar(index + bar_full_width, rot13_avg_cuda, bar_full_width, label = 'CUDA', yerr = rot13_stdd_cuda)

plt.xticks(index + 0.5 * bar_full_width, input_size)
plt.legend(fontsize = 'x-large')
plt.grid()
plt.show()