import numpy as np
import matplotlib.pyplot as plt

fig= plt.subplots()
plt.title('Cifra XOR')
plt.xlabel('Tamanho da entrada (milhões de linhas)')
plt.ylabel('Tempo gasto (s)')

input_size = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

xor_avg_sc = [0.022754962, 0.041498427, 0.078147693, 0.190323915, 0.378360244, 0.763527894, 1.959247464, 4.008736518]
xor_avg_cuda = [0.079463344, 0.081447075, 0.090849024, 0.118383582, 0.162007940, 0.234788659, 0.535577507, 1.086247237]

xor_stdd_sc_pct = [3.75, 2.05, 0.35, 0.69, 0.51, 0.68, 0.69, 1.12]
xor_stdd_cuda_pct = [3.04, 0.51, 0.32, 0.34, 0.21, 0.22, 0.14, 0.16]

xor_stdd_sc = [(xor_avg_sc[x] * xor_stdd_sc_pct[x] / 100) for x in range (0, 8)]
xor_stdd_cuda = [(xor_avg_cuda[x] * xor_stdd_cuda_pct[x] / 100) for x in range (0, 8)]

bar_full_width = 1 / 3
index = np.arange(8)

bar_xor_sc = plt.bar(index, xor_avg_sc, bar_full_width, label = 'Sequencial', yerr = xor_stdd_sc)
bar_xor_cuda = plt.bar(index + bar_full_width, xor_avg_cuda, bar_full_width, label = 'CUDA', yerr = xor_stdd_cuda)

plt.xticks(index + 0.5 * bar_full_width, input_size)
plt.legend(fontsize = 'x-large')
plt.grid()
plt.show()