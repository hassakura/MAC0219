import numpy as np
import matplotlib.pyplot as plt

fig= plt.subplots()
plt.title('Base64')
plt.xlabel('Tamanho da entrada (milhões de linhas)')
plt.ylabel('Tempo gasto (s)')

input_size = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

base64_avg_sc =   [0.030123984, 0.040763553, 0.052374894, 0.676917352, 1.026378344, 1.473691023, 1.959247464, 2.748518333]
base64_avg_cuda = [0.258364844, 0.249846432, 0.257893721, 0.227836384, 0.379467182, 0.520838490, 0.664152638, 0.797890100]

base64_stdd_sc_pct = [3.73, 2.54, 0.12, 0.45, 0.57, 0.61, 0.69, 1.08]
base64_stdd_cuda_pct = [3.43, 0.50, 0.31, 0.33, 0.20, 0.25, 0.12, 0.15]

base64_stdd_sc = [(base64_avg_sc[x] * base64_stdd_sc_pct[x] / 100) for x in range (0, 8)]
base64_stdd_cuda = [(base64_avg_cuda[x] * base64_stdd_cuda_pct[x] / 100) for x in range (0, 8)]

bar_full_width = 1 / 3
index = np.arange(8)

bar_base64_sc = plt.bar(index, base64_avg_sc, bar_full_width, label = 'Sequencial', yerr = base64_stdd_sc)
bar_base64_cuda = plt.bar(index + bar_full_width, base64_avg_cuda, bar_full_width, label = 'CUDA', yerr = base64_stdd_cuda)

plt.xticks(index + 0.5 * bar_full_width, input_size)
plt.legend(fontsize = 'x-large')
plt.grid()
plt.show()