Hélio Assakura - 8941064
Gabriel Baptista - 8941300

EP2: Criptografia em GPUs usando CUDA

Algoritmos escolhidos: ROT-13, Cifra XOR e base64

Execução:

	(ROT-13 Sequencial) Para executar o programa, basta compilar usando o Makefile (make) e executar da seguinte forma:

		./rot-13 [arquivo de texto]

		Em que [arquivo de texto] é o nome do arquivo contendo o texto a ser encriptado.

		O programa executará o algoritmo 2 vezes, pois ROT-13(ROT-13(texto)) = texto. O código é idêntico ao disponibilizado no github (https://github.com/phrb/MAC5742-0219-EP2/blob/master/src/crypto-algorithms/), apenas tendo o nome de um arquivo como entrada.

	(ROT-13 CUDA) Para executar o programa, basta compilar usando o Makefile (make) e executar da seguinte forma:

		./cuda_rot13 [arquivo de texto]

		Em que [arquivo de texto] é o nome do arquivo contendo o texto a ser encriptado.

		O programa executará o algoritmo 2 vezes, pois ROT-13(ROT-13(texto)) = texto.

		Caso deseje verificar se o algoritmo converte corretamente a entrada, há algumas linhas de código comentadas, que farão o mesmo teste presente no código disponibilizado no github do monitor (https://github.com/phrb/MAC5742-0219-EP2/blob/master/src/crypto-algorithms/rot-13_test.c). Basta passar como entrada o arquivo "f.txt" e retirar os comentários do código na função rot13_test().

	(Cifra XOR Sequencial) Para executar o programa, basta compilar usando o Makefile (make) e executar da seguinte forma:

		./xor [arquivo de texto] [chave]

		Em que [arquivo de texto] é o nome do arquivo contendo o texto a ser encriptado e [chave] é o arquivo de texto contendo a chave de encriptação.

		O programa executará o algoritmo 2 vezes, pois XOR(XOR(texto)) = texto, se usada a mesma chave. Isso se deve a tabela-verdade da operação (tabela e outras informações podem ser encontradas em https://en.wikipedia.org/wiki/Exclusive_or).

	(Cifra XOR CUDA)Para executar o programa, basta compilar usando o Makefile (make) e executar da seguinte forma:

		./cuda_xor [arquivo de texto] [chave]

		Em que [arquivo de texto] é o nome do arquivo contendo o texto a ser encriptado e [chave] é o arquivo de texto contendo a chave de encriptação.

		O programa executará o algoritmo 2 vezes, pois XOR(XOR(texto)) = texto, se usada a mesma chave. Para verificar a encriptação é mais complicado, pois não há um exemplo claro de que o texto foi corretamente encriptado, então a "garantia" será da comparação do texto original com o texto encriptado 2 vezes.
		
	(BASE-64 Sequencial) Para executar o programa, basta compilar usando o Makefile (make) e executar da seguinte forma:

		./base64 [arquivo de texto]

		Em que [arquivo de texto] é o nome do arquivo contendo o texto a ser encriptado.

		O programa executará o algoritmo 2 vezes, pois BASE-64(BASE-64(texto)) = texto. O código é idêntico ao disponibilizado no github (https://github.com/phrb/MAC5742-0219-EP2/blob/master/src/crypto-algorithms/), apenas tendo o nome de um arquivo como entrada.

	(ROT-13 CUDA) Para executar o programa, basta compilar usando o Makefile (make) e executar da seguinte forma:

		./cuda_base64 [arquivo de texto] [out] [flag de nova linha]

		Em que [arquivo de texto] é o nome do arquivo contendo o texto a ser encriptado, [out] é o nome do arquivo de sada e [flag de nova linha] é um inteiro que representa na tabela ASCII o símbolo de quebra de linha, normalmente é o número 10.

		O programa executará o algoritmo 2 vezes, pois BASE-64(BASE-64(texto)) = texto.

		Caso deseje verificar se o algoritmo converte corretamente a entrada, há algumas linhas de código comentadas, que farão o mesmo teste presente no código disponibilizado no github do monitor (https://github.com/phrb/MAC5742-0219-EP2/blob/master/src/crypto-algorithms/base64_test.c). Basta passar como entrada o arquivo "f.txt" e retirar os comentários do código na função rot13_test().


Gráficos:

Para gerar os gráficos com os resultados obtidos (30 execuções usando o comando perf stat), execute os arquivos .py da forma:

	python [arquivo.py]

Em que arquivo.py é o arquivo disponível na pasta Graphs.


