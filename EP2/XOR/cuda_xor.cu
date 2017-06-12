#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void
cuda_xor(char * encrypt, char * key, int numElements, size_t len_key){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        encrypt[i] = encrypt[i] ^ key[i % len_key];
    }
}

void xor_encrypt(char h_m[], char h_k[], size_t o_m_len) {

    cudaError_t err = cudaSuccess;

    
    int numElements_m = (int) o_m_len;
    int numElements_k = strlen(h_k);

    size_t size_m = numElements_m * sizeof(char);
    size_t size_k = numElements_k * sizeof(char);

    // Alocacao dos vetores do device

    char * d_m = NULL;
    err = cudaMalloc((void **)&d_m, size_m);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Falha ao alocar string do device (mensagem) (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    char * d_k = NULL;
    err = cudaMalloc((void **)&d_k, size_k);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Falha ao alocar string do device (chave) (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copia do vetor com o texto original do host para o do device

    printf("Copiando mensagem do host para o device\n");
    err = cudaMemcpy(d_m, h_m, size_m, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Falha ao copiar string mensagem do host para o device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copiando chave do host para o device\n");
    err = cudaMemcpy(d_k, h_k, size_k, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Falha ao copiar string chave do host para o device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  Rodando o algoritmo de encriptacao
    //  O tamanho do bloco eh determinado pela funcao cudaOccupancyMaxPotentialBlockSize
    //  das ferramentas do CUDA.
    //  O tamanho maximo do grid eh de 65535

    int minGridSize, blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda_xor, 0, numElements_m); 

    gridSize = (numElements_m + blockSize - 1) / blockSize;
    if (gridSize > 65535) gridSize = 65535;
    printf("CUDA kernel executando com %d blocos de %d threads\n", gridSize, blockSize);
    cuda_xor<<<gridSize, blockSize>>>(d_m, d_k, numElements_m, numElements_k);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao rodar XOR kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copia o resultado do device para o host

    printf("Copiando saida do device para o host\n");
    err = cudaMemcpy(h_m, d_m, size_m, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao copiar mensagem do device pro host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Liberando o conteudo do device

    err = cudaFree(d_m);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao liberar mensagem do device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_k);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao liberar chave do device (error code %s )!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  Resetando o device e terminando

    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao resetar o device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\nDone\n");

}

int xor_test(char * file, char * key_file)
{
    char * o_message, * key, * message;
    int pass = 1, fsize;

    FILE *f;

    f = fopen(file, "r");

    if (f){
        fseek(f, 0, SEEK_END);
        fsize = ftell(f);
        rewind(f);
        o_message = (char *) malloc (fsize * sizeof (char));
        fread(o_message, 1, fsize, f);
        fclose(f);
    }
    else{
        fprintf(stderr, "Erro ao abrir arquivo da mensagem!\n");
        exit(EXIT_FAILURE);
    }

    f = fopen(key_file, "r");
    if (f){
        fseek(f, 0, SEEK_END);
        fsize = ftell(f);
        rewind(f);
        key = (char *) malloc (fsize * sizeof (char));
        fread(key, 1, fsize, f);
        fclose(f);
    }
    else{
        fprintf(stderr, "Erro ao abrir arquivo da chave!\n");
        exit(EXIT_FAILURE);
    }

    message = (char *) malloc (strlen(o_message) * sizeof (char));
    strcpy(message, o_message);

    xor_encrypt(message, key, strlen(o_message));

    xor_encrypt(message, key, strlen(o_message));

    pass = pass && !strcmp(o_message, message);

    return(pass);
}

int main(int argc, char ** argv)
{

    printf("CUDA XOR tests: %s\n", xor_test(argv[1], argv[2]) ? "SUCCEEDED" : "FAILED");

    return(0);
}