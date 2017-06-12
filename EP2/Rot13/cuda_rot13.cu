#include <stdio.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <string.h>

__global__ void
cuda_rot13(char * str, int numElements){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int case_type;
    if (i < numElements){
        if (str[i] < 'A' || (str[i] > 'Z' && str[i] < 'a') || str[i] > 'z')
            return;
            // Determine if the char is upper or lower case.
        if (str[i] >= 'a')
            case_type = 'a';
        else
            case_type = 'A';
          // Rotate the char's value, ensuring it doesn't accidentally "fall off" the end.
        str[i] = (str[i] + 13) % (case_type + 26);
        if (str[i] < 26)
            str[i] += case_type;
    }
}


void rot13_encrypt(char h_t[]){

    cudaError_t err = cudaSuccess;

    
    int numElements = strlen(h_t);
    size_t size = numElements * sizeof(char);

    // Alocacao do vetor do device

    char * d_t = NULL;
    err = cudaMalloc((void **)&d_t, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Falha ao alocar vetor do device (texto original) (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copia do vetor com o texto original do host para o do device

    printf("Copiando texto do host pro device\n");
    err = cudaMemcpy(d_t, h_t, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Falha ao copiar vetor c texto original do host para o device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  Rodando o algoritmo de encriptacao
    //  O tamanho do bloco eh determinado pela funcao cudaOccupancyMaxPotentialBlockSize
    //  das ferramentas do CUDA.
    //  O tamanho maximo do grid eh de 65535

    int minGridSize, blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda_rot13, 0, numElements); 

    gridSize = (numElements + blockSize - 1) / blockSize;
    if (gridSize > 65535) gridSize = 65535;
    printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, blockSize);
    cuda_rot13<<<gridSize, blockSize>>>(d_t, numElements);
    
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao rodar kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copia o resultado do device para o host

    printf("Copiando o texto do device para o host\n");
    err = cudaMemcpy(h_t, d_t, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao copiar do device pro host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Liberando o texto do device
    err = cudaFree(d_t);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao liberar o texto do device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  Resetando o device
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao resetar o device! (error=%s\n)!", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\nDone\n");
}

int rot13_test(char * name)
{
    char * text, *o_text;
    int pass = 1, fsize;
    //char dec_exp[] = {"NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"};

    FILE *f;
    char * filename = (char*) malloc (100 * sizeof(char));
    strcpy(filename, name);
    f = fopen(filename, "r");
    if (f){
        fseek(f, 0, SEEK_END);
        fsize = ftell(f);
        rewind(f);
        text = (char *) malloc (fsize * sizeof (char));
        o_text = (char *) malloc (fsize * sizeof (char));
        fread(o_text, 1, fsize, f);
        fclose(f);
    }
    else{
        fprintf(stderr, "Erro ao abrir arquivo!\n");
        exit(EXIT_FAILURE);
    }
    
    strcpy(text, o_text);

    rot13_encrypt(text);
    //pass = pass && !strcmp(text, dec_exp);
    //printf("%s\n", pass ? "DEC SUCCEEDED" : "DEC FAILED");

    rot13_encrypt(text);

    pass = pass && !strcmp(text, o_text);

    return(pass);
}

int main(int argc, char ** argv)
{

    printf("CUDA ROT-13 tests: %s\n", rot13_test(argv[1]) ? "SUCCEEDED" : "FAILED");

    return(0);
}

