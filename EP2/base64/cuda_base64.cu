/*********************************************************************
* Filename:   base64.cu
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the Base64 encoding algorithm.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <cuda_runtime.h>
extern "C" {
    #include "base64.h"
}
#include <memory.h>
#include <stdio.h>

/****************************** MACROS ******************************/
#define NEWLINE_INVL 76

/**************************** VARIABLES *****************************/
// Note: To change the charset to a URL encoding, replace the '+' and '/' with '*' and '-'
static const BYTE charset[]={"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"};

/*********************** FUNCTION DEFINITIONS ***********************/
BYTE revchar(char ch)
{
    if (ch >= 'A' && ch <= 'Z')
        ch -= 'A';
    else if (ch >= 'a' && ch <='z')
        ch = ch - 'a' + 26;
    else if (ch >= '0' && ch <='9')
        ch = ch - '0' + 52;
    else if (ch == '+')
        ch = 62;
    else if (ch == '/')
        ch = 63;
    return(ch);
}

__global__ 
void cuda_encode(const BYTE *in, BYTE *out, size_t len, int new_line_flag, BYTE *charset){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i_in, i_out, m;

    if (i < len){
        i_in = i * 3;
        m = NEWLINE_INVL + 1;
        i_out = i * 4 + i * 4 / m;;
        
        if(new_line_flag){
            if(i_out % m == m - 1) out[i_out++] = '\n';
            out[i_out++] = charset[in[i_in] >> 2];

            if(i_out % m == m - 1) out[i_out++] = '\n';
            out[i_out++] = charset[((in[i_in] & 0x03) << 4) | (in[i_in + 1] >> 4)];

            if(i_out % m == m - 1) out[i_out++] = '\n';
            out[i_out++] = charset[((in[i_in + 1] & 0x0f) << 2) | (in[i_in + 2] >> 6)];
            
            if(i_out % m == m - 1) out[i_out++] = '\n';
            out[i_out++] = charset[in[i_in + 2] & 0x3F];
        }
        else{
            out[i_out++] = charset[in[i_in] >> 2];
            out[i_out++] = charset[((in[i_in] & 0x03) << 4) | (in[i_in + 1] >> 4)];
            out[i_out++] = charset[((in[i_in + 1] & 0x0f) << 2) | (in[i_in + 2] >> 6)];
            out[i_out++] = charset[in[i_in + 2] & 0x3F];
        }
    }
}

__device__
int next(const BYTE *in, int idx){
    if (in[idx + 1] == '\n') idx++;
    return (idx + 1);
}


__global__
void cuda_decode(const BYTE *in, BYTE *out, size_t len, BYTE *revchar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i_in, i_out, m, a, b, count;
    if (i < len) {
        m = NEWLINE_INVL + 1;
        i_in = next(in, i * 4 + i * 4 / m - 1);
        i_out = i * 3;

        for(count = 1; count <= 3; count++){
            a = revchar[in[i_in]] << (2 * count); 
            i_in = next(in, i_in);

            if(count == 1) b = (revchar[in[i_in]] & 0x30) >> 4;
            else if(count == 2) b = revchar[in[i_in]] >> 2;
            else b = revchar[in[i_in]];
            
            if(count < 3) out[i_out++] = a | b;
            else out[i_out] = a | b;
        }
    }
}

void checkCudaErr() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro na chamada cuda: ");
        printf("%s\n", cudaGetErrorString(err));
    }
}

extern "C"
size_t base64_encode(const BYTE in[], BYTE out[], size_t len, int newline_flag) {
    size_t blks, left_over, blk_ceiling, len_out;
    int block_size = 256;
    BYTE  *cuda_charset, *cuda_in, *cuda_out;

    blks = (len / 3);
    left_over = len % 3;
    len_out = blks * 4;
    if (out == NULL) return len_out;
    if (left_over) len_out = len_out + 4;
    if (newline_flag) len_out = len_out + len / 57;
    
    
    cudaMalloc(&cuda_in, sizeof(BYTE) * len);
    checkCudaErr();
    cudaMalloc(&cuda_out, sizeof(BYTE) * len_out);
    checkCudaErr();
    cudaMalloc(&cuda_charset, sizeof(BYTE) * block_size / 4);
    checkCudaErr();
    cudaMemcpy(cuda_in, in, len * sizeof(BYTE), cudaMemcpyHostToDevice);
    checkCudaErr();
    cudaMemcpy(cuda_charset, charset, (block_size / 4) * sizeof(BYTE), cudaMemcpyHostToDevice);
    checkCudaErr();
    cuda_encode<<<blks / block_size + 1, block_size>>>(cuda_in, cuda_out, blks, newline_flag, cuda_charset);                                                      
    checkCudaErr();
    cudaMemcpy(out, cuda_out, len_out * sizeof(BYTE), cudaMemcpyDeviceToHost);
    checkCudaErr();
    
    len_out = blks * 4;
    blk_ceiling = blks * 3;
     
    if (newline_flag) len_out += blks * 4 / (NEWLINE_INVL + 1);
    if (left_over == 1 | left_over == 2) {
        out[len_out++] = charset[in[blk_ceiling] >> 2];
        if(left_over == 1){
            out[len_out++] = charset[(in[blk_ceiling] & 0x03) << 4];
            out[len_out++] = '=';
        }
        else{
            out[len_out++] = charset[((in[blk_ceiling] & 0x03) << 4) | (in[blk_ceiling + 1] >> 4)];
            out[len_out++] = charset[(in[blk_ceiling + 1] & 0x0F) << 2];
        }
        out[len_out++] = '=';
        out[len_out] = '\0';
    }

    cudaFree(cuda_in);
    cudaFree(cuda_out);
    cudaFree(cuda_charset);
    return(len_out);
}


size_t base64_decode(const BYTE in[], BYTE out[], size_t len) {
    size_t len_out, blks, blk_ceiling, left_over, newline_len, newline_leftover;
    int block_size = 256, i, newline_flag;
    BYTE *revchar = (BYTE *) malloc(block_size * sizeof(BYTE));
    BYTE *cuda_revchar, *cuda_in, *cuda_out;

    for (i = 0; i < block_size; i++) revchar[i] = 'A';
    for (i = 0; i < block_size / 4; i++) revchar[charset[i]] = i;
    newline_len = len;
    newline_flag = newline_len >= (NEWLINE_INVL + 1) && in[NEWLINE_INVL] == '\n';
    len_out = (newline_len / 4) * 3;
    newline_leftover = newline_len % 4;

    if (in[len - 1] == '=' && in[len - 2] == '=') len = len - 2;
    else if (in[len - 1] == '=') len--;
    if (newline_flag) newline_len = newline_len - len / (NEWLINE_INVL + 1);
    if (newline_leftover > 1) len_out += newline_leftover - 1;
    if (out == NULL) return len_out;
 
    blks = len / 4;
    left_over = len % 4;

    cudaMalloc(&cuda_in, sizeof(BYTE) * len);
    checkCudaErr();
    cudaMalloc(&cuda_out, sizeof(BYTE) * len_out);
    checkCudaErr();
    cudaMalloc(&cuda_revchar, sizeof(BYTE) * block_size);
    checkCudaErr();
    cudaMemcpy(cuda_in, in, len * sizeof(BYTE), cudaMemcpyHostToDevice);
    checkCudaErr();
    cudaMemcpy(cuda_revchar, revchar, block_size * sizeof(BYTE), cudaMemcpyHostToDevice);
    checkCudaErr();
    cuda_decode<<<blks / block_size + 1, block_size>>>(cuda_in, cuda_out, blks, cuda_revchar);                                                    
    checkCudaErr();
    cudaMemcpy(out, cuda_out, len_out * sizeof(BYTE), cudaMemcpyDeviceToHost);
    checkCudaErr();
 
    blk_ceiling = blks * 4;
    if (newline_flag) len_out = 3 * (blk_ceiling - blk_ceiling / (NEWLINE_INVL + 1)) / 4;
    else len_out = 3 * blks;
    if (left_over == 2 || left_over == 3){
        out[len_out++] = (revchar[in[blk_ceiling]] << 2) | ((revchar[in[blk_ceiling + 1]] & 0x30) >> 4);
        if (left_over == 3) out[len_out++] = (revchar[in[blk_ceiling + 1]] << 4) | (revchar[in[blk_ceiling + 2]] >> 2);
        out[len_out] = '\0';
    }

    cudaFree(cuda_in);
    cudaFree(cuda_out);
    cudaFree(cuda_revchar);
    return len_out;
}

void base64_encrypt(char h_t[], const BYTE *in, BYTE *out, int new_line_flag, BYTE *charset){

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

    printf("Copy input data from the host memory to the CUDA device\n");
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
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda_encode, 0, numElements); 

    gridSize = (numElements + blockSize - 1) / blockSize;
    if (gridSize > 65535) gridSize = 65535;
    printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, blockSize);
    cuda_encode<<<gridSize, blockSize>>>(in, out, numElements, new_line_flag, charset);

    
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Erro ao rodar treyfer kernel (error code %s)!\n", cudaGetErrorString(err));
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
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  Resetando o device
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\nDone\n");
}

int base64_test(char * name, const BYTE *in, BYTE *out, int new_line_flag, BYTE *charset)
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

    base64_encrypt(text, in, out, new_line_flag, charset);
    //pass = pass && !strcmp(text, dec_exp);
    //printf("%s\n", pass ? "DEC SUCCEEDED" : "DEC FAILED");

    base64_encrypt(text, in, out, new_line_flag, charset);

    pass = pass && !strcmp(text, o_text);

    return(pass);
}

int main(int argc, char ** argv)
{

    printf("CUDA BASE64 tests: %s\n", base64_test(argv[1], (BYTE *)argv[1], (BYTE *)argv[2], *(int *)argv[3], (BYTE *)charset) ? "SUCCEEDED" : "FAILED");
    // nome arq entrada, arq saida, flag mudança

    return(0);
}

