#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char * argv[]) {
    char * text, * o_text, *key;

    int i, pass = 1, fsize;

    FILE *f;

    f = fopen(argv[1], "r");
    if (f){
        fseek(f, 0, SEEK_END);
        fsize = ftell(f);
        rewind(f);
        o_text = (char *) malloc (fsize * sizeof (char));
        fread(o_text, 1, fsize, f);
        fclose(f);
    }
    else{
        fprintf(stderr, "Erro ao abrir arquivo!\n");
        exit(EXIT_FAILURE);
    }

    f = fopen(argv[2], "r");

    if (f){
        fseek(f, 0, SEEK_END);
        fsize = ftell(f);
        rewind(f);
        key = (char *) malloc (fsize * sizeof (char));
        fread(key, 1, fsize, f);
        fclose(f);
    }
    else{
        fprintf(stderr, "Erro ao abrir arquivo!\n");
        exit(EXIT_FAILURE);
    }

    int size_text = strlen(o_text);
    int size_key = strlen(key);
    
    text = (char *) malloc (size_text * sizeof (char));
    strcpy(text, o_text);

    for(i = 0; i < size_text; i++) {
        text[i] = text[i] ^ key[i % size_key];
    }

    for (i = 0; i < size_text; i++){
        text[i] = text[i] ^ key[i % size_key];
    }

    pass = pass && !strcmp(o_text, text);

    printf("XOR tests: %s\n", pass ? "SUCCEEDED" : "FAILED");

    return 0;


}