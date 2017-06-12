/*********************************************************************
* Filename:   rot-13_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding ROT-13
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "rot-13.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int rot13_test(char * file)
{
    char * o_message, * message;
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
    message = (char *) malloc (fsize * sizeof (char));
    strcpy(message, o_message);

    rot13(message);

    rot13(message);

    pass = pass && !strcmp(message, o_message);

    return(pass);
}

int main(int argc, char ** argv)
{
    printf("ROT-13 tests: %s\n", rot13_test(argv[1]) ? "SUCCEEDED" : "FAILED");

    return(0);
}
