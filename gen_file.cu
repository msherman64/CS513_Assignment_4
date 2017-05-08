#include <stdio.h>

//constant for architecture
int SEED = 12; //seed for rand //old was 15
int DIM_LIM = 32; //max size of a matrix
double INIT_VAL = 0.1; //initial value of matrix
int MAT_COUNT = 10000; //


/* file format


ndicate the size of array A)
n_1 n_2 n_3... n_k (k numbers in a single line indicate the dimensions of matrices)
n_1 by n_2 matrix 1 (n_1 rows, each row contains n_2 doubles)
n_2 by n_3 matrix 2 (n_2 rows, each row contains n_3 doubles)
...
n_{k-1} by n_k matrix k-1

*/



double randMToN(double M, double N)
{
    return M + (rand() / ( RAND_MAX / (N-M) ) ) ;  
}

void gen_matrix(int rows, int cols, FILE *fp){
    for(int i = 0; i < rows; i++){//each row as outer loop
        for(int j = 0; j < cols; j++){ //each element in row, across all columns
            fprintf(fp, "%f ", INIT_VAL); //print space after each value
        }
        //print newline after each row.
        fprintf(fp, "\n");
    }
}







int main(int argc, char *argv[]){

    if(argc == 3){
        INIT_VAL = atof(argv[1]);
        MAT_COUNT = atoi(argv[2]);
        printf("main: %d matrices of initial value is %f\n", MAT_COUNT, INIT_VAL);
    } else {
        printf("incorrect input values, must be max initial value, and number of matrices");
        return -1;
    }

	
    //initialize random number gen, get array sizes
    srand(SEED); //init random gen
    int dim[MAT_COUNT + 1]; //stores matrix sizes
    for(int z = 0; z <= MAT_COUNT; z++){
        dim[z] = rand()%DIM_LIM + 1;//random between 1 and limit
    }

    const char* filename = "input.txt";
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d\n", MAT_COUNT + 1); //count of arrays
    for(int i = 0; i < MAT_COUNT + 1; i++){
        fprintf(fp, "%d ", dim[i]);
    }
    //print newline after each row.
    fprintf(fp, "\n");

    for(int i = 0; i < MAT_COUNT; i++){
        gen_matrix(dim[i], dim[i+1], fp);
    }


}
