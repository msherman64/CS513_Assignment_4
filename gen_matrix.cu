#include <stdlib.h> //for rand
#include <iostream> // for cout
#include <vector>  //for vector
using std::cout;

int DIM_LIM = 100;
int MAT_COUNT = 10;

int SEED = 10; //seed for rand

//generate random sized matrices, add pointer to each, to vector
//template <typename T> //handle multiple types
class matrix {
public:
    int row; //number of rows, y
    int col; //number of columns, x
//    std::vector<int> data;
    double* data;

    matrix(int columns, int rows) :
        col(columns), row(rows),
        data(new double[col * row])
        {}

    double& getdata(int x, int y){
       return data[y * col + x]; //vertical position * row length + pos in row
    };
};

matrix** initialize(){
    srand(SEED); //init random gen
    int dim[MAT_COUNT + 1]; //stores matrix sizes
    for(int z = 0; z <= MAT_COUNT; z++){
        dim[z] = rand()%DIM_LIM + 1;//random between 1 and limit
    }

    //declare matrix array as pointer
    matrix **mat = (matrix **)malloc(MAT_COUNT * sizeof(matrix*));
    for(int z = 0; z < MAT_COUNT; z++){
        //each matrix shares a dimension with the previous
        int dimx = dim[z];
        int dimy = dim[z+1];

        mat[z] = new matrix(dimx,dimy); //dimx columns, dimy rows
        for(int x = 0; x<dimx; x++){
            for(int y = 0; y<dimy; y++){
                //TODO change to random double
                mat[z]->getdata(x,y) = 5; //initialize each element
            }
        }
    }
    return mat;
}


int main(){

    matrix **mat = initialize(); //get starting matrix

    //debug by printing size and elements of each matrix in mat
    for(int z = 0; z < MAT_COUNT; z++){
        int dimxn = mat[z]->col;
        int dimyn = mat[z]->row;
        std::cout << dimxn <<" ";
        std::cout << dimyn <<" ";
        std::cout << "\n";
        for(int y = 0; y<dimyn; y++){
            for(int x = 0; x<dimxn; x++){
                cout << mat[z]->getdata(x,y) << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }



};
