/*
 * Copyright (C) 2017  Jaroslaw Piwonski, CAU, jpi@informatik.uni-kiel.de
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef HANNF_TYPE_H
#define HANNF_TYPE_H 1

#include "petsc.h"
#include "petsctao.h"

/*
 *  HANNF type
 */
typedef const char* HANNFType;
#define HANNF_TYPE_MLP "mlp"

/*
 *  HANNF context data type
*/
typedef struct {
    // communication
    MPI_Comm        comm;           // mpi context
    PetscInt        nproc;          // global process count
    PetscInt        myproc;         // local process count
    // debug/timing
    PetscInt        debug;          // debug switch
    PetscLogDouble  startTime;      // overall duration
    PetscLogDouble  timeStamp;      // timing delta
    // network
    HANNFType       type;           // network type
    PetscInt        nin;            // neuron count input layer
    PetscInt        nout;           // neuron count output layer
    PetscInt        nh;             // hidden layer count
    PetscInt        *nhi;           // neuron count per hidden layer
    // load
    PetscInt        *nrow_global;   // global row count per network layer
    PetscInt        *nrow_local;    // local row count per network layer
    PetscInt        *ncol;          // column count per network layer (matrix plus vector)
    PetscInt        nmem_global;    // global length of storage vector
    PetscInt        nmem_local;     // local length of storage vector
    // mapping
    Mat             *W;             // network matrices (n+1)
    Vec             *b;             // network vectors (n+1)
    Vec             *s;             // hidden layer, network input vector
    Vec             *h;             // hidden layer, activated vector
    Vec             *w;             // work vectors per layer
    VecScatter      *h_scatter;     // scatter context for h_all
    Vec             *h_all;         // hidden layer vector, scattered to all
    // derivatives
    Vec             *dW;            // derivatives with respect to a W matrix, columnwise
    Vec             *db;            // derivatives with respect to a b vector
    Vec             *dh;
    Vec             mem;            // storage vector
    // training
    PetscInt        nt;             // training data count (or sequence length)
    Vec             *X;             // input
    Vec             *Y;             // output
    Tao             tao;            // optimization context
    Vec             x;              // initial/result vector
    // work
    Mat             XX;             // input work matrix
    Mat             YY;             // output work matrix
} HANNF;

#endif /* HANNF_TYPE_H */

//// derivatives
//// weight matrices, columnwise only
//// bias vectors
//// activation
//PetscMalloc(nmax * sizeof(Vec), &hannf->dW);
//PetscMalloc(nmax * sizeof(Vec), &hannf->db);
//PetscMalloc(nmax * sizeof(Vec), &hannf->dh);
//// scatter of intermediate results
//// contexts
//// vectors, same copy for each process
//PetscMalloc(nmax * sizeof(VecScatter), &hannf->h_scatter);
//PetscMalloc(nmax * sizeof(Vec), &hannf->h_all);
//// create storage vector
//// use computed sizes (HANNFLoadInit)
//// set to random values
//VecCreate(hannf->comm, &hannf->mem);

//    Vec             y;              // output/result vector
