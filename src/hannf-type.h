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
 *  HANNF network type
 */
typedef const char* HANNFNetType;
#define HANNF_NET_TYPE_FFNN "ffnn"

/*
 *  HANNF network topology constraints
 */
#define HANNF_NET_MAX_LAYER 65535

/*
 *  HANNF context data type
*/
typedef struct {
    // communication
    MPI_Comm            comm;           // mpi context
    PetscInt            nproc;          // global process count
    PetscInt            iproc;          // local process number
    // debug/timing
    PetscInt            debug;          // debug switch
    PetscLogDouble      startTime;      // overall duration
    PetscLogDouble      timeStamp;      // timing delta
    // network
    HANNFNetType        type;           // network type
    PetscInt            nl;             // network layer count
    PetscInt            *nnl;           // neuron count per network layer
    // load
    PetscInt            *nrow_global;   // global row count per network layer
    PetscInt            *nrow_local;    // local row count per network layer
    PetscInt            *ncol;          // column count per network layer (matrix plus vector)
    PetscInt            nmem_global;    // global length of storage vector
    PetscInt            nmem_local;     // local length of storage vector
    // mapping
    Mat                 *W;             // network matrices
    Vec                 *b;             // network vectors
    Vec                 *s;             // hidden layer, network input vector
    Vec                 *h;             // hidden layer, activated vector
    Vec                 *w;             // work vectors per layer
    // derivatives
    VecScatter          *h_scatter;     // scatter context for h_all
    Vec                 *h_all;         // hidden layer vector, scattered to all
    Vec                 *dW;            // derivatives with respect to a W matrix, columnwise
    Vec                 *db;            // derivatives with respect to a b vector
    Vec                 *dh;            // derivative of activation function
    Vec                 umem;           // storage vector
    // training
    Tao                 tao;            // optimization context
    PetscInt            nt;             // training data count (or sequence length)
    Vec                 *x;             // input
    Vec                 *y;             // output
} HANNF;

#endif /* HANNF_TYPE_H */





//    Vec                 u;              // optimization initial/result vector

//    Vec                 x;              // input vector x
//    VecScatter          x_scatter;      // scatter context for the input vector
//    Vec                 x_all;          // input vector scattered to all
// work
//    Mat                 XX;             // input work matrix
//    Mat                 YY;             // output work matrix
