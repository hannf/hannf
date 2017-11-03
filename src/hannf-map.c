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

#include "hannf-map.h"

#undef  __FUNCT__
#define __FUNCT__ "HANNFMapFinal"
PetscErrorCode
HANNFMapFinal(HANNF* hannf)
{
    PetscFunctionBegin;
    // work vars
    PetscInt nl = hannf->nl;
    // destroy net matrices and vectors
    // forward
    MatDestroyMatrices(nl-1, &hannf->W);
    VecDestroyVecs(nl-1, &hannf->b);
    VecDestroyVecs(nl-1, &hannf->s);
    VecDestroyVecs(nl, &hannf->h);
    VecDestroyVecs(nl, &hannf->w);
    // scatter
    for (int i = 0; i < (nl-1); i++) {
        VecScatterDestroy(&hannf->h_scatter[i]);
    }
    PetscFree(hannf->h_scatter);
    VecDestroyVecs(nl-1, &hannf->h_all);
    // derivatives
    VecDestroyVecs(nl-1, &hannf->dW);
    VecDestroyVecs(nl-1, &hannf->db);
    VecDestroyVecs(nl, &hannf->dh);
    // destroy storage/work vector
    VecDestroy(&hannf->umem);
    // debug
    HANNFDebug(hannf, "HANNFMapFinal\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFMapInit"
PetscErrorCode
HANNFMapInit(HANNF* hannf)
{
    /*
     *  allocate memory for objects used during mapping ...
     *  ... and for the computation of derivatives
     */
    PetscFunctionBegin;
    // work vars
    PetscInt    nl = hannf->nl;
    PetscInt    i;
    PetscScalar *memarray;
    PetscInt    memidx = 0;
    // weight matrices
    // bias vectors
    PetscMalloc((nl-1)*sizeof(Mat), &hannf->W);
    PetscMalloc((nl-1)*sizeof(Vec), &hannf->b);
    // state vectors
    // s, input from network
    // h, after activation, special case h_0 = x
    // work vectors
    PetscMalloc((nl-1)*sizeof(Vec), &hannf->s);
    PetscMalloc(nl*sizeof(Vec), &hannf->h);
    PetscMalloc(nl*sizeof(Vec), &hannf->w);
    // scatter of intermediate results
    // needed for the computation of derivatives
    // contexts
    // vectors, same copy for each process
    PetscMalloc((nl-1)*sizeof(VecScatter), &hannf->h_scatter);
    PetscMalloc((nl-1)*sizeof(Vec), &hannf->h_all);
    // derivatives
    // weight matrices, columnwise only
    // bias vectors
    // activation
    PetscMalloc((nl-1)*sizeof(Vec), &hannf->dW);
    PetscMalloc((nl-1)*sizeof(Vec), &hannf->db);
    PetscMalloc(nl*sizeof(Vec), &hannf->dh);
    // create storage vector
    // use computed sizes (HANNFLoadInit)
    // set to random values
    VecCreate(hannf->comm, &hannf->umem);
    VecSetType(hannf->umem, VECSTANDARD);
    VecSetSizes(hannf->umem, hannf->nmem_local, hannf->nmem_global);
    VecSetRandom(hannf->umem, PETSC_NULL);
    VecAssemblyBegin(hannf->umem);
    VecAssemblyEnd(hannf->umem);
    // get array from mem vector
    // use for matrices Wi and vectors bi
    VecGetArray(hannf->umem, &memarray);
    // loop over layers (minus one)
    for(i = 0; i < (nl-1); i++)
    {
        // create matrix
        // zero entries
        // assemble
        MatCreateDense(hannf->comm, PETSC_DECIDE, PETSC_DECIDE, hannf->nrow_global[i], hannf->ncol[i] - 1, &memarray[memidx], &hannf->W[i]);
        MatAssemblyBegin(hannf->W[i], MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(hannf->W[i], MAT_FINAL_ASSEMBLY);
        // create vectors (*left* of matrix)
        // forward
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->b[i]);
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->s[i]);
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->h[i+1]);
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->w[i+1]);
        // derivative
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->dW[i]);
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->db[i]);
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->dh[i+1]);
        // treat special case, vectors *right* of matrix
        if (i == 0) {
            MatCreateVecs(hannf->W[i], &hannf->h[i], PETSC_NULL);
            MatCreateVecs(hannf->W[i], &hannf->w[i], PETSC_NULL);
            MatCreateVecs(hannf->W[i], &hannf->dh[i], PETSC_NULL);
        }
        // create scatter context and h_all from h
        VecScatterCreateToAll(hannf->h[i], &hannf->h_scatter[i], &hannf->h_all[i]);
        // add offset for matrix
        // place array
        memidx = memidx + hannf->nrow_local[i] * (hannf->ncol[i] - 1);
        VecPlaceArray(hannf->b[i], &memarray[memidx]);
        VecAssemblyBegin(hannf->b[i]);
        VecAssemblyEnd(hannf->b[i]);
        // add offset for vector
        memidx = memidx + hannf->nrow_local[i];
    }
    // restore mem vector
    VecRestoreArray(hannf->umem, &memarray);
    // debug
    HANNFDebug(hannf, "HANNFMapInit\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFMapNeuronActivate"
PetscErrorCode
HANNFMapNeuronActivate(HANNF *hannf, Vec dh, Vec h, Vec s)
{
    PetscFunctionBegin;
    // work vars
    PetscScalar *dharray, *harray, *sarray;
    PetscInt nvec, j;
    // get arrays
    VecGetArray(dh, &dharray);
    VecGetArray(h, &harray);
    VecGetArray(s, &sarray);
    // get size
    VecGetLocalSize(h, &nvec);
    // loop over vector entries (dh, h, s)
    // pre-compute derivative (dh)
    for(j = 0; j < nvec; j++)
    {
        // sigma(x) = 1/(1+e^-x)
        // sigma'(x) = sigma(x)*(1 - sigma(x))
        harray[j] = 1.0 / (1.0 + exp(-sarray[j]));
        dharray[j] = harray[j] * (1.0 - harray[j]);
    }
    // restore arrays
    VecRestoreArray(dh, &dharray);
    VecRestoreArray(h, &harray);
    VecRestoreArray(s, &sarray);
    // debug
    HANNFDebug(hannf, "HANNFMapNeuronActivate\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFMapNeuronReceive"
PetscErrorCode
HANNFMapNeuronReceive(HANNF* hannf, Vec s, Mat W, Vec b, Vec x)
{
    PetscFunctionBegin;
    // weight and sum the network inputs (W, weights)
    // translate (b, bias)
    // s = W * x + b
    MatMultAdd(W, x, b, s);
    // debug
    HANNFDebug(hannf, "HANNFMapNeuronReceive\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFMap"
PetscErrorCode
HANNFMap(HANNF* hannf, Vec y, Vec x)
{
    PetscFunctionBegin;
    // work vars
    Mat *W = hannf->W;
    Vec *s = hannf->s;
    Vec *h = hannf->h;
    Vec *b = hannf->b;
    Vec *dh = hannf->dh;
    Vec *h_all = hannf->h_all;
    VecScatter *h_scatter = hannf->h_scatter;
    PetscInt nl = hannf->nl;
    PetscInt i;
    // copy x into h[0]
    VecCopy(x, h[0]);
    // loop over layers (minus one)
    for(i = 0; i < (nl-1); i++)
    {
        // scatter to all
        VecScatterBegin(h_scatter[i], h[i], h_all[i], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(h_scatter[i], h[i], h_all[i], INSERT_VALUES, SCATTER_FORWARD);
        // s[i] = W[i] * h[i] + b[i]
        // dh[i+1], h[i+1] = sigma(s[i])
        HANNFMapNeuronReceive(hannf, s[i], W[i], b[i], h[i]);
        HANNFMapNeuronActivate(hannf, dh[i+1], h[i+1], s[i]);
    }
    // copy h[nl-1] to y
    VecCopy(h[nl-1], y);
    // debug
    HANNFDebug(hannf, "HANNFMap\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFMapGradient"
PetscErrorCode
HANNFMapGradient(HANNF* hannf, Vec y, Vec x, Vec g)
{
    PetscFunctionBegin;
    // work vars
    PetscScalar *garray;
    PetscInt gidx;
    Vec *dh = hannf->dh;
    Vec *db = hannf->db;
    Vec *dW = hannf->dW;
    Mat *W = hannf->W;
    Vec *w = hannf->w;
    Vec *h_all = hannf->h_all;
    PetscScalar *h_all_array;
    PetscInt *nrow_local = hannf->nrow_local;
    PetscInt *ncol = hannf->ncol;
    PetscInt nl = hannf->nl;
    PetscInt i, j;
    // get the array from the gradient vector
    // set the index counter to the local length, i.e. one off the last entry
    VecGetArray(g, &garray);
    gidx = hannf->nmem_local;
    // copy y to w[nl-1]
    VecCopy(y, w[nl-1]);
    // go backward through the chain of network layers
    for(i = nl-2; i >= 0; i--)
    {
        // compute offset for the last vector b
        // place g array into db_i vector
        gidx = gidx - nrow_local[i];
        VecPlaceArray(db[i], &garray[gidx]);
        // db_i = db_{i+1} * W_{i+1} .* sigma'(s_{i+1})
        //      = db_{i+1} * W_{i+1} .* dh_{i+1}
        //      = w_{i+1} .* dh_{i+1}
        // compute the gradient w.r.t. the last vector b
        VecPointwiseMult(db[i], w[i+1], dh[i+1]);
        // dW_i = (db_i * h_{i}^T), dyadic product
        // we treat the derivative w.r.t. the last matrix W, columnwise
        // get h_all array
        // !!! index is i-1 !!!
        VecGetArray(h_all[i], &h_all_array);
        // loop over vector entries, backwards!
        // ncol stores no. of columns of W and b combined,
        // thus the start is ncol-2
        for(j = ncol[i]-2; j >= 0; j--)
        {
            // compute offset for the matrix W
            // place g array into work vector dW_i
            gidx = gidx - nrow_local[i];
            VecPlaceArray(dW[i], &garray[gidx]);
            // W_ij = W_ij + h_ij*db_i
            VecCopy(db[i], dW[i]);
            VecScale(dW[i], h_all_array[j]);
            // reset array
            VecResetArray(dW[i]);
        }
        // restore h_all array
        VecRestoreArray(h_all[i], &h_all_array);
        // prepare the next step
        // w[i]^T = bd[i]^T * W[i]
        MatMultTranspose(W[i], db[i], w[i]);
        // reset db_i vector array
        VecResetArray(db[i]);
    }
    // restore gradient array
    VecRestoreArray(g, &garray);
    // debug
    HANNFDebug(hannf, "HANNFMapGradient\n");
    PetscFunctionReturn(0);
}






//    // we need an additional vector for the computation of the derivative w.r.t. W[0]
//    // create input vector x
//    // create scatter context for input vector x
//    MatCreateVecs(hannf->W[0], &hannf->x, PETSC_NULL);
//    VecScatterCreateToAll(hannf->x, &hannf->x_scatter, &hannf->x_all);

//    // x
//    VecDestroy(&hannf->x);
//    VecDestroy(&hannf->x_all);
//    VecScatterDestroy(&hannf->x_scatter);

//    // scatter input vector x to all
//    VecScatterBegin(hannf->x_scatter, x, hannf->x_all, INSERT_VALUES, SCATTER_FORWARD);
//    VecScatterEnd(hannf->x_scatter, x, hannf->x_all, INSERT_VALUES, SCATTER_FORWARD);
//    // first layer
//    // s[0] = W[0] * x + b[0]
//    // dh[0], h[0] = sigma(s[0])
//    nh = ;
//    i = 0;
//    HANNFMapNeuronReceive(hannf, s[i], W[i], b[i], x);
//    HANNFMapNeuronActivate(hannf, dh[i], h[i], s[i]);

//        // scatter to all
//        VecScatterBegin(h_scatter[i], h[i], h_all[i], INSERT_VALUES, SCATTER_FORWARD);
//        VecScatterEnd(h_scatter[i], h[i], h_all[i], INSERT_VALUES, SCATTER_FORWARD);

//     // last layer
//    i = nl - 1;
//    // s[i] = W[i] * h[i-1] + b[i]
//    // dh[i], y = sigma(s[i])
//    HANNFMapNeuronReceive(hannf, s[i], W[i], b[i], h[i]);
//    HANNFMapNeuronActivate(hannf, dh[i], y, s[i]);
//    // scatter to all
//    VecScatterBegin(h_scatter[i], y, h_all[i], INSERT_VALUES, SCATTER_FORWARD);
//    VecScatterEnd(h_scatter[i], y, h_all[i], INSERT_VALUES, SCATTER_FORWARD);


//    PetscScalar *x_all_array;






//    // last layer
//    // db_nh = y .* sigma'(s_nh)
//    //       = y .* dh_nh
//    // set index to last layer
//    // compute offset for the last vector b
//    // place g array into db_i vector
//    i = hannf->nh;
//    gidx = gidx - nrow_local[i];
//    VecPlaceArray(db[i], &garray[gidx]);
//    // compute the gradient w.r.t. the last vector b
//    VecPointwiseMult(db[i], y, dh[i]);
//
//    // dW_nh = (db_nh * h_{nh-1}^T), dyadic product
//    // we treat the derivative w.r.t. the last matrix W, columnwise
//    // get h_all array
//    // !!! index is i-1 !!!
//    VecGetArray(h_all[i-1], &h_all_array);
//    // loop over vector entries, backwards!
//    for(j = ncol[i]-2; j >= 0; j--)
//    {
//        // compute offset for the matrix W
//        // place g array into work vector dW_i
//        gidx = gidx - nrow_local[i];
//        VecPlaceArray(dW[i], &garray[gidx]);
//        // W_ij = h_{i-1}j*db_i
//        VecCopy(db[i], dW[i]);
//        VecScale(dW[i], h_all_array[j]);
//        // reset array
//        VecResetArray(dW[i]);
//    }
//    // restore h_all array
//    VecRestoreArray(h_all[i-1], &h_all_array);
//    // prepare the next step
//    // w[i-1]^T = db[i]^T * W[i]
//    MatMultTranspose(W[i], db[i], w[i-1]);
//    // reset db_i vector array
//    VecResetArray(db[i]);
//    // layers inbetween



//    // first layer
//    // db_0 = db_1 * W_1 .* sigma'(s_0)
//    //      = db_{i+1} * W_{i+1} .* dh_i
//    //      = w_0 .* dh_0
//    // compute offset for the last vector b
//    // place g array into db_i vector
//    i = 0;
//    gidx = gidx - nrow_local[i];
//    VecPlaceArray(db[i], &garray[gidx]);
//    // compute the gradient w.r.t. the last vector b
//    VecPointwiseMult(db[i], w[i], dh[i]);
//    // dW_0 = db_0 * x^T
//    // get x_all array
//    VecGetArray(hannf->x_all, &x_all_array);
//    // loop over vector entries, backwards!
//    for(j = ncol[i]-2; j >= 0; j--)
//    {
//        // compute offset for the matrix W
//        // place g array into work vector dW_i
//        gidx = gidx - nrow_local[i];
//        VecPlaceArray(dW[i], &garray[gidx]);
//        // W_ij = x_j * db_i
//        VecCopy(db[i], dW[i]);
//        VecScale(dW[i], x_all_array[j]);
//        // reset array
//        VecResetArray(dW[i]);
//    }
//    // restore h_all array
//    VecRestoreArray(hannf->x_all, &x_all_array);
//    // reset db_i vector array
//    VecResetArray(db[i]);
