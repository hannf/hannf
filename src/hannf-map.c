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
    // destroy net matrices and vectors
    // forward
    MatDestroyMatrices(hannf->nh+1, &hannf->W);
    VecDestroyVecs(hannf->nh+1, &hannf->b);
    VecDestroyVecs(hannf->nh+1, &hannf->s);
    VecDestroyVecs(hannf->nh+1, &hannf->h);
    VecDestroyVecs(hannf->nh+1, &hannf->w);
    // scatter
    for (int i = 0; i < hannf->nh+1; i++) {
        VecScatterDestroy(&hannf->h_scatter[i]);
    }
    PetscFree(hannf->h_scatter);
    VecDestroyVecs(hannf->nh+1, &hannf->h_all);
    // derivatives
    VecDestroyVecs(hannf->nh+1, &hannf->dW);
    VecDestroyVecs(hannf->nh+1, &hannf->db);
    VecDestroyVecs(hannf->nh+1, &hannf->dh);
    // destroy storage/work vector
    VecDestroy(&hannf->mem);
    // debug
    HANNFDebug(hannf, "HANNFMapFinal\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFMapInit"
PetscErrorCode
HANNFMapInit(HANNF* hannf)
{
    PetscFunctionBegin;
    // work vars
    PetscInt nmax, i;
    PetscScalar *memarray;
    PetscInt memidx = 0;
    // allocate memory for objects used during mapping ...
    // ... and for the computation of derivatives
    // one more than hidden layers
    nmax = hannf->nh + 1;
    // weight matrices
    // bias vectors
    PetscMalloc(nmax * sizeof(Mat), &hannf->W);
    PetscMalloc(nmax * sizeof(Vec), &hannf->b);
    // state vectors
    // s, input from network
    // h, after activation
    PetscMalloc(nmax * sizeof(Vec), &hannf->s);
    PetscMalloc(nmax * sizeof(Vec), &hannf->h);
    // work vectors
    PetscMalloc(nmax * sizeof(Vec), &hannf->w);
    // scatter of intermediate results
    // needed for the computation of derivatives
    // contexts
    // vectors, same copy for each process
    PetscMalloc(nmax * sizeof(VecScatter), &hannf->h_scatter);
    PetscMalloc(nmax * sizeof(Vec), &hannf->h_all);
    // derivatives
    // weight matrices, columnwise only
    // bias vectors
    // activation
    PetscMalloc(nmax * sizeof(Vec), &hannf->dW);
    PetscMalloc(nmax * sizeof(Vec), &hannf->db);
    PetscMalloc(nmax * sizeof(Vec), &hannf->dh);
    // create storage vector
    // use computed sizes (HANNFLoadInit)
    // set to random values
    VecCreate(hannf->comm, &hannf->mem);
    VecSetType(hannf->mem, VECSTANDARD);
    VecSetSizes(hannf->mem, hannf->nmem_local, hannf->nmem_global);
    VecSetRandom(hannf->mem, PETSC_NULL);
    VecAssemblyBegin(hannf->mem);
    VecAssemblyEnd(hannf->mem);
    // get array from mem vector
    // use for matrices Wi and vectors bi
    VecGetArray(hannf->mem, &memarray);
    // loop over layers
    for(i = 0; i < hannf->nh+1; i++)
    {
        // create matrix
        // zero entries
        // assemble
        // set to random values
        MatCreateDense(hannf->comm, PETSC_DECIDE, PETSC_DECIDE, hannf->nrow_global[i], hannf->ncol[i] - 1, &memarray[memidx], &hannf->W[i]);
        MatAssemblyBegin(hannf->W[i], MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(hannf->W[i], MAT_FINAL_ASSEMBLY);
        // create vectors
        // forward
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->b[i]);
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->s[i]);
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->h[i]);
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->w[i]);
        // create scatter context and h_all from h
        VecScatterCreateToAll(hannf->h[i], &hannf->h_scatter[i], &hannf->h_all[i]);
        // derivative
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->dW[i]);
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->db[i]);
        MatCreateVecs(hannf->W[i], PETSC_NULL, &hannf->dh[i]);
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
    VecRestoreArray(hannf->mem, &memarray);
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
    PetscInt i;
    // first layer
    // s[0] = W[0] * x + b[0]
    // h[0] = sigma(s[0])
    i = 0;
    HANNFMapNeuronReceive(hannf, hannf->s[i], hannf->W[i], hannf->b[i], x);
    HANNFMapNeuronActivate(hannf, hannf->dh[i], hannf->h[i], hannf->s[i]);
    // scatter to all
    VecScatterBegin(hannf->h_scatter[i], hannf->h[i], hannf->h_all[i], INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(hannf->h_scatter[i], hannf->h[i], hannf->h_all[i], INSERT_VALUES, SCATTER_FORWARD);
    // layers inbetween
    for(i = 1; i < hannf->nh; i++)
    {
        // s[i] = W[i] * h[i-1] + b[i]
        // h[i] = sigma(s[i])
        HANNFMapNeuronReceive(hannf, hannf->s[i], hannf->W[i], hannf->b[i], hannf->h[i-1]);
        HANNFMapNeuronActivate(hannf, hannf->dh[i], hannf->h[i], hannf->s[i]);
        // scatter to all
        VecScatterBegin(hannf->h_scatter[i], hannf->h[i], hannf->h_all[i], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(hannf->h_scatter[i], hannf->h[i], hannf->h_all[i], INSERT_VALUES, SCATTER_FORWARD);
    }
    // last layer
    // s[i] = W[i] * h[i-1] + b[i]
    // y = sigma(s[i])
    i = hannf->nh;
    HANNFMapNeuronReceive(hannf, hannf->s[i], hannf->W[i], hannf->b[i], hannf->h[i-1]);
    HANNFMapNeuronActivate(hannf, hannf->dh[i], y, hannf->s[i]);
    // scatter to all
    VecScatterBegin(hannf->h_scatter[i], y, hannf->h_all[i], INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(hannf->h_scatter[i], y, hannf->h_all[i], INSERT_VALUES, SCATTER_FORWARD);
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
    PetscInt i, j;
    PetscScalar *garray;
    PetscScalar *h_all_array;
    PetscInt garrayidx;
    
    // get the array from the gradient vector
    // set the index counter to the local length, i.e. one off the last entry
    garrayidx = hannf->nmem_local;
    VecGetArray(g, &garray);
    
    // place all arrays, b and W
    for (i = hannf->nh; i >= 0; i--) {
        garrayidx = garrayidx - hannf->nrow_local[i];
        VecPlaceArray(hannf->db[i], &garray[garrayidx]);
        
        // loop over columns, backwards!
        for(j = hannf->ncol[i] - 2; j >= 0; j--)
        {
        }
        
    }
    
    // go backward through the chain of network layers
    // last layer
    // db_nh = y .* sigma'(s_nh)
    //       = y .* (h_nh .* (1 - h_nh)) * W_nh       <- FIX ME, here we know we use the logistic function,
    //                                                           it was pre-computed during the computation of sigma
    //       = y .* dh_nh
    
    // set index to last layer
    // compute offset for the last vector b
    // place g array into db_i vector
    i = hannf->nh;
    garrayidx = garrayidx - hannf->nrow_local[i];
    VecPlaceArray(hannf->db[i], &garray[garrayidx]);
    
    // compute the gradient w.r.t. the last vector b
    // add to the gradient vector
    VecPointwiseMult(hannf->w[i], y, hannf->dh[i]);
    VecAXPY(hannf->db[i], 1.0, hannf->w[i]);
    
    // dW_nh = (db_nh * h_{nh-1}^T), dyadic product
    // we treat the derivative w.r.t. the last matrix W, columnwise
    
    // get h_all array
    // !!! index is i-1 !!!
    VecGetArray(hannf->h_all[i-1], &h_all_array);
    
    // loop over vector entries, backwards!
    for(j = hannf->ncol[i] - 2; j >= 0; j--)
    {
        // compute offset for the last matrix W
        // place g array into work vector dW_i
        garrayidx = garrayidx - hannf->nrow_local[i];
        VecPlaceArray(hannf->dW[i], &garray[garrayidx]);
        
        // W_ij = W_ij + h_{i-1}j*db_i
        VecAXPY(hannf->dW[i], h_all_array[j], hannf->db[i]);

        // reset array
        VecResetArray(hannf->dW[i]);
    }
    // restore h_all array
    VecRestoreArray(hannf->h_all[i-1], &h_all_array);
    // we don't need the gradient w.r.t. b anymore
    VecResetArray(hannf->db[i]);
    
    

    // layers inbetween
    for(i = hannf->nh - 1; i > 0; i--)
    {
        
//        garrayidx = garrayidx - hannf->nrow_local[i];
//        VecPlaceArray(hannf->db[i], &garray[garrayidx]);

        // db_i = db_{i+1} * W_{i+1} .* sigma'(s_i)
        //      = db_{i+1} * W_{i+1} .* dh_i
        //      = w_i .* dh_i
        MatMultTranspose(hannf->W[i+1], hannf->db[i+1], hannf->w[i]);
        VecPointwiseMult(hannf->db[i], hannf->w[i], hannf->dh[i]);
//        VecPointwiseMult(hannf->w[i], hannf->w[i], hannf->dh[i]);
//        VecAXPY(hannf->db[i], 1.0, hannf->w[i]);

        
        // dW_i = db_i * h_{i-1}^T
        //
        // ...
        //
        
    }

    // debug
    VecView(g, PETSC_VIEWER_STDOUT_WORLD);
    PetscEnd();
    
    
    
    // first layer
    // db_0 = db_1 * W_1 .* sigma'(s_0)
    //      = db_{i+1} * W_{i+1} .* dh_i
    //      = w_0 .* dh_0
    i = 0;
    MatMultTranspose(hannf->W[i+1], hannf->db[i+1], hannf->w[i]);
    VecPointwiseMult(hannf->db[i], hannf->w[i], hannf->dh[i]);

    // dW_0 = db_0 * x^T
    //
    // ...
    //
    
    
    
    // reset all arrays
    for (i = hannf->nh; i >= 0; i--) {
        VecResetArray(hannf->db[i]);
    }
    // restore g vector array
    VecRestoreArray(g, &garray);
    
    // debug
    HANNFDebug(hannf, "HANNFMapGradient\n");
    PetscFunctionReturn(0);
}




//    PetscPrintf(PETSC_COMM_SELF, "nmem_global: %d, nmem_local: %d\n", hannf->nmem_global, hannf->nmem_local);
//    PetscPrintf(PETSC_COMM_SELF, "i: %d, nmem_global: %d, nmem_local: %d, ncol: %d\n",
//                i, hannf->nrow_global[i], hannf->nrow_local[i], hannf->ncol[i]);
//  !!! AUFADDIEREN NICHT VERGESSEN !!!
//    PetscPrintf(PETSC_COMM_WORLD, "bd[i] ...\n");
//    VecView(hannf->db[i], PETSC_VIEWER_STDOUT_WORLD);

//    PetscPrintf(PETSC_COMM_SELF, "i: %d, nmem_global: %d, nmem_local: %d, ncol: %d\n",
//                i, hannf->nrow_global[i], hannf->nrow_local[i], hannf->ncol[i]);

//    PetscPrintf(PETSC_COMM_WORLD, "g ...\n");
//    PetscViewer pv;
//    PetscViewerBinaryOpen(hannf->comm, "g.petsc", FILE_MODE_WRITE, &pv);
//    VecView(g, PETSC_VIEWER_STDOUT_WORLD);
//    VecView(g, pv);
//    PetscViewerDestroy(&pv);

//        PetscPrintf(PETSC_COMM_SELF, "i: %d, nmem_global: %d, nmem_local: %d, ncol: %d\n",
//                    i, hannf->nrow_global[i], hannf->nrow_local[i], hannf->ncol[i]);

//    VecView(hannf->h[i-1], PETSC_VIEWER_STDOUT_WORLD);
//    VecView(hannf->h_all[i-1], PETSC_VIEWER_STDOUT_SELF);
