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

#include "hannf-load.h"

#undef  __FUNCT__
#define __FUNCT__ "HANNFLoadFinal"
PetscErrorCode
HANNFLoadFinal(HANNF* hannf)
{
    PetscFunctionBegin;
    // free load variables
    PetscFree(hannf->nrow_global);
    PetscFree(hannf->nrow_local);
    PetscFree(hannf->ncol);
    // debug
    HANNFDebug(hannf, "HANNFLoadFinal\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFLoadInit"
PetscErrorCode
HANNFLoadInit(HANNF* hannf)
{
    PetscFunctionBegin;
    // matrix and vector dimensions
    // W1,  (hannf->nnl[1], hannf->nnl[0])  ; b1, h1, s1,  (hannf->nnl[0])
    // W2,  (hannf->nnl[2], hannf->nnl[1])  ; b2, h2, s2,  (hannf->nnl[1])
    // ...
    // Wnl-1, (hannf->nnl[nl], hannf->nl[n-1]); bnl, hnl, snl, (hannf->nnl[nl])
    //
    // work vars
    MPI_Comm comm = hannf->comm;
    PetscInt nl = hannf->nl;
    PetscInt nproc;
    PetscInt iproc;
    PetscInt i;
    PetscInt nmem_global = 0;
    PetscInt nmem_local = 0;
    PetscInt nrow_global, nrow_local, ncol;
    // determine process count and my process number
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &iproc);
    // store
    hannf->nproc = nproc;
    hannf->iproc = iproc;
    // compute local and global sizes for later use
    // allocate memory for storage
    PetscMalloc((nl-1)*sizeof(PetscInt), &hannf->nrow_global);
    PetscMalloc((nl-1)*sizeof(PetscInt), &hannf->nrow_local);
    PetscMalloc((nl-1)*sizeof(PetscInt), &hannf->ncol);
    // loop over layers
    for(i = 0; i < (nl-1); i++)
    {
        // set rows
        // set columns
        // matrix W plus vector b
        nrow_global = hannf->nnl[i+1];
        ncol = (hannf->nnl[i] + 1);
        // get petsc distribution
        nrow_local = PETSC_DECIDE;
        PetscSplitOwnership(comm, &nrow_local, &nrow_global);
        // sum up
        nmem_global = nmem_global + nrow_global * ncol;
        nmem_local = nmem_local + nrow_local * ncol;
        // store
        hannf->nrow_global[i] = nrow_global;
        hannf->nrow_local[i] = nrow_local;
        hannf->ncol[i] = ncol;
    }
    // store global
    hannf->nmem_local = nmem_local;
    hannf->nmem_global = nmem_global;
    // debug, global
    HANNFDebug(hannf, FSSD, "HANNFLoadInit", "nproc:", nproc);
    HANNFDebug(hannf, FSSD, "HANNFLoadInit", "nmem_global:", nmem_global);
    // debug, local
    if (hannf->debug > 0) {
        PetscSynchronizedPrintf(comm, "%17s %-35s %18s %-8d %18s %-8d\n", " ", "HANNFLoadInit", "iproc:", iproc, "nmem_local:", nmem_local);
        PetscSynchronizedFlush(comm, PETSC_STDOUT);
    }
    // debug
    HANNFDebug(hannf, "HANNFLoadInit\n");
    PetscFunctionReturn(0);
}


