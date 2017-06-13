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
    //
    // W1,  (hannf->nhi[0], hannf->nin)   ; b1, h, s,  (hannf->nhi[0])
    // W2,  (hannf->nhi[1], hannf->nhi[0]); b2, h, s,  (hannf->nhi[1])
    // ...
    // Wnh, (hannf->nout, hannf->nhi[n-1]); bnh, h, s, (hannf->nout)
    //
    // work vars
    MPI_Comm comm = hannf->comm;
    PetscInt nproc;
    PetscInt myproc;
    PetscInt nmax, i;
    PetscInt nmem_global = 0;
    PetscInt nmem_local = 0;
    PetscInt nrow_global, nrow_local, ncol;
    // determine process count and my process number
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &myproc);
    // store
    hannf->nproc = nproc;
    hannf->myproc = myproc;
    // compute local and global sizes for later use
    // allocate memory for storage
    nmax = hannf->nh + 1;
    PetscMalloc(nmax*sizeof(PetscInt), &hannf->nrow_global);
    PetscMalloc(nmax*sizeof(PetscInt), &hannf->nrow_local);
    PetscMalloc(nmax*sizeof(PetscInt), &hannf->ncol);
    // loop over layers
    for(i = 0; i < hannf->nh+1; i++)
    {
        // set sizes
        if (i == 0)
        {
            // first hidden layer
            nrow_global = hannf->nhi[i];
            ncol = (hannf->nin + 1);
        }
        else if (i == hannf->nh)
        {
            // output layer
            nrow_global = hannf->nout;
            ncol = (hannf->nhi[i-1] + 1);
        }
        else
        {
            // layers inbetween
            nrow_global = hannf->nhi[i];
            // matrix W plus vector b
            ncol = (hannf->nhi[i-1] + 1);
        }
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
        PetscSynchronizedPrintf(comm, "%17s %-40s %18s %-8d %18s %-8d\n", " ", "HANNFLoadInit", "myproc:", myproc, "nmem_local:", nmem_local);
        PetscSynchronizedFlush(comm, PETSC_STDOUT);
    }
    // debug
    HANNFDebug(hannf, "HANNFLoadInit\n");
    PetscFunctionReturn(0);
}


