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

#include "hannf-train.h"

#undef  __FUNCT__
#define __FUNCT__ "HANNFTrainDataFinal"
PetscErrorCode
HANNFTrainDataFinal(HANNF* hannf)
{
    PetscFunctionBegin;
    // destroy training data vectors
    VecDestroyVecs(hannf->nt, &hannf->X);
    VecDestroyVecs(hannf->nt, &hannf->Y);
    // destroy training data matrices
    MatDestroy(&hannf->XX);
    MatDestroy(&hannf->YY);
    // debug
    HANNFDebug(hannf, "HANNFTrainDataFinal\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFTrainDataInit"
PetscErrorCode
HANNFTrainDataInit(HANNF* hannf)
{
    PetscFunctionBegin;
    // work vars
    MPI_Comm comm = hannf->comm;
    char filePath[PETSC_MAX_PATH_LEN];
    PetscViewer viewer;
    PetscScalar *mat_array;
    PetscInt i, nrow_local;
    
    //
    //  XX
    //
    // read in training data input
    HANNFUtilOptionsGetString(hannf, "-HANNFTrainingDataInput", filePath);
    // create work matrix XX
    // set type
    MatCreate(comm, &hannf->XX);
    MatSetType(hannf->XX, MATDENSE);
    // load matrix
    PetscViewerBinaryOpen(comm, filePath, FILE_MODE_READ, &viewer);
    MatLoad(hannf->XX, viewer);
    PetscViewerDestroy(&viewer);
    // determine sequence length
    MatGetSize(hannf->XX, PETSC_NULL, &hannf->nt);
    // debug
    HANNFDebug(hannf, FSSD, "HANNFTrainDataInit", "nt:", hannf->nt);
    // create X vectors from XX matrix
    // get local row count
    // use XX matrix array
    PetscMalloc(hannf->nt*sizeof(Vec), &hannf->X);
    MatGetLocalSize(hannf->XX, &nrow_local, PETSC_NULL);
    MatDenseGetArray(hannf->XX, &mat_array);
    for (i = 0; i < hannf->nt; i++) {
        // create a vector that will be multiplied against the first W
        MatCreateVecs(hannf->W[0], &hannf->X[i], PETSC_NULL);
        // place the matrix array in the vector
        VecPlaceArray(hannf->X[i], &mat_array[i*nrow_local]);
        // assemble vector
        VecAssemblyBegin(hannf->X[i]);
        VecAssemblyEnd(hannf->X[i]);
    }
    MatDenseRestoreArray(hannf->XX, &mat_array);

    //
    //  YY
    //
    // read in training data output
    HANNFUtilOptionsGetString(hannf, "-HANNFTrainingDataOutput", filePath);
    // create work matrix YY
    // set type
    MatCreate(comm, &hannf->YY);
    MatSetType(hannf->YY, MATDENSE);
    // load matrix
    PetscViewerBinaryOpen(comm, filePath, FILE_MODE_READ, &viewer);
    MatLoad(hannf->YY, viewer);
    PetscViewerDestroy(&viewer);
    // determine sequence length
    MatGetSize(hannf->YY, PETSC_NULL, &hannf->nt);
    // debug
    HANNFDebug(hannf, FSSD, "HANNFTrainDataInit", "nt:", hannf->nt);
    // create Y vectors from YY matrix
    // get local row count
    // use YY matrix array
    PetscMalloc(hannf->nt*sizeof(Vec), &hannf->Y);
    MatGetLocalSize(hannf->YY, &nrow_local, PETSC_NULL);
    MatDenseGetArray(hannf->YY, &mat_array);
    for (i = 0; i < hannf->nt; i++) {
        // create a vector that will be the result of multiplication with the last W
        MatCreateVecs(hannf->W[hannf->nh], PETSC_NULL, &hannf->Y[i]);
        // place the matrix array in the vector
        VecPlaceArray(hannf->Y[i], &mat_array[i*nrow_local]);
        // assemble vector
        VecAssemblyBegin(hannf->Y[i]);
        VecAssemblyEnd(hannf->Y[i]);
    }
    MatDenseRestoreArray(hannf->YY, &mat_array);
    // debug
    HANNFDebug(hannf, "HANNFTrainDataInit\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFTrainFinal"
PetscErrorCode
HANNFTrainFinal(HANNF* hannf)
{
    PetscFunctionBegin;
    // destroy optimization context
    // destroy optimization vector
    TaoDestroy(&hannf->tao);
//    VecDestroy(&hannf->u);
//    // derivative
//    VecDestroy(&hannf->ydiff);
    // final training data
    HANNFTrainDataFinal(hannf);
    // debug
    HANNFDebug(hannf, "HANNFTrainFinal\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFTrainInit"
PetscErrorCode
HANNFTrainInit(HANNF* hannf)
{
    PetscFunctionBegin;
    
    // init training data
    HANNFTrainDataInit(hannf);
    
//    // init variable for computation of derivative
//    VecDuplicate(hannf->Y[0], &hannf->ydiff);
//    VecAssemblyBegin(hannf->ydiff);
//    VecAssemblyEnd(hannf->ydiff);
    
    // create the optimization vector, use the work vector xx
//    VecDuplicate(hannf->mem, &hannf->u);
//    VecSetRandom(hannf->u, PETSC_NULL);
//    VecAssemblyBegin(hannf->u);
//    VecAssemblyEnd(hannf->u);

    // create optimization context
    // set initial vector, use the memory/storage vector
    // set objective and gradient
    TaoCreate(hannf->comm, &hannf->tao);
//    TaoSetInitialVector(hannf->tao, hannf->u);
    TaoSetInitialVector(hannf->tao, hannf->mem);
    TaoSetObjectiveAndGradientRoutine(hannf->tao, HANNFObjectiveAndGradient, (void*)hannf);
//    TaoSetObjectiveRoutine(hannf->tao, HANNFObjective, (void*)hannf);
    
    // set option prefix
    // set from options
    TaoSetOptionsPrefix(hannf->tao, "HANNF_");
    TaoSetFromOptions(hannf->tao);
    
    
    
    // debug
//    TaoSetMaximumIterations(hannf->tao, 1);
//    TaoSetMaximumFunctionEvaluations(hannf->tao, 1);

    
    
    // debug
    HANNFDebug(hannf, "HANNFTrainInit\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFTrain"
PetscErrorCode
HANNFTrain(HANNF* hannf)
{
    PetscFunctionBegin;
    // train
    TaoSolve(hannf->tao);
    // debug
    HANNFDebug(hannf, "HANNFTrain\n");
    PetscFunctionReturn(0);
}

//#undef  __FUNCT__
//#define __FUNCT__ "HANNFObjective"
//PetscErrorCode
//HANNFObjective(Tao tao, Vec u, PetscReal *f, void *ctx)
//{
//    // get context
//    HANNF* hannf = (HANNF*) ctx;
//    PetscFunctionBegin;
//    // work vars
//    PetscInt nt, nh, i;
//    PetscReal norm, sum;
//    // prepare loop
//    sum = 0.0;
//    nt = hannf->nt;
//    nh = hannf->nh;
//    // loop over training data
//    for(i = 0; i < nt; i++)
//    {
//        //
//        // objective
//        //
//        // feed forward
//        // mathematically from right to left
//        // map training data input x to y
//        HANNFMap(hannf, hannf->h[nh], hannf->X[i]);
//        // compute difference
//        // compute norm of difference
//        // sum up
//        VecWAXPY(hannf->w[nh], -1.0, hannf->Y[i], hannf->h[nh]);
//        VecNorm(hannf->w[nh], NORM_2, &norm);
//        sum = sum + norm * norm;
//    }
//    // weight the sum with one half
//    *f = 0.5 * sum;
//    // debug
////    VecView(u, PETSC_VIEWER_STDOUT_WORLD);
//    
////    MatView(hannf->W[0], PETSC_VIEWER_STDOUT_WORLD);
//    
////    HANNFDebug(hannf, "%24.16e\n", *f);
//    HANNFDebug(hannf, "HANNFObjective\n");
//    PetscFunctionReturn(0);
//}

#undef  __FUNCT__
#define __FUNCT__ "HANNFObjectiveAndGradient"
PetscErrorCode
HANNFObjectiveAndGradient(Tao tao, Vec u, PetscReal *f, Vec g, void *ctx)
{
    // get context
    HANNF* hannf = (HANNF*) ctx;
    PetscFunctionBegin;
    // work vars
    PetscInt nt, nh, i;
    PetscReal norm, sum;
    Vec g_i;
    // zero the entries of the gradient vector
    // create vector for the i_th component
    VecZeroEntries(g);
    VecDuplicate(g, &g_i);
    // prepare loop
    sum = 0.0;
    nt = hannf->nt;
    nh = hannf->nh;
    // loop over training data
    for(i = 0; i < nt; i++)
    {
        //
        // objective
        //
        // feed forward
        // mathematically from right to left
        // map training data input x to y
        HANNFMap(hannf, hannf->h[nh], hannf->X[i]);
        // compute difference
        // compute norm of difference
        // sum up
        VecWAXPY(hannf->w[nh], -1.0, hannf->Y[i], hannf->h[nh]);
        VecNorm(hannf->w[nh], NORM_2, &norm);
        sum = sum + norm * norm;
        //
        // gradient
        //
        // backwards
        // mathematically from left to right
        // compute gradient w.r.t. the i_th component of the cost function
        HANNFMapGradient(hannf, hannf->w[nh], hannf->X[i], g_i);
        // add to gradient vector
        // g = g + 1.0 * g_i
        VecAXPY(g, 1.0, g_i);
    }
    // weight the sum with one half
    *f = 0.5 * sum;
    // destroy work vector
    VecDestroy(&g_i);
    // debug
    HANNFDebug(hannf, "HANNFObjectiveAndGradient\n");
    PetscFunctionReturn(0);
}


