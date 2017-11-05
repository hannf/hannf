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

#define kDebugLevel kDebugLevel2

#undef  __FUNCT__
#define __FUNCT__ "HANNFTrainDataFinal"
PetscErrorCode
HANNFTrainDataFinal(HANNF* hannf)
{
    PetscFunctionBegin;
    // works vars
    PetscInt nt = hannf->nt;
    // destroy training data vectors
    VecDestroyVecs(nt, &hannf->x);
    VecDestroyVecs(nt, &hannf->y);
    PetscFree(hannf->x);
    PetscFree(hannf->y);
    // debug
    HANNFDebug(hannf, kDebugLevel, "HANNFTrainDataFinal\n");
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
    char in_file[PETSC_MAX_PATH_LEN];
    char out_file[PETSC_MAX_PATH_LEN];
    PetscViewer in_viewer, out_viewer;
    PetscInt nl = hannf->nl;
    PetscInt nt, i;
    // train data count
    HANNFUtilOptionsGetInt(hannf, "-HANNFTrainingDataCount", &nt);
    hannf->nt = nt;
    // read in training data input and output
    HANNFUtilOptionsGetString(hannf, "-HANNFTrainingDataIn", in_file);
    HANNFUtilOptionsGetString(hannf, "-HANNFTrainingDataOut", out_file);
    // open files
    PetscViewerBinaryOpen(comm, in_file, FILE_MODE_READ, &in_viewer);
    PetscViewerBinaryOpen(comm, out_file, FILE_MODE_READ, &out_viewer);
    // create storage for Vecs
    PetscMalloc(nt*sizeof(Vec), &hannf->x);
    PetscMalloc(nt*sizeof(Vec), &hannf->y);
    // read data
    for (i = 0; i < nt; i++) {
        // create a vector that will be multiplied by the first W
        // load in vec
        MatCreateVecs(hannf->W[0], &hannf->x[i], PETSC_NULL);
        VecLoad(hannf->x[i], in_viewer);
        // create a vector that will be the result of multiplication with the last W
        MatCreateVecs(hannf->W[nl-2], PETSC_NULL, &hannf->y[i]);
        VecLoad(hannf->y[i], out_viewer);
    }
    // close files
    PetscViewerDestroy(&in_viewer);
    PetscViewerDestroy(&out_viewer);
    // debug
    HANNFDebug(hannf, kDebugLevel, "HANNFTrainDataInit\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFTrainFinal"
PetscErrorCode
HANNFTrainFinal(HANNF* hannf)
{
    PetscFunctionBegin;
    // destroy optimization context
    TaoDestroy(&hannf->tao);
    // final training data
    HANNFTrainDataFinal(hannf);
    // debug
    HANNFDebug(hannf, kDebugLevel, "HANNFTrainFinal\n");
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
    // create optimization context
    // set initial vector, use the memory/storage vector
    // set objective and gradient
    TaoCreate(hannf->comm, &hannf->tao);
    TaoSetInitialVector(hannf->tao, hannf->umem);
    TaoSetObjectiveAndGradientRoutine(hannf->tao, HANNFObjectiveAndGradient, (void*)hannf);
    // set option prefix
    // set from options
    TaoSetOptionsPrefix(hannf->tao, "HANNFTraining_");
    TaoSetFromOptions(hannf->tao);
    // debug
    HANNFDebug(hannf, kDebugLevel, "HANNFTrainInit\n");
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
    HANNFDebug(hannf, kDebugLevel, "HANNFTrain\n");
    PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "HANNFObjectiveAndGradient"
PetscErrorCode
HANNFObjectiveAndGradient(Tao tao, Vec u, PetscReal *f, Vec g, void *ctx)
{
    // get context
    HANNF* hannf = (HANNF*) ctx;
    PetscFunctionBegin;
    // work vars
    PetscInt nt = hannf->nt;
    PetscInt nl = hannf->nl;
    PetscInt i;
    PetscReal norm, sum;
    Vec g_i;
    // copy parameter vector to memory work vector
    VecCopy(u, hannf->umem);
    // zero the entries of the gradient vector
    // create vector for the i_th component
    VecZeroEntries(g);
    VecDuplicate(g, &g_i);
    // prepare loop
    sum = 0.0;
    // loop over training data
    for(i = 0; i < nt; i++)
    {
        //
        // objective
        //
        // feed forward
        // mathematically from right to left
        // map training data input x to y
        HANNFMap(hannf, hannf->h[nl-1], hannf->x[i]);
        // compute difference
        // compute norm of difference
        // sum up
        VecWAXPY(hannf->w[nl-1], -1.0, hannf->y[i], hannf->h[nl-1]);
        VecNorm(hannf->w[nl-1], NORM_2, &norm);
        sum = sum + norm * norm;
        //
        // gradient
        //
        // backwards
        // mathematically from left to right
        // compute gradient w.r.t. the i_th component of the cost function
        // is case if least squares, input is the difference, here w[nl-1]
        HANNFMapGradient(hannf, hannf->w[nl-1], hannf->x[i], g_i);
        // add to gradient vector
        // g = g + 1.0 * g_i
        VecAXPY(g, 1.0, g_i);
    }
    // weight the sum with one half
    *f = 0.5 * sum;
    // destroy work vector
    VecDestroy(&g_i);
    // debug
    HANNFDebug(hannf, kDebugLevel, "HANNFObjectiveAndGradient\n");
    PetscFunctionReturn(0);
}



