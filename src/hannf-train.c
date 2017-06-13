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
    VecDestroy(&hannf->x);
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
    VecDuplicate(hannf->xx, &hannf->x);
    VecSetRandom(hannf->x, PETSC_NULL);
    VecAssemblyBegin(hannf->x);
    VecAssemblyEnd(hannf->x);

    // create optimization context
    // set initial vector
    // set objective and gradient
    TaoCreate(hannf->comm, &hannf->tao);
    TaoSetInitialVector(hannf->tao, hannf->x);
    TaoSetObjectiveAndGradientRoutine(hannf->tao, HANNFObjectiveAndGradient, (void*)hannf);

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

#undef  __FUNCT__
#define __FUNCT__ "HANNFObjectiveAndGradient"
PetscErrorCode
HANNFObjectiveAndGradient(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
    // get context
    HANNF* hannf = (HANNF*) ctx;
    PetscFunctionBegin;
    // work vars
    PetscInt nt, nh, i;
    PetscReal norm, sum;

    // zero the entries of the gradient vector
    // before we sum up all gradients internally
    VecZeroEntries(g);
    
    sum = 0.0;
    nt = hannf->nt;
    nh = hannf->nh;
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
        HANNFMapGradient(hannf, hannf->w[nh], hannf->X[i], g);
    }
    // weight the sum with one half
    *f = 0.5 * sum;

    // debug
    HANNFDebug(hannf, "HANNFObjectiveAndGradient\n");
    PetscFunctionReturn(0);
}









////    VecResetArray(Vec vec)
//
//
//    // free matrices and vectors
//    MatDestroyMatrices(hannf->nh + 1, &hannf->W);
//    VecDestroyVecs(hannf->nh + 1, &hannf->b);
//
//    // free hidden layers counts
//    PetscFree(hannf->nhi);
//
//    // destroy class matrix
////    MatDestroy(&hannf->Y);
////    VecDestroyVecs(hannf->Y
//
//    // destroy feature matrix
////    MatDestroy(&hannf->X);



//    // determine sizes
//    HANNFUtilOptionsGetInt(hannf, "-HANNFTrainingFeatureCount", &hannf->nin);
//    HANNFUtilOptionsGetInt(hannf, "-HANNFTrainingClassCount", &hannf->nout);
//    // debug
////    HANNFDebug(hannf, FSSD, "HANNFInitWithOptionFile", "nin:", hannf->nin);
////    HANNFDebug(hannf, FSSD, "HANNFInitWithOptionFile", "nout:", hannf->nout);
////    // destroy X, Y
////    MatDestroy(&X);
////    MatDestroy(&Y);
//
//    for(i = 0; i < hannf->nt; i++)
//    {
//
//    }
//
//    MatDestroy(&T);
//
////    // read in features
////    HANNFUtilOptionsGetString(hannf, "-HANNFTrainingInputData", filePath);
////    // create matrix
////    Mat X;
////    MatCreate(hannf->comm, &X);
////    MatSetType(X, MATDENSE);
////    // load matrix
////    PetscViewerBinaryOpen(hannf->comm, filePath, FILE_MODE_READ, &viewer);
////    MatLoad(X, viewer);
////    PetscViewerDestroy(&viewer);
////
////    // create vectors with matrix array
////    PetscScalar* Xmatarray;
////    MatDenseGetArray(X, &Xmatarray);
////    Vec xx;
////    MatCreateVecs(X, PETSC_NULL, &xx);
////
//////    VecPlaceArray(xx, &Xmatarray[19]);
//////    VecAssemblyBegin(xx);
//////    VecAssemblyEnd(xx);
////
////
////    VecDestroy(&xx);
////    MatDenseRestoreArray(X, &Xmatarray);
////
////    // read in classes
////    HANNFUtilOptionsGetString(hannf, "-HANNFTrainingOutputData", filePath);
////    // create matrix
////    Mat Y;
////    MatCreate(hannf->comm, &Y);
////    MatSetType(Y, MATDENSE);
////    // load matrix
////    PetscViewerBinaryOpen(hannf->comm, filePath, FILE_MODE_READ, &viewer);
////    MatLoad(Y, viewer);
////    PetscViewerDestroy(&viewer);
//
//



//
//
//// create matrix
//Mat T;
//MatCreate(comm, &T);
//MatSetType(T, MATDENSE);
//// load matrix
//PetscViewerBinaryOpen(comm, filePath, FILE_MODE_READ, &viewer);
//MatLoad(T, viewer);
//PetscViewerDestroy(&viewer);
//
//MatSetRandom(T, PETSC_NULL);
//MatAssemblyBegin(T, MAT_FINAL_ASSEMBLY);
//MatAssemblyEnd(T, MAT_FINAL_ASSEMBLY);
////    MatView(T, PETSC_VIEWER_STDOUT_WORLD);
//
//// determine sequence length
//MatGetSize(T, PETSC_NULL, &hannf->nt);
//// debug
//HANNFDebug(hannf, FSSD, "HANNFTrainDataInit", "nt:", hannf->nt);
//
////    // create mapping
////    PetscInt m, n;
////    MatGetOwnershipRange(T, &m, &n);
////    PetscPrintf(PETSC_COMM_SELF, "m: %d, n: %d\n", m, n);
////
////    IS rows_local;
////    ISCreateStride(comm, n-m, m, 1, &rows_local);
////
////    ISLocalToGlobalMapping rmap;
////    ISLocalToGlobalMappingCreateIS(rows_local, &rmap);
////    ISLocalToGlobalMappingView(rmap, PETSC_VIEWER_STDOUT_WORLD);
////
////    IS cols_local;
////    ISCreateStride(comm, hannf->nt, 0, 1, &cols_local);
////
////    ISLocalToGlobalMapping cmap;
////    ISLocalToGlobalMappingCreateIS(cols_local, &cmap);
////    ISLocalToGlobalMappingView(cmap, PETSC_VIEWER_STDOUT_WORLD);
////
////    MatSetLocalToGlobalMapping(T, rmap, cmap);
////
////    // debug
//////    ISLocalToGlobalMapping rmap;
//////    ISLocalToGlobalMapping cmap;
//////    MatGetLocalToGlobalMapping(T, &rmap, &cmap);;
////
//////    ISLocalToGlobalMappingView(rmap, PETSC_VIEWER_STDOUT_WORLD);
//////    ISLocalToGlobalMappingView(cmap, PETSC_VIEWER_STDOUT_WORLD);
////
////    // split matrix into input and output
////
//////
//////    MatGetSubMatrix(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
////
//////    IS rows, cols;
//////
//////    PetscInt nrow_local;
//////    nrow_local = PETSC_DECIDE;
//////    PetscSplitOwnership(comm, &nrow_local, &hannf->nin);
//////
//////    PetscInt ncol_local;
//////    ncol_local = PETSC_DECIDE;
//////    PetscSplitOwnership(comm, &ncol_local, &hannf->nt);
//////
////////    ISCreateStride(comm, hannf->nin, 0, 1, &rows);
//////    ISCreateStride(comm, nrow_local, 0, 1, &rows);
//////    ISView(rows, PETSC_VIEWER_STDOUT_WORLD);
//////
////////    ISCreateStride(comm, hannf->nt, 0, 1, &cols);
////////    ISCreateStride(comm, 1, 0, 1, &cols);
//////    ISCreateStride(comm, ncol_local, 0, 1, &cols);
//////    ISView(cols, PETSC_VIEWER_STDOUT_WORLD);
////
////    IS rows;
//////    ISCreateStride(comm, hannf->nin, 0, 1, &rows);
////    if (hannf->myproc == 0) {
////        ISCreateStride(comm, 10, 0, 1, &rows);
////    } else {
////        ISCreateStride(comm, 9, 10, 1, &rows);
////    }
////    ISView(rows, PETSC_VIEWER_STDOUT_WORLD);
////
//////    IS cols;
//////    ISCreateStride(comm, hannf->nt, 0, 1, &cols);
//////    ISView(cols, PETSC_VIEWER_STDOUT_WORLD);
//////
////////    MatGetLocalSubMatrix(Mat mat,IS isrow,IS iscol,Mat *submat)
////    MatGetSubMatrix(T, rows, PETSC_NULL, MAT_INITIAL_MATRIX, &hannf->XX);
////////    MatGetLocalSubMatrix(T, rows, PETSC_NULL, &hannf->XX);
//////    MatGetLocalSubMatrix(T, rows, cols, &hannf->XX);
////////    MatAssemblyBegin(hannf->XX, MAT_FINAL_ASSEMBLY);
////////    MatAssemblyEnd(hannf->XX, MAT_FINAL_ASSEMBLY);
////////    MatView(hannf->XX, PETSC_VIEWER_STDOUT_WORLD);
//////
//////
//////////    ISCreateStride(comm, hannf->nout, hannf->nin, 1, &rows);
//////////    ISView(rows, PETSC_VIEWER_STDOUT_WORLD);
//////////
//////////    MatGetSubMatrix(T, rows, cols, MAT_INITIAL_MATRIX, &hannf->YY);
//////////    MatView(hannf->YY, PETSC_VIEWER_STDOUT_WORLD);
////
////
////
////    MatDestroy(&hannf->XX);
//MatDestroy(&T);


//    TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&user);
//    PetscErrorCode TaoSetObjectiveAndGradientRoutine(Tao tao, PetscErrorCode (*func)(Tao, Vec, PetscReal *, Vec, void*), void *ctx)

// material ...
//    ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&user);CHKERRQ(ierr);
//    ierr = TaoSetHessianRoutine(tao,H,H,FormHessian,(void*)&user);CHKERRQ(ierr);


//    PetscInt n, i;
//    n = hannf->nh + 1;

// input X
// W1, b1
// S1 = W1 * X + [b1,...,b1]
// H1 = sigma(S1), elementwise
//    MatMatMult()
//    MatMatMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)

// hidden layers
// S2 = W2 * H1 + [b2,...,b2]
// H2 = sigma(S2), elementwise
// ... and so on, until ...
// output Y

// diff to data
//    hannf->Y - Y

//    *f = 0.0;
