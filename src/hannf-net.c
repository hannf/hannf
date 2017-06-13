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

#include "hannf-net.h"

#undef  __FUNCT__
#define __FUNCT__ "HANNFNetFinal"
PetscErrorCode
HANNFNetFinal(HANNF* hannf)
{
    PetscFunctionBegin;
    if (strcmp(hannf->type, HANNF_TYPE_MLP) == 0) {
        // free hidden layer neuron count
        PetscFree(hannf->nhi);
    } else {
        // unkown HANNF type, abort execution
        char message[PETSC_MAX_PATH_LEN];
        sprintf(message, "Unknown HANNF type '%s'.", hannf->type);
        HANNFFlag(PETSC_FALSE, message);
    }
    // debug
    HANNFDebug(hannf, "HANNFNetFinal\n");
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFNetInit"
PetscErrorCode
HANNFNetInit(HANNF* hannf)
{
    PetscFunctionBegin;
    char annType[PETSC_MAX_PATH_LEN];
    // get network type
    HANNFUtilOptionsGetString(hannf, "-HANNFType", annType);
    if (strcmp(annType, HANNF_TYPE_MLP) == 0) {
        // HANNF_TYPE_MLP, "mlp", multi layer perceptron
        PetscInt nmax;
        // store type
        hannf->type = HANNF_TYPE_MLP;
        // input layer size
        // output layer size
        // hidden layer count
        // hidden layer(s) size(s)
        HANNFUtilOptionsGetInt(hannf, "-HANNFInputNeuronCount", &hannf->nin);
        HANNFUtilOptionsGetInt(hannf, "-HANNFOutputNeuronCount", &hannf->nout);
        HANNFUtilOptionsGetInt(hannf, "-HANNFHiddenLayerCount", &hannf->nh);
        nmax = hannf->nh;
        PetscMalloc(nmax * sizeof(PetscInt), &hannf->nhi);
        HANNFUtilOptionsGetIntArray(hannf, "-HANNFHiddenLayerNeuronCount", &nmax, hannf->nhi);
    } else {
        // unkown HANNF type, abort execution
        char message[PETSC_MAX_PATH_LEN];
        sprintf(message, "Unknown HANNF type '%s'.", annType);
        HANNFFlag(PETSC_FALSE, message);
    }
    // debug
    HANNFDebug(hannf, "HANNFNetInit\n");
    PetscFunctionReturn(0);
}



