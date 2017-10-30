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

#include "hannf-init.h"

#undef  __FUNCT__
#define __FUNCT__ "HANNFInitWithOptionFile"
PetscErrorCode
HANNFInitWithOptionFile(HANNF* hannf, const char* filepath)
{
    PetscFunctionBegin;
    // set communicator
    // set time stamp to start time
    hannf->comm = PETSC_COMM_WORLD;
    hannf->timeStamp = hannf->startTime;
    // insert option file into database
    PetscOptionsInsertFile(hannf->comm, PETSC_NULL, filepath, PETSC_TRUE);
    // read in debug
    HANNFUtilOptionsGetInt(hannf, "-HANNFDebug", &hannf->debug);
    // init net, map, train, ...
    HANNFNetInit(hannf);
    HANNFLoadInit(hannf);
    HANNFMapInit(hannf);
//    HANNFTrainInit(hannf);
    // wait for all processors
    PetscBarrier(PETSC_NULL);
    // debug
    HANNFDebug(hannf, FSSS, "HANNFInitWithOptionFile", "filepath:", filepath);
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFFinal"
PetscErrorCode
HANNFFinal(HANNF* hannf)
{
    PetscFunctionBegin;
    // wait for all processors
    PetscBarrier(PETSC_NULL);
    // final ..., train, map, net
//    HANNFTrainFinal(hannf);
    HANNFMapFinal(hannf);
    HANNFLoadFinal(hannf);
    HANNFNetFinal(hannf);
    // debug
    HANNFDebug(hannf, "HANNFFinal\n");
    PetscFunctionReturn(0);
}


