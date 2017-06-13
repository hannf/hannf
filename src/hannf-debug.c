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

#include "hannf-debug.h"

#undef  __FUNCT__
#define __FUNCT__ "HANNFDebug"
PetscErrorCode
HANNFDebug(HANNF* hannf, const char *format, ...)
{
    PetscFunctionBegin;
    if (hannf->debug > 0) {
        PetscMPIInt rank;
        MPI_Comm_rank(hannf->comm, &rank);
        if (!rank) {
            // work vars
            char            newformat[PETSC_MAX_PATH_LEN];
            va_list         args;
            PetscLogDouble  endTime, elapsedTime, elapsedTimeDelta;

            // get end time
            // compute elapsed time
            // and init new timing
            PetscTime(&endTime);
            elapsedTime      = endTime - hannf->startTime;
            elapsedTimeDelta = endTime - hannf->timeStamp;
            sprintf(newformat, "%8.1fs %6.3fs %s", elapsedTime, elapsedTimeDelta, format);
            PetscTime(&hannf->timeStamp);
            
            // get variable arg list
            va_start(args, format);
            // print with petsc
            PetscVFPrintf(PETSC_STDOUT, newformat, args);
            va_end(args);
        }
    }
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFFlag"
PetscErrorCode
HANNFFlag(PetscBool flag, char *message)
{
    PetscFunctionBegin;
    if (flag == PETSC_FALSE) {
        PetscPrintf(PETSC_COMM_WORLD, "\n");
        PetscPrintf(PETSC_COMM_WORLD, "### ERROR:\n");
        PetscPrintf(PETSC_COMM_WORLD, "### ERROR: %s\n", message);
        PetscPrintf(PETSC_COMM_WORLD, "### ERROR:\n");
        PetscPrintf(PETSC_COMM_WORLD, "\n");
        PetscEnd();
    }
    PetscFunctionReturn(0);
}


