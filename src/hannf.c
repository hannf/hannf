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

#include "hannf.h"

#undef  __FUNCT__
#define __FUNCT__ "main"
int
main(int argc, char **args)
{
    // init petsc
    PetscInitialize(&argc, &args, PETSC_NULL, PETSC_NULL);
    PetscPushErrorHandler(PetscAbortErrorHandler, NULL);
//    PetscPushErrorHandler(PetscTraceBackErrorHandler, NULL);
//    PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);

    // check arguments
    if (argc >= 2) {
        // create hannf data type
        // set timing start
        // init, train and final
        HANNF hannf;
        PetscTime(&hannf.startTime);
        HANNFInitWithOptionFile(&hannf, args[1]);
//        HANNFTrain(&hannf);
        HANNFFinal(&hannf);
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "### ERROR:\n");
        PetscPrintf(PETSC_COMM_WORLD, "### ERROR: %s\n", "Please provide an option file!");
        PetscPrintf(PETSC_COMM_WORLD, "### ERROR:\n");
    }

    // final petsc
    PetscPopErrorHandler();
    PetscFinalize();
    return 0;
}


