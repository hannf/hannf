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

#include "hannf-util.h"

#define kDebugLevel kDebugLevel1

#undef  __FUNCT__
#define __FUNCT__ "HANNFUtilOptionsGetInt"
PetscErrorCode
HANNFUtilOptionsGetInt(HANNF* hannf, const char* optionName, PetscInt* ivalue)
{
    PetscBool   flag = PETSC_FALSE;
    char        message[PETSC_MAX_PATH_LEN];
    PetscFunctionBegin;
    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, optionName, ivalue, &flag);
    sprintf(message, "Please provide the '%s' option", optionName);
    HANNFFlag(flag, message);
    // debug
    HANNFDebug(hannf, kDebugLevel, F4SD, "HANNFUtilOptionsGetInt", "optionName:", optionName, "value:", *ivalue);
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFUtilOptionsGetString"
PetscErrorCode
HANNFUtilOptionsGetString(HANNF* hannf, const char *optionName, char *string)
{
    PetscBool   flag = PETSC_FALSE;
    char        message[PETSC_MAX_PATH_LEN];
    PetscFunctionBegin;
    PetscOptionsGetString(PETSC_NULL, PETSC_NULL, optionName, string, PETSC_MAX_PATH_LEN, &flag);
    sprintf(message, "Please provide the '%s' option", optionName);
    HANNFFlag(flag, message);
    // debug
    HANNFDebug(hannf, kDebugLevel, F5S, "HANNFUtilOptionsGetString", "optionName:", optionName, "value:", string);
    PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "HANNFUtilOptionsGetIntArray"
PetscErrorCode
HANNFUtilOptionsGetIntArray(HANNF* hannf, const char* optionName, PetscInt* nmax, PetscInt* ivalue)
{
    PetscBool   flag = PETSC_FALSE;
    char        message[PETSC_MAX_PATH_LEN];
    PetscInt    i;
    PetscFunctionBegin;
    PetscOptionsGetIntArray(PETSC_NULL, PETSC_NULL, optionName, ivalue, nmax, &flag);
    sprintf(message, "Please provide the '%s' option", optionName);
    HANNFFlag(flag, message);
    for (i=0; i<(*nmax); i++)
    {
        HANNFDebug(hannf, kDebugLevel, F4SD, "HANNFUtilOptionsGetIntArray", "optionName:", optionName, "value:", ivalue[i]);
    }
    // debug
    PetscFunctionReturn(0);
}



