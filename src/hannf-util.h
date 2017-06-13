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

#ifndef HANNF_UTIL_H
#define HANNF_UTIL_H 1

#include "hannf-debug.h"

extern PetscErrorCode HANNFUtilOptionsGetInt(HANNF*, const char*, PetscInt*);
extern PetscErrorCode HANNFUtilOptionsGetIntArray(HANNF*, const char*, PetscInt*, PetscInt*);
extern PetscErrorCode HANNFUtilOptionsGetString(HANNF*, const char*, char*);

#endif /* HANNF_UTIL_H */
