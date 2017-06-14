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

#ifndef HANNF_MAP_H
#define HANNF_MAP_H 1

#include "hannf-load.h"

extern PetscErrorCode HANNFMapInit(HANNF*);
extern PetscErrorCode HANNFMapFinal(HANNF*);

extern PetscErrorCode HANNFMap(HANNF*, Vec, Vec);
extern PetscErrorCode HANNFMapGradient(HANNF*, Vec, Vec, Vec);

extern PetscErrorCode HANNFMapNeuronReceive(HANNF*, Vec, Mat, Vec, Vec);
extern PetscErrorCode HANNFMapNeuronActivate(HANNF*, Vec, Vec, Vec);

#endif /* HANNF_MAP_H */
