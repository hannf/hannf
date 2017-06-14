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

#ifndef HANNF_TRAIN_H
#define HANNF_TRAIN_H 1

#include "hannf-map.h"

extern PetscErrorCode HANNFTrainInit(HANNF*);
extern PetscErrorCode HANNFTrainFinal(HANNF*);

extern PetscErrorCode HANNFTrainDataInit(HANNF*);
extern PetscErrorCode HANNFTrainDataFinal(HANNF*);

extern PetscErrorCode HANNFTrain(HANNF*);
extern PetscErrorCode HANNFObjectiveAndGradient(Tao, Vec, PetscReal*, Vec, void*);

#endif /* HANNF_TRAIN_H */
