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

#ifndef HANNF_DEBUG_H
#define HANNF_DEBUG_H 1

#include "hannf-type.h"

// debug levels
#define kDebugLevel0 0
#define kDebugLevel1 1
#define kDebugLevel2 2
#define kDebugLevel3 3
#define kDebugLevel4 4

// formats
#define FSS         "%-35s %18s\n"                                                      // 2 x string
#define FSSE        "%-35s %18s %-8e\n"                                                 // 2 x string, 1 x double
#define FSSS        "%-35s %18s %-35s\n"                                                // 3 x string
#define FSSD        "%-35s %18s %-8d\n"                                                 // 2 x string, 1 x int
#define FSSDSE      "%-35s %18s %-8d %14s %-8e\n"                                       // 2 x string, 1 x int, 1 x string, 1 x double
#define FS2SESD     "%-35s %18s %-8e %14s %-8d %14s %-8e %14s %-8d\n"
#define FSSSSD      "%-35s %18s %-35s %14s %-8d\n"                                      // 4 x string, 1 x int
#define F4SE        "%-35s %18s %-35s %14s %-8e\n"                                      // 4 x string, 1 x double
#define F5S         "%-35s %18s %-35s %14s %-35s\n"                                     // 5 x string
#define FS5SD       "%-35s %18s %-8d, %14s %-8d, %14s %-8d, %14s %-8d, %14s %-8d\n"     // 1 x string, 5 x (1 x string, 1 x int)
#define FDSE        "%0004d %s %.12e\n"
#define FDSEE       "%0004d %s %.12e %.12e\n"
#define FSSDSD      "%-35s %18s %-8d %18s %-8d\n"
#define FSSDSDSD    "%-35s %18s %-8d, %14s %-8d, %14s %-8d\n"

extern PetscErrorCode HANNFDebug(HANNF*, PetscInt, const char*, ...);
extern PetscErrorCode HANNFDebugSynchronizedFSSDSD(HANNF*, PetscInt, const char*, const char*, PetscInt, const char*, PetscInt);
extern PetscErrorCode HANNFFlag(PetscBool, char*);

#endif /* HANNF_DEBUG_H */


