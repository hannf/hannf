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

// formats
#define F2S     "%-40s %18s\n"                                                      // 2 x string
#define F2SE    "%-40s %18s %-8e\n"                                                 // 2 x string, 1 x double
#define F3S     "%-40s %18s %-42s\n"                                                // 3 x string
#define F2SD    "%-40s %18s %-8d\n"                                                 // 2 x string, 1 x int
#define FSSDSE  "%-40s %18s %-8d %14s %-8e\n"                                       // 2 x string, 1 x int, 1 x string, 1 x double
#define FS2SESD "%-40s %18s %-8e %14s %-8d %14s %-8e %14s %-8d\n"
#define F4SD    "%-40s %18s %-42s %14s %-8d\n"                                      // 4 x string, 1 x int
#define F4SE    "%-40s %18s %-42s %14s %-8e\n"                                      // 4 x string, 1 x double
#define F5S     "%-40s %18s %-42s %14s %-42s\n"                                     // 5 x string
#define FS5SD   "%-40s %18s %-8d, %14s %-8d, %14s %-8d, %14s %-8d, %14s %-8d\n"     // 1 x string, 5 x (1 x string, 1 x int)
#define FDSE    "%0004d %s %.12e\n"
#define FDSEE   "%0004d %s %.12e %.12e\n"

#define FSSDSD  "%-40s %18s %-8d, %14s %-8d, %14s %-8d\n"
#define FSSD    "%-40s %18s %-8d\n"
#define FSSS    "%-40s %18s %-42s\n"                                                // 3 x string

extern PetscErrorCode HANNFDebug(HANNF*, const char*, ...);
extern PetscErrorCode HANNFFlag(PetscBool, char*);

#endif /* HANNF_DEBUG_H */


