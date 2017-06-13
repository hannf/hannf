#
# Copyright (C) 2017  Jaroslaw Piwonski, CAU, jpi@informatik.uni-kiel.de
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# training
HANNF_TRAIN_PROGRAM = hannf-train.exe
HANNF_TRAIN_OBJCS = \
	src/hannf-debug.o \
	src/hannf-util.o \
	src/hannf-init.o \
	src/hannf-net.o \
	src/hannf-map.o \
	src/hannf-train.o \
	src/hannf-main-train.o

CLEANFILES = $(HANNF_TRAIN_OBJCS) $(HANNF_TRAIN_PROGRAM)

all: $(HANNF_TRAIN_PROGRAM)

include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

$(HANNF_TRAIN_PROGRAM): $(HANNF_TRAIN_OBJCS)
	-$(CLINKER) -o $@ $(HANNF_TRAIN_OBJCS) $(PETSC_LIB)
