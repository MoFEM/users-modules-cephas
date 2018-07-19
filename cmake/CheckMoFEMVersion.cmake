# MoFEM is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# MoFEM is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MoFEM. If not, see <http://www.gnu.org/licenses/>

macro(CHECK_MOFEM_VERSION VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

  if(${MoFEM_VERSION} VERSION_LESS ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})
    message(
      FATAL_ERROR
      "(Update MoFEM) Wrong MoFEM version: "
      "${MoFEM_VERSION} < ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
    )
  endif(${MoFEM_VERSION} VERSION_LESS ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

endmacro(CHECK_MOFEM_VERSION)