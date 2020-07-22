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

find_package(Git)

function(get_git_hash GIT_DIR _hashvar)
  execute_process(COMMAND
    "${GIT_EXECUTABLE}"  rev-parse  HEAD
    WORKING_DIRECTORY ${GIT_DIR}
    OUTPUT_VARIABLE HEAD_HASH
    RESULT_VARIABLE res)
  if(NOT ${res})
    string(REGEX REPLACE "\n$" "" HEAD_HASH "${HEAD_HASH}")
    set(${_hashvar} "${HEAD_HASH}" PARENT_SCOPE CACHE STRING "Git hash" FORCE)
  else(NOT ${res})
    set(${_hashvar} "SHA1-NOT FOUND" PARENT_SCOPE CACHE STRING "Git hash" FORCE)
  endif(NOT ${res})
endfunction()

function(get_git_tag GIT_DIR FALLBACK _gittag) 
  execute_process(COMMAND
    "${GIT_EXECUTABLE}" describe --tags 
    WORKING_DIRECTORY ${GIT_DIR}
    OUTPUT_VARIABLE GIT_TAG
    RESULT_VARIABLE res)
  if(NOT ${res})
    string(REGEX REPLACE "\n$" "" GIT_TAG "${GIT_TAG}")
    set(${_gittag} "${GIT_TAG}" PARENT_SCOPE CACHE STRING "Git tag" FORCE)
  else(NOT ${res})
    set(
      ${_gittag} "${FALLBACK}-fallback" PARENT_SCOPE 
      CACHE STRING "Git tag" FORCE)
  endif(NOT ${res}) 
endfunction()

function(get_git_version
  GIT_TAG_VERSION _version_major _version_minor _version_build) 
  string(REGEX REPLACE 
    "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${GIT_TAG_VERSION}")
  string(REGEX REPLACE 
    "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${GIT_TAG_VERSION}")
  string(REGEX REPLACE 
    "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_BUILD "${GIT_TAG_VERSION}")
  set(
    ${_version_major} "${VERSION_MAJOR}" PARENT_SCOPE 
    CACHE STRING "Major version" FORCE)
  set(${_version_minor} "${VERSION_MINOR}" PARENT_SCOPE
    CACHE STRING "Mainor version" FORCE)
  set(${_version_build} "${VERSION_BUILD}" PARENT_SCOPE
    CACHE STRING "Build version" FORCE)
endfunction()