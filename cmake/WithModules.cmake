# Those modules could be downloaded with MoFEM.
# for testing with docker container.

if(WITH_MODULE_OBSOLETE)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/obsolete)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_obsolete.git obsolete
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/obsolete)
endif(WITH_MODULE_OBSOLETE)

if(WITH_MODULE_HOMOGENISATION)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/homogenisation)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_homogenisation homogenisation
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/homogenisation)
endif(WITH_MODULE_HOMOGENISATION)

if(WITH_MODULE_BONE_REMODELLING)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/bone_remodelling)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_bone_remodelling.git bone_remodelling
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/bone_remodelling)
endif(WITH_MODULE_BONE_REMODELLING)

 if(WITH_MODULE_MWLS_APPROX)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/mwls_approx)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone git clone https://bitbucket.org/karol41/mofem_um_mwls.git mwls_approx
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/mwls_approx)
endif(WITH_MODULE_MWLS_APPROX)

if(WITH_MODULE_FRACTURE_MECHANICS)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/fracture_mechanics)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_fracture_mechanics fracture_mechanics
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/fracture_mechanics)
endif(WITH_MODULE_FRACTURE_MECHANICS)

if(WITH_MODULE_GELS)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/gels)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_gels gels
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/gels)
endif(WITH_MODULE_GELS)

if(WITH_MODULE_STRAIN_PLASTICITY)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/small_strain_plasticity)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_small_strain_plasticity small_strain_plasticity
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/small_strain_plasticity)
endif(WITH_MODULE_STRAIN_PLASTICITY)

if(WITH_MODULE_SOLID_SHELL_PRISM_ELEMENT)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/solid_shell_prism_element)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_solid_shell_prism_element solid_shell_prism_element
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/solid_shell_prism_element)
endif(WITH_MODULE_SOLID_SHELL_PRISM_ELEMENT)

if(WITH_MODULE_MINIMAL_SURFACE_EQUATION)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/minimal_surface_equation)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_minimal_surface_equation minimal_surface_equation
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/minimal_surface_equation)
endif(WITH_MODULE_MINIMAL_SURFACE_EQUATION)
