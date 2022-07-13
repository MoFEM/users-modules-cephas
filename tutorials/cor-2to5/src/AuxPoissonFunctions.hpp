/**
 * \file AuxPoissonFunctions.hpp
 * \example AuxPoissonFunctions.hpp
 *
 */



#ifndef ___AUX_FUNCTIONS_HPP__
#define ___AUX_FUNCTIONS_HPP__

namespace PoissonExample {

struct AuxFunctions {

  AuxFunctions(const MoFEM::Interface &m_field)
      : cOmm(m_field.get_comm()), rAnk(m_field.get_comm_rank()) {}

  /**
   *  Create ghost vector to assemble errors from all element on distributed
   mesh.
   *  Ghost vector has size 1, where one element is owned by processor 0, other
   processor
   *  have one ghost element of zero element at processor 0.

   * [createGhostVec description]
   * @param  ghost_vec pointer to created ghost vector
   * @return           error code
   */
  MoFEMErrorCode createGhostVec(Vec *ghost_vec) const {

    MoFEMFunctionBegin;
    int ghosts[] = {0};
    int nb_locals = rAnk == 0 ? 1 : 0;
    int nb_ghosts = rAnk > 0 ? 1 : 0;
    CHKERR VecCreateGhost(cOmm, nb_locals, 1, nb_ghosts, ghosts, ghost_vec);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Assemble error vector
   */
  MoFEMErrorCode assembleGhostVector(Vec ghost_vec) const {

    MoFEMFunctionBegin;
    CHKERR VecAssemblyBegin(ghost_vec);
    CHKERR VecAssemblyEnd(ghost_vec);
    // accumulate errors from processors
    CHKERR VecGhostUpdateBegin(ghost_vec, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(ghost_vec, ADD_VALUES, SCATTER_REVERSE);
    // scatter errors to all processors
    CHKERR VecGhostUpdateBegin(ghost_vec, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(ghost_vec, INSERT_VALUES, SCATTER_FORWARD);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Print error
   */
  MoFEMErrorCode printError(Vec ghost_vec) {

    MoFEMFunctionBegin;
    double *e;
    CHKERR VecGetArray(ghost_vec, &e);
    CHKERR PetscPrintf(cOmm, "Approximation error %4.3e\n", sqrt(e[0]));
    CHKERR VecRestoreArray(ghost_vec, &e);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Test error
   */
  MoFEMErrorCode testError(Vec ghost_vec) {

    MoFEMFunctionBegin;
    double *e;
    CHKERR VecGetArray(ghost_vec, &e);
    // Check if error is zero, otherwise throw error
    const double eps = 1e-8;
    if ((sqrt(e[0]) > eps) || (!boost::math::isnormal(e[0]))) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
              "Test failed, error too big");
    }
    CHKERR VecRestoreArray(ghost_vec, &e);
    MoFEMFunctionReturn(0);
  }

private:
  MPI_Comm cOmm;
  const int rAnk;
};

} // namespace PoissonExample

#endif //___AUX_FUNCTIONS_HPP__
