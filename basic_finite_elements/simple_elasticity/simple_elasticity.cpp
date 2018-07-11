/** \file simple_elasticity.cpp
 * \ingroup nonlinear_elastic_elem
 * \example simple_elasticity.cpp

 The example shows how to solve the linear elastic problem.

*/

/* MoFEM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * MoFEM is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */

#include <BasicFiniteElements.hpp>

using namespace boost::numeric;
using namespace MoFEM;

static char help[] = "-my_block_config set block data\n"
                     "\n";

struct OpK : public VolumeElementForcesAndSourcesCore::UserDataOperator {

  // B_i matrix
  MatrixDouble rowB;

  // B_j matrix
  MatrixDouble colB;

  // Matrix used to evaluate matrix product D B_j
  MatrixDouble CB;

  // Finite element stiffness sub-matrix K_ij
  MatrixDouble K;

  // Elastic stiffness matrix
  MatrixDouble D;

  // Young's modulus
  double yOung;
  // Poisson's ratio
  double pOisson;

  OpK(bool symm = true)
      : VolumeElementForcesAndSourcesCore::UserDataOperator("U", "U", OPROWCOL,
                                                            symm) {

    // Evaluation of the elastic stiffness matrix, D, in the Voigt notation is
    // done in the constructor

    // hardcoded choice of elastic parameters
    pOisson = 0.1;
    yOung   = 10;

    // coefficient used in intermediate calculation
    double coefficient = 0.;

    coefficient = yOung / ((1 + pOisson) * (1 - 2 * pOisson));
    D.resize(6, 6, false);
    D.clear();

    D(0, 0) = 1 - pOisson;
    D(1, 1) = 1 - pOisson;
    D(2, 2) = 1 - pOisson;
    D(3, 3) = 0.5 * (1 - 2 * pOisson);
    D(4, 4) = 0.5 * (1 - 2 * pOisson);
    D(5, 5) = 0.5 * (1 - 2 * pOisson);

    D(0, 1) = pOisson;
    D(0, 2) = pOisson;
    D(1, 0) = pOisson;

    D(1, 2) = pOisson;
    D(2, 0) = pOisson;
    D(2, 1) = pOisson;

    D *= coefficient;
  }

  /**
  Evaluates B matrix of
  @param diffN array of gradients of shape functions
  @param B returning B matrix
  */
  MoFEMErrorCode makeB(const MatrixAdaptor &diffN, MatrixDouble &B) {

    // initiation of error handler
    MoFEMFunctionBegin;

    // number of gradients of shape functions is passed to nb_dofs variable
    // total number of degrees of freedom is nb_dofs*3
    unsigned int nb_dofs = diffN.size1();

    // inidialise B matrix
    // 6 rows equal to the number of strains in Voigt notation
    // 3 * nb_dofs number of columns equal to the number of degrees of freedom
    // of the element
    B.resize(6, 3 * nb_dofs, false);

    // B matrix is cleared
    B.clear();

    // Loop over degrees of freedom treated as groups of three
    for (unsigned int dd = 0; dd < nb_dofs; ++dd) {

      // array diff containing the gradients of shape functions
      // gradient in x direction for 0, in y direction for 1 and in x for 2
      const double diff[] = {diffN(dd, 0), diffN(dd, 1), diffN(dd, 2)};
      const int dd3       = 3 * dd;
      for (int rr = 0; rr < 3; ++rr) {
        // gamma_xx for rr = 0, gamma_yy for rr = 1 and gamma_zz for rr = 2
        B(rr, dd3 + rr) = diff[rr];
      }
      // gamma_xy
      B(3, dd3 + 0) = diff[1];
      B(3, dd3 + 1) = diff[0];
      // gamma_yz
      B(4, dd3 + 1) = diff[2];
      B(4, dd3 + 2) = diff[1];
      // gamma_xz
      B(5, dd3 + 0) = diff[2];
      B(5, dd3 + 2) = diff[0];
    }

    // End of error handling
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Do calculations for give operator
   * @param  row_side row side number (local number) of entity on element
   * @param  col_side column side number (local number) of entity on element
   * @param  row_type type of row entity MBVERTEX, MBEDGE, MBTRI or MBTET
   * @param  col_type type of column entity MBVERTEX, MBEDGE, MBTRI or MBTET
   * @param  row_data data for row
   * @param  col_data data for column
   * @return          error code
   */
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data) {

    MoFEMFunctionBegin;
    // get number of dofs on row
    nbRows = row_data.getIndices().size();
    // if no dofs on row, exit that work, nothing to do here
    if (!nbRows)
      MoFEMFunctionReturnHot(0);
    // get number of dofs on column
    nbCols = col_data.getIndices().size();
    // if no dofs on Columbia, exit nothing to do here
    if (!nbCols)
      MoFEMFunctionReturnHot(0);
    // get number of integration points
    nbIntegrationPts = getGaussPts().size2();
    // check if entity block is on matrix diagonal
    if (row_side == col_side && row_type == col_type) {
      isDiag = true;
    } else {
      isDiag = false;
    }
    // integrate local matrix for entity block
    CHKERR iNtegrate(row_data, col_data);

    // assemble local matrix
    CHKERR aSsemble(row_data, col_data);

    MoFEMFunctionReturn(0);
  }

protected:
  int nbRows;           ///< number of dofs on rows
  int nbCols;           ///< number if dof on column
  int nbIntegrationPts; ///< number of integration points
  bool isDiag;          ///< true if this block is on diagonal

  /**
   * \brief Integrate B^T D B operator
   * @param  row_data row data (consist base functions on row entity)
   * @param  col_data column data (consist base functions on column entity)
   * @return error code
   */
  virtual MoFEMErrorCode
  iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
            DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    // check if one of i-th or j-th entity sets are not populated
    int nb_dofs_row = row_data.getFieldData().size();
    if (nb_dofs_row == 0)
      MoFEMFunctionReturnHot(0);
    int nb_dofs_col = col_data.getFieldData().size();
    if (nb_dofs_col == 0)
      MoFEMFunctionReturnHot(0);

    // K_ij matrix will have 3 times the number of degrees of freedom of the
    // i-th entity set (nb_dofs_row)
    // and 3 times the number of degrees of freedom of the j-th entity set
    // (nb_dofs_col)
    K.resize(nb_dofs_row, nb_dofs_col, false);
    K.clear();

    // matrix CB will have 6 rows since D matrix has 6 rows
    // number of columns nb_dofs_col  that is equal to the number of columns of
    // B_j matrix
    CB.resize(6, nb_dofs_col, false);

    for (int gg = 0; gg != nbIntegrationPts; ++gg) {
      // get element volume
      // get integration weight
      double val = getVolume() * getGaussPts()(3, gg);

      const MatrixAdaptor &diffN_row = row_data.getDiffN(gg, nb_dofs_row / 3);
      const MatrixAdaptor &diffN_col = col_data.getDiffN(gg, nb_dofs_col / 3);

      // evaluate B_i
      CHKERR makeB(diffN_row, rowB);

      // evaluate B_j
      CHKERR makeB(diffN_col, colB);

      // compute matrix product D B_i
      noalias(CB) = prod(D, colB);

      // compute product (B_j)^T D B_i
      noalias(K) += val * prod(trans(rowB), CB);
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Assemble local entity block matrix
   * @param  row_data row data (consist base functions on row entity)
   * @param  col_data column data (consist base functions on column entity)
   * @return          error code
   */
  virtual MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data,
                                  DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;
    // get pointer to first global index on row
    const int *row_indices = &*row_data.getIndices().data().begin();
    // get pointer to first global index on column
    const int *col_indices = &*col_data.getIndices().data().begin();
    Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                               : getFEMethod()->snes_B;
    // assemble local matrix
    CHKERR MatSetValues(B, nbRows, row_indices, nbCols, col_indices,
                        &*K.data().begin(), ADD_VALUES);

    if (!isDiag && sYmm) {
      // if not diagonal term and since global matrix is symmetric assemble
      // transpose term.
      K = trans(K);
      CHKERR MatSetValues(B, nbCols, col_indices, nbRows, row_indices,
                          &*K.data().begin(), ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }
};

struct OpPressure : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  double pressureVal;

  OpPressure(const double pressure_val = 1)
      :

        MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator("U", OPROW),
        pressureVal(pressure_val) {}

  // vector used to store force vector for each degree of freedom
  VectorDouble nF;

  FTensor::Index<'i', 3> i;

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {

    MoFEMFunctionBegin;
    // check that the faces have associated degrees of freedom
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs == 0)
      MoFEMFunctionReturnHot(0);

    // size of force vector associated to the entity
    // set equal to the number of degrees of freedom of associated with the
    // entity
    nF.resize(nb_dofs, false);
    nF.clear();

    // get number of gauss points
    const int nb_gauss_pts = data.getN().size1();

    // create a 3d vector to be used as the normal to the face with length equal
    // to the face area
    auto t_normal = getTensor1Normal();

    // vector of base functions
    auto t_base = data.getFTensor0N();

    // loop over all gauss points of the face
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      // weight of gg gauss point
      double w = 0.5 * getGaussPts()(2, gg);

      // create a vector t_nf whose pointer points an array of 3 pointers
      // pointing to nF  memory location of components
      FTensor::Tensor1<double *, 3> t_nf(&nF[0], &nF[1], &nF[2], 3);
      for (int bb = 0; bb != nb_dofs / 3; ++bb) {
        // scale the three components of t_normal and pass them to the t_nf
        // (hence to nF)
        t_nf(i) += (w * pressureVal * t_base) * t_normal(i);
        // move the pointer to next element of t_nf
        ++t_nf;
        // move to next base function
        ++t_base;
      }
    }

    // add computed values of pressure in the global right hand side vector
    CHKERR VecSetValues(getFEMethod()->ksp_f, nb_dofs, &data.getIndices()[0],
                        &nF[0], ADD_VALUES);

    MoFEMFunctionReturn(0);
  }
};

struct ApplyDirichletBc : public MoFEM::FEMethod {

  Range fixFaces, fixNodes, fixSecondNode;

  ApplyDirichletBc(const Range &fix_faces, const Range &fix_nodes,
                   const Range &fix_second_node)
      : MoFEM::FEMethod(), fixFaces(fix_faces), fixNodes(fix_nodes),
        fixSecondNode(fix_second_node) {
    // constructor
  }

  MoFEMErrorCode postProcess() {

    MoFEMFunctionBegin;
    std::set<int> set_fix_dofs;

    for (_IT_NUMEREDDOF_ROW_FOR_LOOP_(problemPtr, dit)) {
      if (dit->get()->getDofCoeffIdx() == 2) {
        if (fixFaces.find(dit->get()->getEnt()) != fixFaces.end()) {
          set_fix_dofs.insert(dit->get()->getPetscGlobalDofIdx());
        }
      }

      if (fixSecondNode.find(dit->get()->getEnt()) != fixSecondNode.end()) {
        if (dit->get()->getDofCoeffIdx() == 1) {
          set_fix_dofs.insert(dit->get()->getPetscGlobalDofIdx());
        }
      }

      if (fixNodes.find(dit->get()->getEnt()) != fixNodes.end()) {
        set_fix_dofs.insert(dit->get()->getPetscGlobalDofIdx());
      }
    }

    std::vector<int> fix_dofs(set_fix_dofs.size());

    std::copy(set_fix_dofs.begin(), set_fix_dofs.end(), fix_dofs.begin());

    CHKERR MatAssemblyBegin(ksp_B, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(ksp_B, MAT_FINAL_ASSEMBLY);
    CHKERR VecAssemblyBegin(ksp_f);
    CHKERR VecAssemblyEnd(ksp_f);

    Vec x;

    CHKERR VecDuplicate(ksp_f, &x);
    CHKERR VecZeroEntries(x);
    CHKERR MatZeroRowsColumns(ksp_B, fix_dofs.size(), &fix_dofs[0], 1, x,
                              ksp_f);

    CHKERR VecDestroy(&x);

    MoFEMFunctionReturn(0);
  }
};

int main(int argc, char *argv[]) {

  // Initialize MoFEM
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  // Create mesh database
  moab::Core mb_instance;              // create database
  moab::Interface &moab = mb_instance; // create interface to database

  try {
    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    CHKERR DMRegister_MoFEM("DMMOFEM");

    // Get command line options
    int order          = 3;           // default approximation order
    PetscBool flg_test = PETSC_FALSE; // true check if error is numerical error
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "SimpleElasticProblem",
                             "none");

    // Set approximation order
    CHKERR PetscOptionsInt("-order", "approximation order", "", order, &order,
                           PETSC_NULL);

    // Set testing (used by CTest)
    CHKERR PetscOptionsBool("-test", "if true is ctest", "", flg_test,
                            &flg_test, PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    Simple *simple_interface = m_field.getInterface<MoFEM::Simple>();

    CHKERR simple_interface->getOptions();
    CHKERR simple_interface->loadFile();
    CHKERR simple_interface->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE,
                                            3);
    CHKERR simple_interface->setFieldOrder("U", order);

    Range fix_faces, pressure_faces, fix_nodes, fix_second_node;

    enum MyBcTypes {
      FIX_BRICK_FACES      = 1,
      FIX_NODES            = 2,
      BRICK_PRESSURE_FACES = 3,
      FIX_NODES_Y_DIR      = 4
    };

    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      EntityHandle meshset = bit->getMeshset();
      int id               = bit->getMeshsetId();

      if (id == FIX_BRICK_FACES) { // brick-faces

        CHKERR m_field.get_moab().get_entities_by_dimension(meshset, 2,
                                                            fix_faces, true);

        Range adj_ents;
        CHKERR m_field.get_moab().get_adjacencies(fix_faces, 0, false, adj_ents,
                                                  moab::Interface::UNION);

        CHKERR m_field.get_moab().get_adjacencies(fix_faces, 1, false, adj_ents,
                                                  moab::Interface::UNION);
        fix_faces.merge(adj_ents);
      } else if (id == FIX_NODES) { // node(s)

        CHKERR m_field.get_moab().get_entities_by_dimension(meshset, 0,
                                                            fix_nodes, true);

      } else if (id == BRICK_PRESSURE_FACES) { // brick pressure faces
        CHKERR m_field.get_moab().get_entities_by_dimension(
            meshset, 2, pressure_faces, true);

      } else if (id ==
                 FIX_NODES_Y_DIR) { // restrained second node in y direction
        CHKERR m_field.get_moab().get_entities_by_dimension(
            meshset, 0, fix_second_node, true);

      } else {
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "Unknown blockset");
      }
    }

    CHKERR simple_interface->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE,
                                            3);

    CHKERR simple_interface->setFieldOrder("U", order);

    CHKERR simple_interface->defineFiniteElements();

    // Add pressure element
    CHKERR m_field.add_finite_element("PRESSURE");
    CHKERR m_field.modify_finite_element_add_field_row("PRESSURE", "U");
    CHKERR m_field.modify_finite_element_add_field_col("PRESSURE", "U");
    CHKERR m_field.modify_finite_element_add_field_data("PRESSURE", "U");

    CHKERR simple_interface->defineProblem();

    DM dm;
    CHKERR simple_interface->getDM(&dm);

    CHKERR DMMoFEMAddElement(dm, "PRESSURE");
    CHKERR DMMoFEMSetIsPartitioned(dm, PETSC_TRUE);

    CHKERR simple_interface->buildFields();
    CHKERR simple_interface->buildFiniteElements();

    CHKERR m_field.add_ents_to_finite_element_by_dim(
        0, simple_interface->getDim(), simple_interface->getDomainFEName(),
        true);
    CHKERR m_field.build_finite_elements(simple_interface->getDomainFEName());
    CHKERR m_field.add_ents_to_finite_element_by_dim(pressure_faces, 2,
                                                     "PRESSURE");
    CHKERR m_field.build_finite_elements("PRESSURE", &pressure_faces);

    CHKERR simple_interface->buildProblem();

    boost::shared_ptr<VolumeElementForcesAndSourcesCore> elastic_fe(
        new VolumeElementForcesAndSourcesCore(m_field));

    elastic_fe->getOpPtrVector().push_back(new OpK());

    // push operators to elastic_fe
    boost::shared_ptr<FaceElementForcesAndSourcesCore> pressure_fe(
        new FaceElementForcesAndSourcesCore(m_field));
    pressure_fe->getOpPtrVector().push_back(new OpPressure());

    boost::shared_ptr<FEMethod> fix_dofs_fe(
        new ApplyDirichletBc(fix_faces, fix_nodes, fix_second_node));

    boost::shared_ptr<FEMethod> null_fe;

    // Set operators for KSP solver
    CHKERR DMMoFEMKSPSetComputeOperators(
        dm, simple_interface->getDomainFEName(), elastic_fe, null_fe, null_fe);

    CHKERR DMMoFEMKSPSetComputeRHS(dm, "PRESSURE", pressure_fe, null_fe,
                                   null_fe);

    // initialise matrix A used as the global stiffness matrix
    Mat A;

    // initialise left hand side vector x and right hand side vector f
    Vec x, f;

    // allocate memory handled by MoFEM discrete manager for matrix A
    CHKERR DMCreateMatrix(dm, &A);

    // allocate memory handled by MoFEM discrete manager for vector x
    CHKERR DMCreateGlobalVector(dm, &x);

    // allocate memory handled by MoFEM discrete manager for vector f of the
    // same size as x
    CHKERR VecDuplicate(x, &f);

    // precondition matrix A according to fix_dofs_fe  and elastic_fe finite
    // elements
    elastic_fe->ksp_B  = A;
    fix_dofs_fe->ksp_B = A;

    // precondition the right hand side vector f according to fix_dofs_fe  and
    // elastic_fe finite elements
    fix_dofs_fe->ksp_f = f;
    pressure_fe->ksp_f = f;

    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                    elastic_fe);

    CHKERR DMoFEMLoopFiniteElements(dm, "PRESSURE", pressure_fe);

    // This is done because only post processor is implemented in the
    // ApplyDirichletBc struct
    CHKERR DMoFEMPostProcessFiniteElements(dm, fix_dofs_fe.get());

    // make available a KSP solver
    KSP solver;

    // make the solver available for parallel computing by determining its MPI
    // communicator
    CHKERR KSPCreate(PETSC_COMM_WORLD, &solver);

    // making available running all options available for KSP solver in running
    // command
    CHKERR KSPSetFromOptions(solver);

    // set A matrix with preconditioner
    CHKERR KSPSetOperators(solver, A, A);

    // set up the solver data strucure for the iterative solver
    CHKERR KSPSetUp(solver);

    // solve the system of linear equations
    CHKERR KSPSolve(solver, f, x);

    // destroy solver no needed any more
    CHKERR KSPDestroy(&solver);

    // make vector x available for parallel computations for visualization
    // context
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    // save solution in vector x on mesh
    CHKERR DMoFEMMeshToGlobalVector(dm, x, INSERT_VALUES, SCATTER_REVERSE);

    // Set up post-processor. It is some generic implementation of finite
    // element
    PostProcVolumeOnRefinedMesh post_proc(m_field);
    // Add operators to the elements, starting with some generic
    CHKERR post_proc.generateReferenceElementMesh();

    CHKERR post_proc.addFieldValuesPostProc("U");

    CHKERR post_proc.addFieldValuesGradientPostProc("U");

    CHKERR DMoFEMLoopFiniteElements(
        dm, simple_interface->getDomainFEName().c_str(), &post_proc);

    // write output
    CHKERR post_proc.writeFile("out.h5m");

    {
      if (flg_test == PETSC_TRUE) {

        const PetscReal x_vec_norm_const = 4.975588;

        // Check norm_1  value
        PetscReal norm_check;
        CHKERR VecNorm(x, NORM_1, &norm_check);
        if (fabs(norm_check - x_vec_norm_const) < 1.e-8) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID, "test failed");
        }
      }
    }

    // free memory handled by mofem discrete manager for A, x and f
    CHKERR MatDestroy(&A);
    CHKERR VecDestroy(&x);
    CHKERR VecDestroy(&f);

    // free memory allocated for mofem discrete manager
    CHKERR DMDestroy(&dm);

    // This is a good reference for the future
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}
