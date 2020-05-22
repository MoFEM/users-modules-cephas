/** \file simple_elasticity.cpp
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
#include "./TransverseIsotropicElasticityTensor.h"

using namespace boost::numeric;
using namespace MoFEM;

static char help[] = "-order approximation order\n"
                     "\n";


struct OpK : public VolumeElementForcesAndSourcesCore::UserDataOperator {

  // Finite element stiffness sub-matrix K_ij
  MatrixDouble K;

  // Elastic stiffness tensor (4th rank tensor with minor and major symmetry)
  // FTensor::Ddg<double, 3, 3> tD;
  FTensor::Tensor4<double, 3, 3, 3, 3> tD;

  OpK(FTensor::Tensor4<double, 3, 3, 3, 3> t_elastic_In, bool symm = true)
      : VolumeElementForcesAndSourcesCore::UserDataOperator("U", "U", OPROWCOL,
                                                            symm) {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Index<'l', 3> l;

    tD = t_elastic_In;
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

    // K_ij matrix will have 3 times the number of degrees of freedom of the
    // i-th entity set (nbRows)
    // and 3 times the number of degrees of freedom of the j-th entity set
    // (nbCols)
    K.resize(nbRows, nbCols, false);
    K.clear();

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
  MoFEMErrorCode
  iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
            DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    // get sub-block (3x3) of local stiffens matrix, here represented by second
    // order tensor
    auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Index<'l', 3> l;

    // get element volume
    double vol = getVolume();

    // get intergration weights
    auto t_w = getFTensor0IntegrationWeight();

    // get derivatives of base functions on rows
    auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
    // iterate over integration points
    for (int gg = 0; gg != nbIntegrationPts; ++gg) {

      // calculate scalar weight times element volume
      const double a = t_w * vol;

      // iterate over row base functions
      for (int rr = 0; rr != nbRows / 3; ++rr) {

        // get sub matrix for the row
        auto t_m = get_tensor2(K, 3 * rr, 0);


        // get derivatives of base functions for columns
        auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

        // iterate column base functions
        for (int cc = 0; cc != nbCols / 3;++cc) {

          // integrate block local stiffens matrix
          t_m(i, k) +=
              a * (tD(i, j, k, l) * (t_row_diff_base(j) * t_col_diff_base(l)));

          // move to next column base function
          ++t_col_diff_base;

          // move to next block of local stiffens matrix
          ++t_m;
        }

        // move to next row base function
        ++t_row_diff_base;
      }

      // move to next integration weight
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Assemble local entity block matrix
   * @param  row_data row data (consist base functions on row entity)
   * @param  col_data column data (consist base functions on column entity)
   * @return          error code
   */
  MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data,
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


struct OpInitialStress : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

  // Vector used to store equivalent force vector of the initial stresses
  VectorDouble F0;

  // Initial stresses tensor
  FTensor::Tensor2<double, 3, 3> tSigma0;

  OpInitialStress(FTensor::Tensor2<double, 3, 3> t_sigma0):
  MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator("U", OPROW),
  tSigma0(t_sigma0) {
    // constructor
  }

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {

    MoFEMFunctionBegin;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    // get number of dofs
    const size_t nbDofs = data.getIndices().size();
    // if no dofs, exit that work, nothing to do here
    if (!nbDofs)
      MoFEMFunctionReturnHot(0);

    F0.resize(nbDofs, false);
    F0.clear();

    const size_t nbBaseFunctions = data.getN().size2();
    if (3 * nbBaseFunctions < nbDofs) {
      SETERRQ(
          PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
          "Number of DOFs is larger than number of base functions on entity");
    }

    // get number of integration points
    const size_t nbGaussPts = data.getN().size1();

    auto t_w = getFTensor0IntegrationWeight();
    auto t_diff_base = data.getFTensor1DiffN<3>();

    double vol = getVolume();

    for (size_t gg = 0; gg != nbGaussPts; ++gg) {

      double alpha = vol * t_w;

      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_F0(&F0[0], &F0[1], &F0[2]);

      size_t bb = 0;
      for(; bb != nbDofs / 3; ++bb) {

        t_F0(i) += -alpha * tSigma0(i,j) * t_diff_base(j);

        ++t_diff_base;

        ++t_F0;
      }
      for(; bb < nbBaseFunctions; ++bb) ++t_diff_base;

      ++t_w;
    }

    // add computed values of equivalent force in the global right hand side vector
    CHKERR VecSetValues(getFEMethod()->ksp_f, nbDofs, &data.getIndices()[0],
                        &F0[0], ADD_VALUES);

    MoFEMFunctionReturn(0);
  }
};



struct OpPressure : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  double pressureVal;

  OpPressure(const double pressure_val = -100)
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator("U", OPROW),
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
    auto t_normal = getFTensor1Normal();

    // get intergration weights
    auto t_w = getFTensor0IntegrationWeight();

    // vector of base functions
    auto t_base = data.getFTensor0N();

    // loop over all gauss points of the face
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      // weight of gg gauss point
      double w = 0.5 * t_w;

      // create a vector t_nf whose pointer points an array of 3 pointers
      // pointing to nF  memory location of components
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf(&nF[0], &nF[1],
                                                              &nF[2]);
      for (int bb = 0; bb != nb_dofs / 3; ++bb) {
        // scale the three components of t_normal and pass them to the t_nf
        // (hence to nF)
        t_nf(i) += (w * pressureVal * t_base) * t_normal(i);
        // move the pointer to next element of t_nf
        ++t_nf;
        // move to next base function
        ++t_base;
      }

      // move to next integration weight
      ++t_w;
    }

    // add computed values of pressure in the global right hand side vector
    CHKERR VecSetValues(getFEMethod()->ksp_f, nb_dofs, &data.getIndices()[0],
                        &nF[0], ADD_VALUES);

    MoFEMFunctionReturn(0);
  }
};

struct ApplyDirichletBc : public MoFEM::FEMethod {

  // Range fixFaces, fixNodes, fixSecondFaces;
  Range fixFacesX, fixFacesY, fixFacesZ;

  ApplyDirichletBc(const Range &fix_faces_x, const Range &fix_faces_y, const Range &fix_faces_z) : 
  MoFEM::FEMethod(), 
  fixFacesX(fix_faces_x), 
  fixFacesY(fix_faces_y), 
  fixFacesZ(fix_faces_z) {}

  MoFEMErrorCode postProcess() {

    MoFEMFunctionBegin;
    std::set<int> set_fix_dofs;

    for (_IT_NUMEREDDOF_ROW_FOR_LOOP_(problemPtr, dit)) {

      if (fixFacesX.find(dit->get()->getEnt()) != fixFacesX.end()) { // fix faces X dir

        if(dit->get()->getDofCoeffIdx() == 0) {

          set_fix_dofs.insert(dit->get()->getPetscGlobalDofIdx());
        }
      }

      if (fixFacesY.find(dit->get()->getEnt()) != fixFacesY.end()) { // fix faces Y dir

        if(dit->get()->getDofCoeffIdx() == 1) {

          set_fix_dofs.insert(dit->get()->getPetscGlobalDofIdx());
        }
      }

      if (fixFacesZ.find(dit->get()->getEnt()) != fixFacesZ.end()) { // fix faces Z dir

        if(dit->get()->getDofCoeffIdx() == 2) {

          set_fix_dofs.insert(dit->get()->getPetscGlobalDofIdx());
        }
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

/**
 * \brief Set integration rule to volume elements
 *
 * This rule is used to integrate \f$\nabla v \cdot \nabla u\f$, thus
 * if the approximation field and the testing field are polynomials of order "p",
 * then the rule for the exact integration is 2*(p-1).
 *
 * Integration rule is order of polynomial which is calculated exactly. Finite
 * element selects integration method based on return of this function.
 *
 */
struct VolRule {
  int operator()(int, int, int p) const {
     return 2 * (p - 1); 
  }
};

/**
 * \brief Set integration rule to boundary elements
 *
 * This rule is used to integrate the work of external forces on a face, 
 * i.e. \f$f \cdot v\f$, where f is the traction vector and v is the test
 * vector function. The current problem features a Neumann bc with 
 * a pre-defined constant pressure. Therefore, if the test field is 
 * represented by polynomials of order "p", then the rule for the exact 
 * integration is also p.
 *
 * Integration rule is order of polynomial which is calculated exactly. Finite
 * element selects integration method based on return of this function.
 *
 */
struct FaceRule {
  int operator()(int, int, int p) const {
    return p;
  }
};

int main(int argc, char *argv[]) {

  FTensor::Tensor4<double, 3, 3, 3, 3> t_elastic;

  TransverseIsotropicElasticityTensor TI_MAT({0.0,0.0,0.0},           // euler angle, in degree, material coordinate
                                                    10.0e9,           // EL
                                                    20.0e9,           // ET
                                                     7.0e9,           // Gxz and Gyz
                                                       0.1,           // Vxy
                                                       0.1);          // Vzx and Vzy
  
  TI_MAT.calculateElasticityTensor();
  t_elastic = TI_MAT.getElasticityTensor();

  // Initial stresses tensor
  FTensor::Tensor2<double, 3, 3> t_sigma0;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  t_sigma0(i,j) = 0.0;
  t_sigma0(0,0) = -10.0e6;
  t_sigma0(1,1) = -15.0e6;
  t_sigma0(2,2) = -25.0e6;

  // internal pressure
  const double fluid_pressure = -8.0e6;

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_package mumps \n"
                                 "-ksp_monitor\n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

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
    
    Range fix_faces_x, fix_faces_y, fix_faces_z, pressure_faces;

    enum MyBcTypes {
      // FIX_BRICK_FACES      = 1,
      // FIX_NODES            = 2,
      // BRICK_PRESSURE_FACES = 3,
      // FIX_FACES_Y_DIR      = 4
      FIX_FACES_X_DIR = 1,
      FIX_FACES_Y_DIR = 2,
      FIX_FACES_Z_DIR = 3,
      PRESSURE_FACES  = 4
    };

    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      EntityHandle meshset = bit->getMeshset();
      int id               = bit->getMeshsetId();

      if (id == FIX_FACES_X_DIR) { // brick-faces

        CHKERR m_field.get_moab().get_entities_by_dimension(meshset, 2,
                                                            fix_faces_x, true);

        Range adj_ents_x;
        CHKERR m_field.get_moab().get_adjacencies(fix_faces_x, 0, false, adj_ents_x,
                                                  moab::Interface::UNION);

        CHKERR m_field.get_moab().get_adjacencies(fix_faces_x, 1, false, adj_ents_x,
                                                  moab::Interface::UNION);
        fix_faces_x.merge(adj_ents_x);

      }else if (id == FIX_FACES_Y_DIR){

        CHKERR m_field.get_moab().get_entities_by_dimension(meshset, 2,
                                                            fix_faces_y, true);

        Range adj_ents_y;
        CHKERR m_field.get_moab().get_adjacencies(fix_faces_y, 0, false, adj_ents_y,
                                                  moab::Interface::UNION);

        CHKERR m_field.get_moab().get_adjacencies(fix_faces_y, 1, false, adj_ents_y,
                                                  moab::Interface::UNION);
        fix_faces_y.merge(adj_ents_y);

      }else if (id == FIX_FACES_Z_DIR){

        CHKERR m_field.get_moab().get_entities_by_dimension(meshset, 2,
                                                            fix_faces_z, true);

        Range adj_ents_z;
        CHKERR m_field.get_moab().get_adjacencies(fix_faces_z, 0, false, adj_ents_z,
                                                  moab::Interface::UNION);

        CHKERR m_field.get_moab().get_adjacencies(fix_faces_z, 1, false, adj_ents_z,
                                                  moab::Interface::UNION);
        fix_faces_z.merge(adj_ents_z);

      } else if (id == PRESSURE_FACES) { // wellbore pressure face

        CHKERR m_field.get_moab().get_entities_by_dimension(
            meshset, 2, pressure_faces, true);

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

    CHKERR m_field.add_ents_to_finite_element_by_dim(pressure_faces, 2,
                                                     "PRESSURE");
    CHKERR m_field.build_finite_elements("PRESSURE", &pressure_faces);

    CHKERR simple_interface->buildProblem();

    // Create elements instances
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> elastic_fe(
        new VolumeElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<FaceElementForcesAndSourcesCore> pressure_fe(
        new FaceElementForcesAndSourcesCore(m_field));

    // Set integration rule to elements instances
    elastic_fe->getRuleHook = VolRule();
    pressure_fe->getRuleHook = FaceRule();

    // Add operators to element instances
    // push operators to elastic_fe
    elastic_fe->getOpPtrVector().push_back(new OpK(t_elastic));
    elastic_fe->getOpPtrVector().push_back(new OpInitialStress(t_sigma0));
    // push operators to pressure_fe
    pressure_fe->getOpPtrVector().push_back(new OpPressure(fluid_pressure));

    boost::shared_ptr<FEMethod> fix_dofs_fe(
        new ApplyDirichletBc(fix_faces_x, fix_faces_y, fix_faces_z));

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
    elastic_fe->ksp_f  = f;

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
        const double x_vec_norm_const = 0.4;
        // Check norm_1  value
        double norm_check;
        // Takes maximal element of the vector, that should be maximal
        // displacement at the end of the bar
        CHKERR VecNorm(x, NORM_INFINITY, &norm_check);
        if (std::abs(norm_check - x_vec_norm_const) / x_vec_norm_const > 1.e-10) {
          SETERRQ1(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
           "test failed (nrm 2 %6.4e)", norm_check);
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