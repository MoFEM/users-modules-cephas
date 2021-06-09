/**
 * \file eigen_elastic.cpp
 * \example eigen_elastic.cpp
 *
 * Calculate natural frequencies in 2d and 3d problems.
 *
 */

/* This file is part of MoFEM.
 * MoFEM is free software: you can redistribute it and/or modify it under
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

#include <MoFEM.hpp>
#undef EPS
#include <slepceps.h>

using namespace MoFEM;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = FaceElementForcesAndSourcesCoreBase;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcEle = PostProcVolumeOnRefinedMesh;
};

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = ElementsAndOps<SPACE_DIM>::DomainEleOp;
using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;

using OpK = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;

double rho = 7829e-12;
double young_modulus = 207e3;
double poisson_ratio = 0.33;

double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

int order = 1;

#include <OpPostProcElastic.hpp>
using namespace Tutorial;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();

  MatrixDouble invJac;
  boost::shared_ptr<MatrixDouble> matGradPtr;
  boost::shared_ptr<MatrixDouble> matStrainPtr;
  boost::shared_ptr<MatrixDouble> matStressPtr;
  boost::shared_ptr<MatrixDouble> matDPtr;

  SmartPetscObj<Mat> M;
  SmartPetscObj<Mat> K;
  SmartPetscObj<EPS> ePS;

  std::array<SmartPetscObj<Vec>, 6> rigidBodyMotion;
};

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-rho", &rho, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-young_modulus", &young_modulus,
                               PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-poisson_ratio", &poisson_ratio,
                               PETSC_NULL);

  auto set_matrial_stiffens = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    MoFEMFunctionBegin;
    auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*matDPtr);

    const double A = (SPACE_DIM == 2)
                         ? 2 * shear_modulus_G /
                               (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                         : 1;
    t_D(i, j, k, l) = 2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l);

    MoFEMFunctionReturn(0);
  };

  matGradPtr = boost::make_shared<MatrixDouble>();
  matStrainPtr = boost::make_shared<MatrixDouble>();
  matStressPtr = boost::make_shared<MatrixDouble>();
  matDPtr = boost::make_shared<MatrixDouble>();

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  matDPtr->resize(size_symm * size_symm, 1);

  CHKERR set_matrial_stiffens();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();

  MOFEM_LOG("EXAMPLE", Sev::inform)
      << "Read mesh for problem in " << EXECUTABLE_DIMENSION;
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();
  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  auto *simple = mField.getInterface<Simple>();
  MoFEMFunctionBegin;
  // Add field
  CHKERR simple->addDomainField("U", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE,
                                SPACE_DIM);
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  auto *simple = mField.getInterface<Simple>();
  MoFEMFunctionBegin;

  rigidBodyMotion[0] = smartCreateDMVector(simple->getDM());
  for (int n = 1; n != 6; ++n)
    rigidBodyMotion[n] = smartVectorDuplicate(rigidBodyMotion[0]);

  // Create space of vectors or rigid motion
  auto problem_ptr = mField.get_problem(simple->getProblemName());
  auto dofs = problem_ptr->getNumeredRowDofsPtr();

  // Get all vertices
  const auto bit_number = mField.get_field_bit_number("U");
  auto lo_uid =
      DofEntity::getUniqueIdCalculate(0, get_id_for_min_type<MBVERTEX>());
  auto hi_uid =
      DofEntity::getUniqueIdCalculate(2, get_id_for_max_type<MBVERTEX>());

  auto hi = dofs->upper_bound(lo_uid);
  std::array<double, 3> coords;

  for (auto lo = dofs->lower_bound(lo_uid); lo != hi; ++lo) {

    if ((*lo)->getPart() == mField.get_comm_rank()) {

      auto ent = (*lo)->getEnt();
      CHKERR mField.get_moab().get_coords(&ent, 1, coords.data());

      if ((*lo)->getDofCoeffIdx() == 0) {
        CHKERR VecSetValue(rigidBodyMotion[0], (*lo)->getPetscGlobalDofIdx(), 1,
                           INSERT_VALUES);
        CHKERR VecSetValue(rigidBodyMotion[3], (*lo)->getPetscGlobalDofIdx(),
                           -coords[1], INSERT_VALUES);
        if (SPACE_DIM == 3)
          CHKERR VecSetValue(rigidBodyMotion[4], (*lo)->getPetscGlobalDofIdx(),
                             -coords[2], INSERT_VALUES);

      } else if ((*lo)->getDofCoeffIdx() == 1) {
        CHKERR VecSetValue(rigidBodyMotion[1], (*lo)->getPetscGlobalDofIdx(), 1,
                           INSERT_VALUES);
        CHKERR VecSetValue(rigidBodyMotion[3], (*lo)->getPetscGlobalDofIdx(),
                           coords[0], INSERT_VALUES);
        if (SPACE_DIM == 3)
          CHKERR VecSetValue(rigidBodyMotion[5], (*lo)->getPetscGlobalDofIdx(),
                             -coords[2], INSERT_VALUES);

      } else if ((*lo)->getDofCoeffIdx() == 2) {
        if (SPACE_DIM == 3) {
          CHKERR VecSetValue(rigidBodyMotion[2], (*lo)->getPetscGlobalDofIdx(),
                             1, INSERT_VALUES);
          CHKERR VecSetValue(rigidBodyMotion[4], (*lo)->getPetscGlobalDofIdx(),
                             coords[0], INSERT_VALUES);
          CHKERR VecSetValue(rigidBodyMotion[5], (*lo)->getPetscGlobalDofIdx(),
                             coords[1], INSERT_VALUES);
        }
      }
    }
  }

  for (int n = 0; n != rigidBodyMotion.size(); ++n) {
    CHKERR VecAssemblyBegin(rigidBodyMotion[n]);
    CHKERR VecAssemblyEnd(rigidBodyMotion[n]);
    CHKERR VecGhostUpdateBegin(rigidBodyMotion[n], INSERT_VALUES,
                               SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(rigidBodyMotion[n], INSERT_VALUES,
                             SCATTER_FORWARD);
  }

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto dm = simple->getDM();
  CHKERR DMCreateMatrix_MoFEM(dm, K);
  M = smartMatDuplicate(K, MAT_SHARE_NONZERO_PATTERN);

  auto calculate_stiffness_matrix = [&]() {
    MoFEMFunctionBegin;
    pipeline_mng->getDomainLhsFE().reset();

    if (SPACE_DIM == 2) {
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateInvJacForFace(invJac));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpSetInvJacH1ForFace(invJac));
    }

    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpK("U", "U", matDPtr));
    auto integration_rule = [](int, int, int approx_order) {
      return 2 * (approx_order - 1);
    };

    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
    pipeline_mng->getDomainLhsFE()->B = K;
    CHKERR MatZeroEntries(K);
    CHKERR pipeline_mng->loopFiniteElements();
    CHKERR MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    MoFEMFunctionReturn(0);
  };

  auto calculate_mass_matrix = [&]() {
    MoFEMFunctionBegin;
    pipeline_mng->getDomainLhsFE().reset();
    auto get_rho = [](const double, const double, const double) { return rho; };
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpMass("U", "U", get_rho));
    auto integration_rule = [](int, int, int approx_order) {
      return 2 * approx_order;
    };
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
    CHKERR MatZeroEntries(M);
    pipeline_mng->getDomainLhsFE()->B = M;
    CHKERR pipeline_mng->loopFiniteElements();
    CHKERR MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    MoFEMFunctionReturn(0);
  };

  CHKERR calculate_stiffness_matrix();
  CHKERR calculate_mass_matrix();

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  auto simple = mField.getInterface<Simple>();
  auto is_manager = mField.getInterface<ISManager>();
  MoFEMFunctionBegin;

  auto create_eps = [](MPI_Comm comm) {
    EPS eps;
    CHKERR EPSCreate(comm, &eps);
    return SmartPetscObj<EPS>(eps);
  };

  auto deflate_vectors = [&]() {
    MoFEMFunctionBegin;
    // Deflate vectors
    std::array<Vec, 6> deflate_vectors;
    for (int n = 0; n != 6; ++n) {
      deflate_vectors[n] = rigidBodyMotion[n];
    }
    CHKERR EPSSetDeflationSpace(ePS, 6, &deflate_vectors[0]);
    MoFEMFunctionReturn(0);
  };

  auto print_info = [&]() {
    MoFEMFunctionBegin;
    ST st;
    EPSType type;
    PetscReal tol;
    PetscInt nev, maxit, its;
    // Optional: Get some information from the solver and display it
    CHKERR EPSGetIterationNumber(ePS, &its);
    MOFEM_LOG_C("EXAMPLE", Sev::inform,
                " Number of iterations of the method: %d", its);
    CHKERR EPSGetST(ePS, &st);
    CHKERR EPSGetType(ePS, &type);
    MOFEM_LOG_C("EXAMPLE", Sev::inform, " Solution method: %s", type);
    CHKERR EPSGetDimensions(ePS, &nev, NULL, NULL);
    MOFEM_LOG_C("EXAMPLE", Sev::inform, " Number of requested eigenvalues: %d",
                nev);
    CHKERR EPSGetTolerances(ePS, &tol, &maxit);
    MOFEM_LOG_C("EXAMPLE", Sev::inform,
                " Stopping condition: tol=%.4g, maxit=%d", (double)tol, maxit);

    PetscScalar eigr, eigi;
    for (int nn = 0; nn < nev; nn++) {
      CHKERR EPSGetEigenpair(ePS, nn, &eigr, &eigi, PETSC_NULL, PETSC_NULL);
      MOFEM_LOG_C("EXAMPLE", Sev::inform,
                  " ncov = %d eigr = %.4g eigi = %.4g (inv eigr = %.4g)", nn,
                  eigr, eigi, 1. / eigr);
    }

    MoFEMFunctionReturn(0);
  };

  auto setup_eps = [&]() {
    MoFEMFunctionBegin;
    CHKERR EPSSetProblemType(ePS, EPS_GHEP);
    CHKERR EPSSetWhichEigenpairs(ePS, EPS_SMALLEST_MAGNITUDE);
    CHKERR EPSSetFromOptions(ePS);
    MoFEMFunctionReturn(0);
  };

  // Create eigensolver context
  ePS = create_eps(mField.get_comm());
  CHKERR EPSSetOperators(ePS, K, M);

  // Setup eps 
  CHKERR setup_eps();

  // Deflate vectors
  CHKERR deflate_vectors(); 

  // Solve problem
  CHKERR EPSSolve(ePS);

  // Print info
  CHKERR print_info();

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto *simple = mField.getInterface<Simple>();

  pipeline_mng->getDomainLhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  post_proc_fe->generateReferenceElementMesh();

  if (SPACE_DIM == 2) {
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    post_proc_fe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
  }

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               matGradPtr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", matGradPtr, matStrainPtr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", matStrainPtr, matStressPtr, matDPtr));
  post_proc_fe->getOpPtrVector().push_back(new OpPostProcElastic<SPACE_DIM>(
      "U", post_proc_fe->postProcMesh, post_proc_fe->mapGaussPts, matStrainPtr,
      matStressPtr));

  post_proc_fe->addFieldValuesPostProc("U");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);

  PetscInt nev;
  CHKERR EPSGetDimensions(ePS, &nev, NULL, NULL);
  PetscScalar eigr, eigi, nrm2r;
  for (int nn = 0; nn < nev; nn++) {
    CHKERR EPSGetEigenpair(ePS, nn, &eigr, &eigi, D, PETSC_NULL);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecNorm(D, NORM_2, &nrm2r);
    MOFEM_LOG_C("EXAMPLE", Sev::inform,
                " ncov = %d omega2 = %.8g omega = %.8g frequency  = %.8g", nn,
                eigr, sqrt(std::abs(eigr)), sqrt(std::abs(eigr)) / (2 * M_PI));
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR pipeline_mng->loopFiniteElements();
    post_proc_fe->writeFile("out_eig_" + boost::lexical_cast<std::string>(nn) +
                            ".h5m");
  }

  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  PetscBool test_flg = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test", &test_flg, PETSC_NULL);
  if (test_flg) {
    PetscScalar eigr, eigi;
    CHKERR EPSGetEigenpair(ePS, 0, &eigr, &eigi, PETSC_NULL, PETSC_NULL);
    constexpr double regression_value = 12579658;
    if (fabs(eigr - regression_value) > 1)
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
              "Regression test faileed; wrong eigen value.");
  }
  MoFEMFunctionReturn(0);
}
//! [Check]

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  SlepcInitialize(&argc, &argv, param_file, help);
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  LogManager::setLog("EXAMPLE");
  MOFEM_LOG_TAG("EXAMPLE", "example");

  try {

    //! [Register MoFEM discrete manager in PETSc]
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);
    //! [Register MoFEM discrete manager in PETSc

    //! [Create MoAB]
    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface
    //! [Create MoAB]

    //! [Create MoFEM]
    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database insterface
    //! [Create MoFEM]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  SlepcFinalize();
  CHKERR MoFEM::Core::Finalize();
}
