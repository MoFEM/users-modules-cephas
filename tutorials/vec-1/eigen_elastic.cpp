/**
 * \file eigen_elastic.cpp
 * \example eigen_elastic.cpp
 *
 * Plane stress elastic problem
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

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;

constexpr int SPACE_DIM = 2; //< Space dimension of problem, mesh

using OpK = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;

constexpr double young_modulus = 1;
constexpr double poisson_ratio = 0.25;
constexpr double rho = 1;

#include <OpPostProcElastic.hpp>
using namespace Tutorial;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  template <int DIM = SPACE_DIM> MoFEMErrorCode createCommonData();
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
};

//! [Create common data]
template <> MoFEMErrorCode Example::createCommonData<2>() {
  MoFEMFunctionBegin;

  auto set_matrial_stiffens = [&]() {
    MoFEMFunctionBegin;

    auto t_D = getFTensor4DdgFromMat<2, 2, 0>(*matDPtr);

    FTensor::Index<'i', 2> i;
    FTensor::Index<'j', 2> j;
    FTensor::Index<'k', 2> k;
    FTensor::Index<'l', 2> l;
    t_D(i, j, k, l) = 0;

    constexpr double c = young_modulus / (1 - poisson_ratio * poisson_ratio);
    constexpr double o = poisson_ratio * c;

    t_D(0, 0, 0, 0) = c;
    t_D(0, 0, 1, 1) = o;

    t_D(1, 1, 0, 0) = o;
    t_D(1, 1, 1, 1) = c;

    t_D(0, 1, 0, 1) = (1 - poisson_ratio) * c;

    MoFEMFunctionReturn(0);
  };

  matGradPtr = boost::make_shared<MatrixDouble>();
  matStrainPtr = boost::make_shared<MatrixDouble>();
  matStressPtr = boost::make_shared<MatrixDouble>();
  matDPtr = boost::make_shared<MatrixDouble>();

  matDPtr->resize(9, 1);

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
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();
  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  // Add field
  CHKERR simple->addDomainField("U", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE,
                                SPACE_DIM);
  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;

  auto fix_disp = [&](const std::string blockset_name) {
    Range fix_ents;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blockset_name.length(), blockset_name) ==
          0) {
        CHKERR mField.get_moab().get_entities_by_handle(it->meshset, fix_ents,
                                                        true);
      }
    }
    return fix_ents;
  };

  auto remove_ents = [&](const Range &&ents, const bool fix_x,
                         const bool fix_y) {
    auto prb_mng = mField.getInterface<ProblemsManager>();
    auto simple = mField.getInterface<Simple>();
    MoFEMFunctionBegin;
    Range verts;
    CHKERR mField.get_moab().get_connectivity(ents, verts, true);
    verts.merge(ents);
    if (SPACE_DIM == 3) {
      Range adj;
      CHKERR mField.get_moab().get_adjacencies(ents, 1, false, adj,
                                               moab::Interface::UNION);
      verts.merge(adj);
    };
    const int lo_coeff = fix_x ? 0 : 1;
    const int hi_coeff = fix_y ? 1 : 0;
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "U", verts,
                                         lo_coeff, hi_coeff);
    MoFEMFunctionReturn(0);
  };

  CHKERR remove_ents(fix_disp("FIX_X"), true, false);
  CHKERR remove_ents(fix_disp("FIX_Y"), false, true);
  CHKERR remove_ents(fix_disp("FIX_ALL"), true, true);

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
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetInvJacH1ForFace(invJac));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpK("U", "U", matDPtr));
    auto integration_rule = [](int, int, int approx_order) {
      return 2 * (approx_order - 1);
    };
    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
    pipeline_mng->getDomainLhsFE()->B = K;
    CHKERR pipeline_mng->loopFiniteElements();
    CHKERR MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    MoFEMFunctionReturn(0);
  };

  auto salculate_matrix = [&]() {
    MoFEMFunctionBegin;
    pipeline_mng->getDomainLhsFE().reset();
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetInvJacH1ForFace(invJac));
    auto get_rho = [](const double, const double, const double) { return rho; };
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpMass("U", "U", get_rho));
    auto integration_rule = [](int, int, int approx_order) {
      return 2 * (approx_order - 1);
    };
    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
    pipeline_mng->getDomainLhsFE()->B = M;
    CHKERR pipeline_mng->loopFiniteElements();
    CHKERR MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    MoFEMFunctionReturn(0);
  };

  CHKERR calculate_stiffness_matrix();
  CHKERR salculate_matrix();

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;

  auto createEPS = [](MPI_Comm comm) {
    EPS eps;
    ierr = EPSCreate(comm, &eps);
    CHKERRABORT(comm, ierr);
    return SmartPetscObj<EPS>(eps);
  };

  // Create eigensolver context
  ePS = createEPS(mField.get_comm());

  ST st;
  EPSType type;
  PetscReal tol;
  PetscInt nev, maxit, its;

  // Set operators. In this case, it is a generalized eigenvalue problem
  CHKERR EPSSetOperators(ePS, M, K);

  // Set solver parameters at runtime
  CHKERR EPSSetFromOptions(ePS);

  // Optional: Get some information from the solver and display it
  CHKERR EPSSolve(ePS);

  // Optional: Get some information from the solver and display it
  CHKERR EPSGetIterationNumber(ePS, &its);
  MOFEM_LOG_C("WORLD", Sev::inform, " Number of iterations of the method: %D",
              its);
  CHKERR EPSGetST(ePS, &st);
  CHKERR EPSGetType(ePS, &type);
  MOFEM_LOG_C("WORLD", Sev::inform, " Solution method: %s", type);
  CHKERR EPSGetDimensions(ePS, &nev, NULL, NULL);
  MOFEM_LOG_C("WORLD", Sev::inform, " Number of requested eigenvalues: %D",
              nev);
  CHKERR EPSGetTolerances(ePS, &tol, &maxit);
  MOFEM_LOG_C("WORLD", Sev::inform, " Stopping condition: tol=%.4g, maxit=%D",
              (double)tol, maxit);

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto *simple = mField.getInterface<Simple>();

  pipeline_mng->getDomainLhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateInvJacForFace(invJac));
  post_proc_fe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
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
    MOFEM_LOG_C(
        "WORLD", Sev::inform,
        " ncov = %D eigr = %.4g eigi = %.4g (inv eigr = %.4g) nrm2r = %.4g", nn,
        eigr, eigi, 1. / eigr, nrm2r);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR pipeline_mng->loopFiniteElements();
    CHKERR post_proc_fe->writeFile(
        "out_eig_" + boost::lexical_cast<std::string>(nn) + ".h5m");
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
    constexpr double regression_value = 5.862;
    if (fabs(eigr - regression_value) > 1e-2)
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
