/**
 * \file lesson4_elastic.cpp
 * \example lesson4_elastic.cpp
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

using namespace MoFEM;

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;

constexpr int SPACE_DIM = 2; //< Space dimension of problem, mesh

using OpK = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;
using OpInternalForce =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
        GAUSS>::OpGradTimesSymmetricTensor<1, SPACE_DIM, SPACE_DIM>;

constexpr double young_modulus = 1;
constexpr double poisson_ratio = 0.25;

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
  template <> MoFEMErrorCode createCommonData<2>();
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
  boost::shared_ptr<MatrixDouble> bodyForceMatPtr;
};

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
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);
  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

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

  auto set_body_force = [&]() {
    MoFEMFunctionBegin;
    auto t_force = getFTensor1FromMat<2, 0>(*bodyForceMatPtr);
    t_force(0) = 0;
    t_force(1) = -1;
    MoFEMFunctionReturn(0);
  };

  matGradPtr = boost::make_shared<MatrixDouble>();
  matStrainPtr = boost::make_shared<MatrixDouble>();
  matStressPtr = boost::make_shared<MatrixDouble>();
  matDPtr = boost::make_shared<MatrixDouble>();
  bodyForceMatPtr = boost::make_shared<MatrixDouble>();

  matDPtr->resize(9, 1);
  bodyForceMatPtr->resize(2, 1);

  CHKERR set_matrial_stiffens();
  CHKERR set_body_force();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

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
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetInvJacH1ForFace(invJac));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpK("U", "U", matDPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpBodyForce("U", bodyForceMatPtr));

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * (approx_order - 1);
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPSetUp(solver);

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);
  auto F = smartVectorDuplicate(D);

  CHKERR KSPSolve(solver, F, D);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

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
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_elastic.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() {
  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  MoFEMFunctionBegin;
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getDomainLhsFE().reset();

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetInvJacH1ForFace(invJac));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               matGradPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", matGradPtr, matStrainPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", matStrainPtr, matStressPtr, matDPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpInternalForce("U", matStressPtr));
  (*bodyForceMatPtr) *= -1;
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpBodyForce("U", bodyForceMatPtr));

  auto integration_rule = [](int, int, int p_data) { return 2 * (p_data-1); };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);

  auto dm = simple->getDM();
  auto res = smartCreateDMVector(dm);
  pipeline_mng->getDomainRhsFE()->ksp_f = res;

  CHKERR VecZeroEntries(res);
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR VecGhostUpdateBegin(res, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateEnd(res, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecAssemblyBegin(res);
  CHKERR VecAssemblyEnd(res);

  double nrm2;
  CHKERR VecNorm(res, NORM_2, &nrm2);
  PetscPrintf(PETSC_COMM_WORLD, "residual = %3.4e\n", nrm2);
  // constexpr double eps = 1e-8;
  // if (nrm2 > eps)
  //   SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "Residual is not
  //   zero");

  MoFEMFunctionReturn(0);
}
//! [Check]

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
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

  CHKERR MoFEM::Core::Finalize();
}
