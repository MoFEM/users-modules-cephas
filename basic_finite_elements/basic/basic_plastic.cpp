/**
 * \file basic_plastic.cpp
 * \example basic_plastic.cpp
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
using DomianEle = FaceElementForcesAndSourcesCoreBase;
using DomianEleOp = DomianEle::UserDataOperator;
using BoundaryEle = EdgeElementForcesAndSourcesCoreBase;
using BoundaryEleOp = BoundaryEle::UserDataOperator;

constexpr double young_modulus = 1e1;
constexpr double poisson_ratio = 0.25;
constexpr double sigmaY2 = 1;
constexpr double cn = 1;

#include <ElasticOps.hpp>
#include <PlasticOps.hpp>
using namespace OpElasticTools;
using namespace OpPlasticTools;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode setUP();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode tsSolve();
  MoFEMErrorCode postProcess();
  MoFEMErrorCode checkResults();

  MatrixDouble invJac;
  boost::shared_ptr<OpPlasticTools::CommonData> commonDataPtr;
};

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setUP();
  CHKERR createCommonData();
  CHKERR bC();
  CHKERR OPs();
  CHKERR tsSolve();
  CHKERR postProcess();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Set up problem]
MoFEMErrorCode Example::setUP() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  // Add field
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, 2);
  CHKERR simple->addDomainField("TAU", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addDomainField("EP", L2, AINSWORTH_LEGENDRE_BASE, 3);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, 2);
  constexpr int order = 1;
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("TAU", order - 1);
  CHKERR simple->setFieldOrder("EP", order - 1);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  auto get_matrial_stiffens = [&](FTensor::Ddg<double, 2, 2> &t_D) {
    MoFEMFunctionBegin;
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

  commonDataPtr = boost::make_shared<OpPlasticTools::CommonData>();
  CHKERR get_matrial_stiffens(commonDataPtr->tD);
  commonDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStrainPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->plasticSurfacePtr = boost::make_shared<VectorDouble>();
  commonDataPtr->plasticFlowPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->plasticTauPtr = boost::make_shared<VectorDouble>();
  commonDataPtr->plasticStrainPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->plasticStrainDotPtr = boost::make_shared<MatrixDouble>();

  commonDataPtr->E = young_modulus;
  commonDataPtr->mu = poisson_ratio;

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() {
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
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  Basic *basic = mField.getInterface<Basic>();

  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  basic->getOpDomainLhsPipeline().push_back(new OpSetInvJacH1ForFace(invJac));
  basic->getOpDomainLhsPipeline().push_back(
      new OpStiffnessMatrixRhs("U", "U", commonDataPtr));

  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<2, 2>("U", commonDataPtr->mGradPtr));
  basic->getOpDomainLhsPipeline().push_back(new OpCalculateScalarFieldValues(
      "TAU", commonDataPtr->plasticTauPtr, MBTRI));
  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculateTensor2SymmetricFieldValues<2>(
          "EP", commonDataPtr->plasticStrainPtr, MBTRI));
  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculateTensor2SymmetricFieldValuesDot<2>(
          "EP", commonDataPtr->plasticStrainDotPtr, MBTRI));

  basic->getOpDomainLhsPipeline().push_back(new OpStrain("U", commonDataPtr));
  basic->getOpDomainLhsPipeline().push_back(
      new OpPlasticStress("U", commonDataPtr));
  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculatePlasticSurface("U", commonDataPtr));

  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculatePlasticInternalForceLhs_dEP("U", "EP", commonDataPtr));

  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculatePlasticFlowLhs_dU("EP", "U", commonDataPtr));
  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculatePlasticFlowLhs_dEP("EP", "EP", commonDataPtr));
  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculatePlasticFlowLhs_dTAU("EP", "TAU", commonDataPtr));

  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculateContrainsLhs_dU("TAU", "U", commonDataPtr));
  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculateContrainsLhs_dEP("TAU", "EP", commonDataPtr));
  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculateContrainsLhs_dTAU("TAU", "TAU", commonDataPtr));

  auto gravity = [](double x, double y) {
    return FTensor::Tensor1<double, 2>{0., 1.};
  };
  basic->getOpDomainRhsPipeline().push_back(
      new OpBodyForceRhs("U", commonDataPtr, gravity));

  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  basic->getOpDomainRhsPipeline().push_back(new OpSetInvJacH1ForFace(invJac));

  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<2, 2>("U", commonDataPtr->mGradPtr));
  basic->getOpDomainRhsPipeline().push_back(new OpCalculateScalarFieldValues(
      "TAU", commonDataPtr->plasticTauPtr, MBTRI));
  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculateTensor2SymmetricFieldValues<2>(
          "EP", commonDataPtr->plasticStrainPtr, MBTRI));
  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculateTensor2SymmetricFieldValuesDot<2>(
          "EP", commonDataPtr->plasticStrainDotPtr, MBTRI));

  basic->getOpDomainRhsPipeline().push_back(new OpStrain("U", commonDataPtr));
  basic->getOpDomainRhsPipeline().push_back(
      new OpPlasticStress("U", commonDataPtr));
  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculatePlasticSurface("U", commonDataPtr));

  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculatePlasticFlowRhs("EP", commonDataPtr));
  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculateContrainsRhs("TAU", commonDataPtr));
  basic->getOpDomainRhsPipeline().push_back(
      new OpInternalForceRhs("U", commonDataPtr));

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR basic->setDomainRhsIntegrationRule(integration_rule);
  CHKERR basic->setDomainLhsIntegrationRule(integration_rule);

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::tsSolve() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();
  Basic *basic = mField.getInterface<Basic>();

  auto solver = basic->createTS();

  DM dm;
  CHKERR TSGetDM(solver,&dm);
  auto D = smartCreateDMDVector(dm);

  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetFromOptions(solver);
  CHKERR TSSetUp(solver);

  auto get_section_monitor = [&]() {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    PetscViewerAndFormat *vf;
    CHKERR PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                      PETSC_VIEWER_DEFAULT, &vf);
    CHKERR SNESMonitorSet(
        snes,
        (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal, void *))SNESMonitorFields,
        vf, (MoFEMErrorCode(*)(void **))PetscViewerAndFormatDestroy);
    MoFEMFunctionReturn(0);
  };

  CHKERR get_section_monitor();
  CHKERR TSSolve(solver, D);

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::postProcess() {
  MoFEMFunctionBegin;
  Basic *basic = mField.getInterface<Basic>();

  basic->getDomainLhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();


  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateInvJacForFace(invJac));
  post_proc_fe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<2, 2>("U", commonDataPtr->mGradPtr));
  post_proc_fe->getOpPtrVector().push_back(new OpStrain("U", commonDataPtr));

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<2, 2>("U", commonDataPtr->mGradPtr));
  post_proc_fe->getOpPtrVector().push_back(new OpStrain("U", commonDataPtr));
  post_proc_fe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
      "TAU", commonDataPtr->plasticTauPtr, MBTRI));
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateTensor2SymmetricFieldValues<2>(
          "EP", commonDataPtr->plasticStrainPtr, MBTRI));

  post_proc_fe->getOpPtrVector().push_back(new OpStress("U", commonDataPtr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpPostProcElastic("U", post_proc_fe->postProcMesh,
                            post_proc_fe->mapGaussPts, commonDataPtr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculatePlasticSurface("U", commonDataPtr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpPostProcPlastic("U", post_proc_fe->postProcMesh,
                            post_proc_fe->mapGaussPts, commonDataPtr));
  post_proc_fe->addFieldValuesPostProc("U");
  basic->getDomainRhsFE() = post_proc_fe;
  CHKERR basic->loopFiniteElements();

  CHKERR post_proc_fe->writeFile("out_plastic.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Check]

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

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

    //! [Load mesh]
    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();
    CHKERR simple->loadFile();
    //! [Load mesh]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
