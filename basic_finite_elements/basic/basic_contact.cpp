/**
 * \file basic_elastic.cpp
 * \example basic_elastic.cpp
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

constexpr int order = 1;
constexpr double young_modulus = 1;
constexpr double poisson_ratio = 0.25;
constexpr double cn = young_modulus;

#include <ElasticOps.hpp>
#include <ContactOps.hpp>
using namespace OpElasticTools;
using namespace OpContactTools;

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

  MatrixDouble invJac, jAc;
  boost::shared_ptr<OpContactTools::CommonData> commonDataPtr;
  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;

  Range getEntsOnMeshSkin();
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
  CHKERR simple->addDomainField("SIGMA", HCURL, AINSWORTH_LEGENDRE_BASE, 2);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, 2);
  CHKERR simple->addBoundaryField("SIGMA", H1, AINSWORTH_LEGENDRE_BASE, 2);
  CHKERR simple->setFieldOrder("U", order);

  CHKERR simple->setFieldOrder("SIGMA", 0);
  auto skin_ents = getEntsOnMeshSkin();
  CHKERR simple->setFieldOrder("SIGMA", order, &skin_ents);

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

  commonDataPtr = boost::make_shared<OpContactTools::CommonData>();
  CHKERR get_matrial_stiffens(commonDataPtr->tD);
  commonDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStrainPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactStressDivergencePtr =
      boost::make_shared<MatrixDouble>();
  commonDataPtr->contactTractionPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactDispPtr = boost::make_shared<MatrixDouble>();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  Basic *basic = mField.getInterface<Basic>();

  auto add_domain_ops = [&]() {
    basic->getOpDomainLhsPipeline().push_back(new OpCalculateJacForFace(jAc));
    basic->getOpDomainLhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    basic->getOpDomainLhsPipeline().push_back(new OpSetInvJacH1ForFace(invJac));
    basic->getOpDomainLhsPipeline().push_back(new OpMakeHdivFromHcurl());
    basic->getOpDomainLhsPipeline().push_back(
        new OpSetContravariantPiolaTransformFace(jAc));
    basic->getOpDomainLhsPipeline().push_back(new OpSetInvJacHcurlFace(invJac));
    basic->getOpDomainRhsPipeline().push_back(
        new OpInternalContactLhs("U", "SIGMA"));

    basic->getOpDomainLhsPipeline().push_back(
        new OpStiffnessMatrixLhs("U", "U", commonDataPtr));

    auto gravity = [](double x, double y) {
      return FTensor::Tensor1<double, 2>{0., -1.};
    };
    basic->getOpDomainRhsPipeline().push_back(
        new OpForceRhs("U", commonDataPtr, gravity));
    basic->getOpDomainRhsPipeline().push_back(new OpCalculateJacForFace(jAc));
    basic->getOpDomainRhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    basic->getOpDomainRhsPipeline().push_back(new OpSetInvJacH1ForFace(invJac));
    basic->getOpDomainRhsPipeline().push_back(new OpMakeHdivFromHcurl());
    basic->getOpDomainRhsPipeline().push_back(
        new OpSetContravariantPiolaTransformFace(jAc));
    basic->getOpDomainRhsPipeline().push_back(new OpSetInvJacHcurlFace(invJac));

    basic->getOpDomainRhsPipeline().push_back(
        new OpCalculateHVecTensorDivergence<2, 2>(
            "SIGMA", commonDataPtr->contactStressDivergencePtr));
    basic->getOpDomainRhsPipeline().push_back(
        new OpInternalContactRhs("U", commonDataPtr));
  };

  auto add_boundary_ops = [&]() {
    basic->getOpBoundaryLhsPipeline().push_back(
        new OpSetContrariantPiolaTransformOnEdge());
    basic->getOpBoundaryLhsPipeline().push_back(
        new OpConstrainDisp("U", commonDataPtr));
    basic->getOpBoundaryLhsPipeline().push_back(
        new OpConstrainTraction("SIGMA", commonDataPtr));
    basic->getOpBoundaryLhsPipeline().push_back(
        new OpConstrainLhs_dU("SIGMA", "U", commonDataPtr));
    basic->getOpBoundaryLhsPipeline().push_back(
        new OpConstrainLhs_dTraction("SIGMA", "SIGMA", commonDataPtr));

    basic->getOpBoundaryRhsPipeline().push_back(
        new OpSetContrariantPiolaTransformOnEdge());
    basic->getOpBoundaryLhsPipeline().push_back(
        new OpConstrainDisp("U", commonDataPtr));
    basic->getOpBoundaryLhsPipeline().push_back(
        new OpConstrainTraction("SIGMA", commonDataPtr));
    basic->getOpBoundaryLhsPipeline().push_back(
      new OpConstrainRhs("SIGMA", commonDataPtr));
  };

  add_domain_ops();
  add_boundary_ops();

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
  ISManager *is_manager = mField.getInterface<ISManager>();

  auto solver = basic->createTS();

  auto dm = simple->getDM();
  auto D = smartCreateDMDVector(dm);

  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetFromOptions(solver);
  CHKERR TSSetUp(solver);

  auto set_section_monitor = [&]() {
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

  auto create_post_process_element = [&]() {
    MoFEMFunctionBegin;
    postProcFe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
    postProcFe->generateReferenceElementMesh();
    auto postProcFe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
    postProcFe->generateReferenceElementMesh();
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    postProcFe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<2, 2>("U", commonDataPtr->mGradPtr));
    postProcFe->getOpPtrVector().push_back(new OpStrain("U", commonDataPtr));
    postProcFe->getOpPtrVector().push_back(new OpStress("U", commonDataPtr));
    postProcFe->getOpPtrVector().push_back(new OpPostProcElastic(
        "U", postProcFe->postProcMesh, postProcFe->mapGaussPts, commonDataPtr));
    postProcFe->addFieldValuesPostProc("U");
    MoFEMFunctionReturn(0);
  };

  auto scatter_create = [&](auto coeff) {
    SmartPetscObj<IS> is;
    CHKERR is_manager->isCreateProblemFieldAndRank(simple->getProblemName(),
                                                   ROW, "U", coeff, coeff, is);
    int loc_size;
    CHKERR ISGetLocalSize(is, &loc_size);
    Vec v;
    CHKERR VecCreateMPI(mField.get_comm(), loc_size, PETSC_DETERMINE, &v);
    VecScatter scatter;
    CHKERR VecScatterCreate(D, is, v, PETSC_NULL, &scatter);
    return std::make_tuple(SmartPetscObj<Vec>(v),
                           SmartPetscObj<VecScatter>(scatter));
  };

  auto set_time_monitor = [&]() {
    MoFEMFunctionBegin;
    boost::shared_ptr<Monitor> monitor_ptr(
        new Monitor(dm, postProcFe, uXScatter, uYScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  CHKERR set_section_monitor();
  CHKERR create_post_process_element();
  uXScatter = scatter_create(0);
  uYScatter = scatter_create(1);
  CHKERR set_time_monitor();

  CHKERR TSSolve(solver, D);

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::postProcess() { return 0; }
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() { return 0; }
//! [Check]

Range Example::getEntsOnMeshSkin() {
  Range faces;
  CHKERR mField.get_moab().get_entities_by_type(0, MBTRI, faces);
  Skinner skin(&mField.get_moab());
  Range skin_edges;
  CHKERR skin.find_skin(0, faces, false, skin_edges);
  Range skin_verts;
  CHKERR mField.get_moab().get_connectivity(skin_edges, skin_verts, true);
  skin_edges.merge(skin_verts);
  return skin_edges;
};

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
    CHKERR simple->loadFile("");
    //! [Load mesh]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
