/**
 * \file mixed_poisson.cpp
 * \example mixed_poisson.cpp
 *
 * MixedPoisson intended to show how to solve mixed formulation of the Dirichlet
 * problem for the Poisson equation using error indicators and p-adaptivity
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

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

using DomainEle = PipelineManager::FaceEle2D;
using DomainEleOp = DomainEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

using OpHdivHdiv = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<3, 3>;
using OpHdivU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixDivTimesScalar<2>;
using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

inline double sqr(double x) { return x * x; }

struct MixedPoisson {

  MixedPoisson(MoFEM::Interface &m_field) : mField(m_field) {}
  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  MatrixDouble invJac;
  MatrixDouble jAC;

  Range domainEntities;
  int baseOrder;


  //! [Source function]
  static double sourceFunction(const double x, const double y, const double z) {
    return -exp(-100. * (sqr(x) + sqr(y))) *
           (400. * M_PI *
                (x * cos(M_PI * y) * sin(M_PI * x) +
                 y * cos(M_PI * x) * sin(M_PI * y)) +
            2. * (20000. * (sqr(x) + sqr(y)) - 200. - sqr(M_PI)) *
                cos(M_PI * x) * cos(M_PI * y));
  }
  //! [Source function]

  static MoFEMErrorCode getTagHandle(MoFEM::Interface &m_field,
                                     const char *name, DataType type,
                                     Tag &tag_handle) {
    MoFEMFunctionBegin;
    int int_val = 0;
    double double_val = 0;
    switch (type) {
    case MB_TYPE_INTEGER:
      CHKERR m_field.get_moab().tag_get_handle(
          name, 1, type, tag_handle, MB_TAG_CREAT | MB_TAG_SPARSE, &int_val);
      break;
    case MB_TYPE_DOUBLE:
      CHKERR m_field.get_moab().tag_get_handle(
          name, 1, type, tag_handle, MB_TAG_CREAT | MB_TAG_SPARSE, &double_val);
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
               "Wrong data type %d for tag", type);
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  struct CommonData {
    boost::shared_ptr<VectorDouble> approxVals;
    boost::shared_ptr<MatrixDouble> approxValsGrad;
    boost::shared_ptr<MatrixDouble> approxFlux;
    SmartPetscObj<Vec> petscVec;

    enum VecElements {
      ERROR_L2_NORM = 0,
      ERROR_H1_SEMINORM,
      ERROR_INDICATOR,
      TOTAL_NUMBER,
      LAST_ELEMENT
    };
  };

  boost::shared_ptr<CommonData> commonDataPtr;
  struct OpError : public DomainEleOp {
    boost::shared_ptr<CommonData> commonDataPtr;
    MoFEM::Interface &mField;
    OpError(boost::shared_ptr<CommonData> &common_data_ptr,
            MoFEM::Interface &m_field)
        : DomainEleOp("U", OPROW), commonDataPtr(common_data_ptr),
          mField(m_field) {
      std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
      doEntities[MBTRI] = doEntities[MBQUAD] = true;
    }
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };
};

//! [Run programme]
MoFEMErrorCode MixedPoisson::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR setIntegrationRules();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

//! [Read mesh]
MoFEMErrorCode MixedPoisson::readMesh() {
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();
  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode MixedPoisson::setupProblem() {
  MoFEMFunctionBegin;
  // Note that in 2D case HDIV and HCURL spaces are isomorphic, and therefore
  // only base for HCURL has been implemented in 2D. Base vectors for HDIV space
  // are be obtained after rotation of HCURL base vectors by a right angle
  CHKERR simpleInterface->addDomainField("FLUX", HCURL, DEMKOWICZ_JACOBI_BASE,
                                         1);
  // We use AINSWORTH_LEGENDRE_BASE since DEMKOWICZ_JACOBI_BASE for triangle
  // is not yet implemented for L2 space. For quads DEMKOWICZ_JACOBI_BASE and
  // AINSWORTH_LEGENDRE_BASE are construcreed in the same way
  CHKERR simpleInterface->addDomainField("U", L2, AINSWORTH_LEGENDRE_BASE, 1);

  baseOrder = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-base_order", &baseOrder,
                            PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder("FLUX", baseOrder);
  CHKERR simpleInterface->setFieldOrder("U", baseOrder - 1);
  CHKERR simpleInterface->setUp();

  CHKERR mField.get_moab().get_entities_by_dimension(0, 2, domainEntities,
                                                     false);
  Tag th_order;
  CHKERR getTagHandle(mField, "ORDER", MB_TYPE_INTEGER, th_order);
  for (auto ent : domainEntities) {
    CHKERR mField.get_moab().tag_set_data(th_order, &ent, 1, &baseOrder);
  }
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Set integration rule]
MoFEMErrorCode MixedPoisson::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule = [](int, int, int p) -> int { return 2 * p + 1; };

  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule);

  MoFEMFunctionReturn(0);
}
//! [Set integration rule]

//! [Assemble system]
MoFEMErrorCode MixedPoisson::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getOpDomainRhsPipeline().clear();
  pipeline_mng->getOpDomainLhsPipeline().clear();

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateJacForFace(jAC));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpMakeHdivFromHcurl());
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetContravariantPiolaTransformFace(jAC));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetInvJacHcurlFace(invJac));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpMakeHighOrderGeometryWeightsOnFace());

  auto beta = [](const double, const double, const double) { return 1; };
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpHdivHdiv("FLUX", "FLUX", beta));
  auto unity = []() { return 1; };
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpHdivU("FLUX", "U", unity, true));
  auto source = [&](const double x, const double y, const double z) {
    return -sourceFunction(x, y, z);
  };
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainSource("U", source));
  MoFEMFunctionReturn(0);
}
//! [Assemble system]

//! [Solve]
MoFEMErrorCode MixedPoisson::solveSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPSetUp(solver);

  auto dm = simpleInterface->getDM();
  auto D = smartCreateDMVector(dm);
  auto F = smartVectorDuplicate(D);
  CHKERR VecZeroEntries(D);

  CHKERR KSPSolve(solver, F, D);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Output results]
  MoFEMErrorCode MixedPoisson::outputResults() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  auto post_proc_fe =
      boost::make_shared<PostProcFaceOnRefinedMeshFor2D>(mField);
  post_proc_fe->generateReferenceElementMesh();

  post_proc_fe->getOpPtrVector().push_back(new OpCalculateJacForFace(jAC));
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateInvJacForFace(invJac));
  post_proc_fe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
  post_proc_fe->getOpPtrVector().push_back(
      new OpSetContravariantPiolaTransformFace(jAC));
  post_proc_fe->getOpPtrVector().push_back(new OpSetInvJacHcurlFace(invJac));

  post_proc_fe->addFieldValuesPostProc("FLUX");
  post_proc_fe->addFieldValuesPostProc("U");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();

  std::ostringstream strm;
  CHKERR post_proc_fe->writeFile(strm.str().c_str());
  MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {
  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example problem
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  LogManager::setLog("EXAMPLE");
  MOFEM_LOG_TAG("EXAMPLE", "MixedPoisson");

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
    MoFEM::Interface &m_field = core; ///< finite element database interface
    //! [Create MoFEM]

    //! [MixedPoisson]
    MixedPoisson ex(m_field);
    CHKERR ex.runProblem();
    //! [MixedPoisson]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}