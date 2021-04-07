/**
 * \file mix-poisson.cpp
 * \example mix-poisson.cpp
 *
 * Using PipelineManager interface calculate the divergence of base functions,
 * and integral of flux on the boundary. Since the h-div space is used, volume
 * integral and boundary integral should give the same result.
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

using OpQQ = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<3, 3, 1>;
using OpDivQU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpMixDivTimesU<1, 2>;
using OpUDivQ = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesScalarField<1>;

using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  MatrixDouble invJac;
  MatrixDouble jAC;

  Range domainEntities;
  double errorEstimator;

  int baseOrder;
  int iterNum;

  //! [Analytical function]
  static double analyticalFunction(const double x, const double y,
                                   const double z) {
    return cos(M_PI * x) * cos(M_PI * y);
  }
  //! [Analytical function]

  //! [Analytical function gradient]
  static VectorDouble analyticalFunctionGrad(const double x, const double y,
                                             const double z) {
    VectorDouble res;
    res.resize(2);
    res.clear();
    res[0] = -M_PI * sin(M_PI * x) * cos(M_PI * y);
    res[1] = -M_PI * cos(M_PI * x) * sin(M_PI * y);
    return res;
  }
  //! [Analytical function gradient]

  //! [Source function]
  static double sourceFunction(const double x, const double y, const double z) {
    return 2. * M_PI * M_PI * cos(M_PI * x) * cos(M_PI * y);
  }
  //! [Source function]

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode solveRefineLoop();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkError();

  struct CommonData {
    boost::shared_ptr<VectorDouble> approxVals;
    boost::shared_ptr<MatrixDouble> approxGradVals;
    boost::shared_ptr<MatrixDouble> fluxVals;
    SmartPetscObj<Vec> errorL2Norm;
    SmartPetscObj<Vec> errorH1SemiNorm;
    SmartPetscObj<Vec> errorEstimator;
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
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR setIntegrationRules();
  CHKERR createCommonData();
  CHKERR assembleSystem();
  CHKERR solveRefineLoop();
  CHKERR outputResults();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  // Add field
  CHKERR simpleInterface->addDomainField("FLUX", HCURL, DEMKOWICZ_JACOBI_BASE,
                                         1);
  // We using AINSWORTH_LEGENDRE_BASE because DEMKOWICZ_JACOBI_BASE for triangle
  // and tet is not yet implemented for L2 space. DEMKOWICZ_JACOBI_BASE and
  // AINSWORTH_LEGENDRE_BASE are construcreed in the same way on quad.
  CHKERR simpleInterface->addDomainField("U", L2, AINSWORTH_LEGENDRE_BASE, 1);

  iterNum = 1;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-iter", &iterNum, PETSC_NULL);

  baseOrder = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &baseOrder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder("FLUX", baseOrder);
  CHKERR simpleInterface->setFieldOrder("U", baseOrder - 1);
  CHKERR simpleInterface->setUp();

  CHKERR mField.get_moab().get_entities_by_dimension(0, 2, domainEntities,
                                                     false);
  Tag th_order;
  int def_order = 2;
  CHKERR mField.get_moab().tag_get_handle("ORDER", 1, MB_TYPE_INTEGER, th_order,
                                          MB_TAG_CREAT | MB_TAG_SPARSE,
                                          &def_order);
  for (Range::iterator fit = domainEntities.begin();
       fit != domainEntities.end(); fit++) {
    CHKERR mField.get_moab().tag_set_data(th_order, &*fit, 1, &baseOrder);
  }

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Set integration rule]
MoFEMErrorCode Example::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule = [](int, int, int p) -> int { return 2 * p + 1; };

  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule);

  MoFEMFunctionReturn(0);
}
//! [Set integration rule]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;
  commonDataPtr = boost::make_shared<CommonData>();
  commonDataPtr->errorL2Norm = createSmartVectorMPI(
      mField.get_comm(), (!mField.get_comm_rank()) ? 1 : 0, 1);
  commonDataPtr->errorH1SemiNorm = createSmartVectorMPI(
      mField.get_comm(), (!mField.get_comm_rank()) ? 1 : 0, 1);
  commonDataPtr->errorEstimator = createSmartVectorMPI(
      mField.get_comm(), (!mField.get_comm_rank()) ? 1 : 0, 1);
  commonDataPtr->approxVals = boost::make_shared<VectorDouble>();
  commonDataPtr->approxGradVals = boost::make_shared<MatrixDouble>();
  commonDataPtr->fluxVals = boost::make_shared<MatrixDouble>();
  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
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
  auto source = [&](const double x, const double y, const double z) {
    return -sourceFunction(x, y, z);
  };

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpHdivU("FLUX", "U", unity, true));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainSource("U", source));

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
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

//! [Refinement loop]
MoFEMErrorCode Example::solveRefineLoop() {
  MoFEMFunctionBegin;

  Tag th_error_estimator;
  double def_err = 0.0;
  CHKERR mField.get_moab().tag_get_handle(
      "ERROR_ESTIMATOR", 1, MB_TYPE_DOUBLE, th_error_estimator,
      MB_TAG_CREAT | MB_TAG_SPARSE, &def_err);

  Tag th_order;
  int def_order = 2;
  CHKERR mField.get_moab().tag_get_handle("ORDER", 1, MB_TYPE_INTEGER, th_order,
                                          MB_TAG_CREAT | MB_TAG_SPARSE,
                                          &def_order);

  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR checkError();

  for (int ii = 0; ii < iterNum; ii++) {

    std::vector<Range> refinement_levels;
    refinement_levels.resize(iterNum + 1);

    for (Range::iterator fit = domainEntities.begin();
         fit != domainEntities.end(); fit++) {
      double err_est = 0;
      CHKERR mField.get_moab().tag_get_data(th_error_estimator, &*fit, 1,
                                            &err_est);
      int order, new_order;
      CHKERR mField.get_moab().tag_get_data(th_order, &*fit, 1, &order);
      new_order = order + 1;
      Range refined_ents;
      if (err_est > 0.5 * errorEstimator / domainEntities.size()) {
        refined_ents.insert(*fit);
        Range adj;
        CHKERR mField.get_moab().get_adjacencies(&*fit, 1, 1, false, adj,
                                                 moab::Interface::UNION);
        refined_ents.merge(adj);
        refinement_levels[new_order - baseOrder].merge(refined_ents);

        CHKERR mField.get_moab().tag_set_data(th_order, &*fit, 1, &new_order);
      }
    }

    CHKERR mField.set_field_order(0, MBEDGE, "FLUX", baseOrder +1);
    //CHKERR mField.set_field_order(0, MBTRI, "FLUX", baseOrder + 1);
    CHKERR mField.set_field_order(0, MBQUAD, "FLUX", baseOrder + 1);

    //CHKERR mField.set_field_order(0, MBTRI, "U", baseOrder);
    CHKERR mField.set_field_order(0, MBQUAD, "U", baseOrder);

    // for (int ll = 1; ll < refinement_levels.size(); ll++) {
    //   CHKERR mField.set_field_order(refinement_levels[ll], "FLUX",
    //                                 baseOrder + ll);
    //   CHKERR mField.set_field_order(refinement_levels[ll], "U",
    //                                 baseOrder + ll - 1);
    // }

    // CHKERR simpleInterface->reSetUp();
    CHKERR mField.build_fields();

    CHKERR mField.build_finite_elements();
    CHKERR mField.build_adjacencies(simpleInterface->getBitRefLevel());

    // CHKERR simpleInterface->buildFiniteElements();
    CHKERR simpleInterface->buildProblem();

    CHKERR assembleSystem();
    CHKERR solveSystem();
    CHKERR checkError();
  }

  MoFEMFunctionReturn(0);
}
//! [Refinement loop]

//! [Output results]
MoFEMErrorCode Example::outputResults() {
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
  CHKERR post_proc_fe->writeFile("out_mix_poisson.h5m");
  MoFEMFunctionReturn(0);
}
//! [Output results]

//! [Check results]
// MoFEMErrorCode Example::checkError() {
//   MoFEMFunctionBegin;
//   auto *pipeline_mng = mField.getInterface<PipelineManager>();
//   auto *simple = mField.getInterface<Simple>();

//   pipeline_mng->getDomainLhsFE().reset();
//   pipeline_mng->getDomainRhsFE() = boost::make_shared<DomainEle>(mField);

//   auto dm = simple->getDM();
//   auto F = smartCreateDMVector(dm);
//   pipeline_mng->getDomainRhsFE()->f = F;

//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpCalculateJacForFace(jAC));
//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpCalculateInvJacForFace(invJac));
//   pipeline_mng->getOpDomainRhsPipeline().push_back(new
//   OpMakeHdivFromHcurl()); pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpSetContravariantPiolaTransformFace(jAC));
//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpSetInvJacHcurlFace(invJac));
//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpMakeHighOrderGeometryWeightsOnFace());

//   auto res_source = [&](const double x, const double y, const double z) {
//     return -sourceFunction(x, y, z);
//   };
//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpDomainSource("U", res_source));

//   auto q_ptr = boost::make_shared<MatrixDouble>();
//   auto div_ptr = boost::make_shared<VectorDouble>();
//   auto u_ptr = boost::make_shared<VectorDouble>();

//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpCalculateScalarFieldValues("U", u_ptr));
//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpCalculateHdivVectorField<3>("FLUX", q_ptr));
//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpCalculateHdivVectorDivergence<3, 2>("FLUX", div_ptr));

//   auto one = [](double, double, double) { return 1.; };
//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpQQ("FLUX", q_ptr, one));
//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpDivQU("FLUX", u_ptr, -1));
//   pipeline_mng->getOpDomainRhsPipeline().push_back(
//       new OpUDivQ("U", div_ptr, -1));

//   CHKERR setIntegrationRules();
//   CHKERR pipeline_mng->loopFiniteElements();
//   CHKERR VecAssemblyBegin(F);
//   CHKERR VecAssemblyEnd(F);

//   double nrm2;
//   CHKERR VecNorm(F, NORM_2, &nrm2);

//   MOFEM_LOG("WORLD", Sev::inform) << "Residual norm " << nrm2;

//   constexpr double eps = 1e-8;
//   if(std::abs(nrm2) < eps)
//     nrm2 = 0;

//   if (nrm2 != 0)
//     SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "Nonzero residual");

//   MoFEMFunctionReturn(0);
// }

MoFEMErrorCode Example::checkError() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getOpDomainRhsPipeline().clear();

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateJacForFace(jAC));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpMakeHdivFromHcurl());
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetContravariantPiolaTransformFace(jAC));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetInvJacHcurlFace(invJac));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetInvJacL2ForFace(invJac));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpMakeHighOrderGeometryWeightsOnFace());

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("U", commonDataPtr->approxVals));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldGradient<2>("U",
                                            commonDataPtr->approxGradVals));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateHdivVectorField<3>("FLUX", commonDataPtr->fluxVals));

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpError(commonDataPtr, mField));

  CHKERR VecZeroEntries(commonDataPtr->errorL2Norm);
  CHKERR VecZeroEntries(commonDataPtr->errorH1SemiNorm);
  CHKERR VecZeroEntries(commonDataPtr->errorEstimator);
  CHKERR pipeline_mng->loopFiniteElements();

  {
    CHKERR VecAssemblyBegin(commonDataPtr->errorL2Norm);
    CHKERR VecAssemblyEnd(commonDataPtr->errorL2Norm);
    const double *array;
    CHKERR VecGetArrayRead(commonDataPtr->errorL2Norm, &array);
    if (mField.get_comm_rank() == 0)
      PetscPrintf(PETSC_COMM_SELF, "Error L2 norm: %6.4e\n", sqrt(array[0]));
    CHKERR VecRestoreArrayRead(commonDataPtr->errorL2Norm, &array);
  }
  {
    CHKERR VecAssemblyBegin(commonDataPtr->errorH1SemiNorm);
    CHKERR VecAssemblyEnd(commonDataPtr->errorH1SemiNorm);
    const double *array;
    CHKERR VecGetArrayRead(commonDataPtr->errorH1SemiNorm, &array);
    if (mField.get_comm_rank() == 0)
      PetscPrintf(PETSC_COMM_SELF, "Error H1 seminorm: %6.4e\n",
                  sqrt(array[0]));
    CHKERR VecRestoreArrayRead(commonDataPtr->errorH1SemiNorm, &array);
  }
  {
    CHKERR VecAssemblyBegin(commonDataPtr->errorEstimator);
    CHKERR VecAssemblyEnd(commonDataPtr->errorEstimator);
    const double *array;
    CHKERR VecGetArrayRead(commonDataPtr->errorEstimator, &array);
    if (mField.get_comm_rank() == 0)
      PetscPrintf(PETSC_COMM_SELF, "Error estimator: %6.4e\n", sqrt(array[0]));
    errorEstimator = array[0];
    CHKERR VecRestoreArrayRead(commonDataPtr->errorEstimator, &array);
  }
  CHKERR mField.get_moab().write_file("error.h5m");
  MoFEMFunctionReturn(0);
}
//! [Check results]

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
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

  CHKERR MoFEM::Core::Finalize();
}

MoFEMErrorCode Example::OpError::doWork(int side, EntityType type,
                                        EntData &data) {
  MoFEMFunctionBegin;

  auto sqr = [](double x) { return x * x; };

  if (const size_t nb_dofs = data.getIndices().size()) {

    const int nb_integration_pts = getGaussPts().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_val = getFTensor0FromVec(*(commonDataPtr->approxVals));
    auto t_val_grad = getFTensor1FromMat<2>(*(commonDataPtr->approxGradVals));
    auto t_flux = getFTensor1FromMat<3>(*(commonDataPtr->fluxVals));
    auto t_coords = getFTensor1CoordsAtGaussPts();

    FTensor::Index<'i', 3> i;
    const double volume = getMeasure();

    auto t_row_base = data.getFTensor0N();
    double error_l2_norm = 0;
    double error_h1_seminorm = 0;
    double error_estimator = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {

      const double alpha = t_w * volume;
      double diff = t_val - Example::analyticalFunction(
                                t_coords(0), t_coords(1), t_coords(2));
      error_l2_norm += alpha * sqr(diff);

      VectorDouble vec = Example::analyticalFunctionGrad(
          t_coords(0), t_coords(1), t_coords(2));

      error_h1_seminorm +=
          alpha * (sqr(t_val_grad(0) - vec(0)) + sqr(t_val_grad(1) - vec(1)));

      error_estimator += alpha * (sqr(t_val_grad(0) - t_flux(0)) +
                                  sqr(t_val_grad(1) - t_flux(1)));

      ++t_w;
      ++t_val;
      ++t_val_grad;
      ++t_flux;
      ++t_coords;
    }
    const EntityHandle ent = getFEEntityHandle();
    Tag th_error_l2, th_error_h1, th_error_estimator;
    double def_err = 0.0;
    CHKERR mField.get_moab().tag_get_handle(
        "ERROR_L2_NORM", 1, MB_TYPE_DOUBLE, th_error_l2,
        MB_TAG_CREAT | MB_TAG_SPARSE, &def_err);
    CHKERR mField.get_moab().tag_set_data(th_error_l2, &ent, 1, &error_l2_norm);

    CHKERR mField.get_moab().tag_get_handle(
        "ERROR_H1_SEMINORM", 1, MB_TYPE_DOUBLE, th_error_h1,
        MB_TAG_CREAT | MB_TAG_SPARSE, &def_err);
    CHKERR mField.get_moab().tag_set_data(th_error_h1, &ent, 1,
                                          &error_h1_seminorm);

    CHKERR mField.get_moab().tag_get_handle(
        "ERROR_ESTIMATOR", 1, MB_TYPE_DOUBLE, th_error_estimator,
        MB_TAG_CREAT | MB_TAG_SPARSE, &def_err);
    CHKERR mField.get_moab().tag_set_data(th_error_estimator, &ent, 1,
                                          &error_estimator);

    const int index = 0;
    CHKERR VecSetValue(commonDataPtr->errorL2Norm, index, error_l2_norm,
                       ADD_VALUES);
    CHKERR VecSetValue(commonDataPtr->errorH1SemiNorm, index, error_h1_seminorm,
                       ADD_VALUES);
    CHKERR VecSetValue(commonDataPtr->errorEstimator, index, error_estimator,
                       ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}
