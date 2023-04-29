/**
 * \file mixed_poisson.cpp
 * \example mixed_poisson.cpp
 *
 * MixedPoisson intended to show how to solve mixed formulation of the Dirichlet
 * problem for the Poisson equation using error indicators and p-adaptivity
 *
 */

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

using DomainEle = PipelineManager::FaceEle;
using DomainEleOp = DomainEle::UserDataOperator;
using EntData = EntitiesFieldData::EntData;

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

  Range domainEntities;
  double errorIndicatorIntegral;
  int totalElementNumber;
  int baseOrder;
  int refIterNum;

  //! [Analytical function]
  static double analyticalFunction(const double x, const double y,
                                   const double z) {
    return exp(-100. * (sqr(x) + sqr(y))) * cos(M_PI * x) * cos(M_PI * y);
  }
  //! [Analytical function]

  //! [Analytical function gradient]
  static VectorDouble analyticalFunctionGrad(const double x, const double y,
                                             const double z) {
    VectorDouble res;
    res.resize(2);
    res[0] = -exp(-100. * (sqr(x) + sqr(y))) *
             (200. * x * cos(M_PI * x) + M_PI * sin(M_PI * x)) * cos(M_PI * y);
    res[1] = -exp(-100. * (sqr(x) + sqr(y))) *
             (200. * y * cos(M_PI * y) + M_PI * sin(M_PI * y)) * cos(M_PI * x);
    return res;
  }
  //! [Analytical function gradient]

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
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode checkError(int iter_num = 0);
  MoFEMErrorCode refineOrder();
  MoFEMErrorCode solveRefineLoop();
  MoFEMErrorCode outputResults(int iter_num = 0);

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
  CHKERR createCommonData();
  CHKERR solveRefineLoop();
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

  refIterNum = 0;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-ref_iter_num", &refIterNum,
                            PETSC_NULL);
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

//! [Create common data]
MoFEMErrorCode MixedPoisson::createCommonData() {
  MoFEMFunctionBegin;
  commonDataPtr = boost::make_shared<CommonData>();
  PetscInt ghosts[4] = {0, 1, 2, 3};
  if (!mField.get_comm_rank())
    commonDataPtr->petscVec =
        createGhostVector(mField.get_comm(), 4, 4, 0, ghosts);
  else
    commonDataPtr->petscVec =
        createGhostVector(mField.get_comm(), 0, 4, 4, ghosts);
  commonDataPtr->approxVals = boost::make_shared<VectorDouble>();
  commonDataPtr->approxValsGrad = boost::make_shared<MatrixDouble>();
  commonDataPtr->approxFlux = boost::make_shared<MatrixDouble>();
  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Assemble system]
MoFEMErrorCode MixedPoisson::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getOpDomainRhsPipeline().clear();
  pipeline_mng->getOpDomainLhsPipeline().clear();

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateHOJacForFace(jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpMakeHdivFromHcurl());
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetInvJacHcurlFace(inv_jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpSetHOWeightsOnFace());

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
  auto D = createDMVector(dm);
  auto F = vectorDuplicate(D);
  CHKERR VecZeroEntries(D);

  CHKERR KSPSolve(solver, F, D);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Refine]
MoFEMErrorCode MixedPoisson::refineOrder() {
  MoFEMFunctionBegin;
  Tag th_error_ind, th_order;
  CHKERR getTagHandle(mField, "ERROR_INDICATOR", MB_TYPE_DOUBLE, th_error_ind);
  CHKERR getTagHandle(mField, "ORDER", MB_TYPE_INTEGER, th_order);

  std::vector<Range> refinement_levels;
  refinement_levels.resize(refIterNum + 1);
  for (auto ent : domainEntities) {
    double err_indic = 0;
    CHKERR mField.get_moab().tag_get_data(th_error_ind, &ent, 1, &err_indic);
    int order, new_order;
    CHKERR mField.get_moab().tag_get_data(th_order, &ent, 1, &order);
    new_order = order + 1;
    Range refined_ents;
    if (err_indic > errorIndicatorIntegral / totalElementNumber) {
      refined_ents.insert(ent);
      Range adj;
      CHKERR mField.get_moab().get_adjacencies(&ent, 1, 1, false, adj,
                                               moab::Interface::UNION);
      refined_ents.merge(adj);
      refinement_levels[new_order - baseOrder].merge(refined_ents);
      CHKERR mField.get_moab().tag_set_data(th_order, &ent, 1, &new_order);
    }
  }

  for (int ll = 1; ll < refinement_levels.size(); ll++) {
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
        refinement_levels[ll]);
    CHKERR mField.set_field_order(refinement_levels[ll], "FLUX",
                                  baseOrder + ll);
    CHKERR mField.set_field_order(refinement_levels[ll], "U",
                                  baseOrder + ll - 1);
  }

  CHKERR mField.getInterface<CommInterface>()->synchroniseFieldEntities("FLUX");
  CHKERR mField.getInterface<CommInterface>()->synchroniseFieldEntities("U");
  CHKERR mField.build_fields();
  CHKERR mField.build_finite_elements();
  CHKERR mField.build_adjacencies(simpleInterface->getBitRefLevel());
  mField.getInterface<ProblemsManager>()->buildProblemFromFields = PETSC_TRUE;
  CHKERR DMSetUp_MoFEM(simpleInterface->getDM());
  MoFEMFunctionReturn(0);
}
//! [Refine]

//! [Solve and refine loop]
MoFEMErrorCode MixedPoisson::solveRefineLoop() {
  MoFEMFunctionBegin;
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR checkError();
  CHKERR outputResults();

  for (int ii = 0; ii < refIterNum; ii++) {
    CHKERR refineOrder();

    CHKERR assembleSystem();
    CHKERR solveSystem();
    CHKERR checkError(ii + 1);
    CHKERR outputResults(ii + 1);
  }
  MoFEMFunctionReturn(0);
}
//! [Solve and refine loop]

//! [Check error]
MoFEMErrorCode MixedPoisson::checkError(int iter_num) {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getOpDomainRhsPipeline().clear();

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateHOJacForFace(jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpMakeHdivFromHcurl());
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetInvJacHcurlFace(inv_jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetInvJacL2ForFace(inv_jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpSetHOWeightsOnFace());

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("U", commonDataPtr->approxVals));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldGradient<2>("U",
                                            commonDataPtr->approxValsGrad));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateHVecVectorField<3>("FLUX", commonDataPtr->approxFlux));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpError(commonDataPtr, mField));

  CHKERR VecZeroEntries(commonDataPtr->petscVec);
  CHKERR VecGhostUpdateBegin(commonDataPtr->petscVec, INSERT_VALUES,
                             SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(commonDataPtr->petscVec, INSERT_VALUES,
                           SCATTER_FORWARD);
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR VecAssemblyBegin(commonDataPtr->petscVec);
  CHKERR VecAssemblyEnd(commonDataPtr->petscVec);
  CHKERR VecGhostUpdateBegin(commonDataPtr->petscVec, ADD_VALUES,
                             SCATTER_REVERSE);
  CHKERR VecGhostUpdateEnd(commonDataPtr->petscVec, ADD_VALUES,
                           SCATTER_REVERSE);
  CHKERR VecGhostUpdateBegin(commonDataPtr->petscVec, INSERT_VALUES,
                             SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(commonDataPtr->petscVec, INSERT_VALUES,
                           SCATTER_FORWARD);
  const double *array;
  CHKERR VecGetArrayRead(commonDataPtr->petscVec, &array);
  MOFEM_LOG("EXAMPLE", Sev::inform)
      << "Global error L2 norm: " << std::sqrt(array[0]);
  MOFEM_LOG("EXAMPLE", Sev::inform)
      << "Global error H1 seminorm: " << std::sqrt(array[1]);
  MOFEM_LOG("EXAMPLE", Sev::inform)
      << "Global error indicator: " << std::sqrt(array[2]);
  MOFEM_LOG("EXAMPLE", Sev::inform)
      << "Total number of elements: " << (int)array[3];

  errorIndicatorIntegral = array[2];
  totalElementNumber = (int)array[3];
  CHKERR VecRestoreArrayRead(commonDataPtr->petscVec, &array);

  std::vector<Tag> tag_handles;
  tag_handles.resize(4);
  CHKERR getTagHandle(mField, "ERROR_L2_NORM", MB_TYPE_DOUBLE, tag_handles[0]);
  CHKERR getTagHandle(mField, "ERROR_H1_SEMINORM", MB_TYPE_DOUBLE,
                      tag_handles[1]);
  CHKERR getTagHandle(mField, "ERROR_INDICATOR", MB_TYPE_DOUBLE,
                      tag_handles[2]);
  CHKERR getTagHandle(mField, "ORDER", MB_TYPE_INTEGER, tag_handles[3]);

  ParallelComm *pcomm =
      ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
  if (pcomm == NULL)
    SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "Communicator not set");

  tag_handles.push_back(pcomm->part_tag());
  std::ostringstream strm;
  strm << "error_" << iter_num << ".h5m";
  CHKERR mField.get_moab().write_file(strm.str().c_str(), "MOAB",
                                      "PARALLEL=WRITE_PART", 0, 0,
                                      tag_handles.data(), tag_handles.size());
  MoFEMFunctionReturn(0);
}
//! [Check error]

//! [Output results]
MoFEMErrorCode MixedPoisson::outputResults(int iter_num) {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();

  using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

  auto post_proc_fe =
      boost::make_shared<PostProcEle>(mField);

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateHOJacForFace(jac_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
  post_proc_fe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
  post_proc_fe->getOpPtrVector().push_back(
      new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpSetInvJacHcurlFace(inv_jac_ptr));

  auto u_ptr = boost::make_shared<VectorDouble>();
  auto flux_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("U", u_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateHVecVectorField<3>("FLUX", flux_ptr));

  using OpPPMap = OpPostProcMapInMoab<3, 3>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(post_proc_fe->getPostProcMesh(),
                  post_proc_fe->getMapGaussPts(),

                  OpPPMap::DataMapVec{{"U", u_ptr}},

                  OpPPMap::DataMapMat{{"FLUX", flux_ptr}},

                  OpPPMap::DataMapMat{},

                  OpPPMap::DataMapMat{}

                  )

  );

  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();

  std::ostringstream strm;
  strm << "out_" << iter_num << ".h5m";
  CHKERR post_proc_fe->writeFile(strm.str().c_str());
  MoFEMFunctionReturn(0);
}
//! [Output results]

//! [OpError]
MoFEMErrorCode MixedPoisson::OpError::doWork(int side, EntityType type,
                                             EntData &data) {
  MoFEMFunctionBegin;
  const int nb_integration_pts = getGaussPts().size2();
  const double area = getMeasure();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_val = getFTensor0FromVec(*(commonDataPtr->approxVals));
  auto t_val_grad = getFTensor1FromMat<2>(*(commonDataPtr->approxValsGrad));
  auto t_flux = getFTensor1FromMat<3>(*(commonDataPtr->approxFlux));
  auto t_coords = getFTensor1CoordsAtGaussPts();
  FTensor::Tensor1<double, 2> t_diff;
  FTensor::Index<'i', 2> i;

  double error_l2 = 0;
  double error_h1 = 0;
  double error_ind = 0;
  for (int gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = t_w * area;

    double diff = t_val - MixedPoisson::analyticalFunction(
                              t_coords(0), t_coords(1), t_coords(2));
    error_l2 += alpha * sqr(diff);

    VectorDouble vec = MixedPoisson::analyticalFunctionGrad(
        t_coords(0), t_coords(1), t_coords(2));
    auto t_fun_grad = getFTensor1FromArray<2, 2>(vec);
    t_diff(i) = t_val_grad(i) - t_fun_grad(i);
    error_h1 += alpha * t_diff(i) * t_diff(i);

    t_diff(i) = t_val_grad(i) - t_flux(i);
    error_ind += alpha * t_diff(i) * t_diff(i);

    ++t_w;
    ++t_val;
    ++t_val_grad;
    ++t_flux;
    ++t_coords;
  }

  const EntityHandle ent = getFEEntityHandle();
  Tag th_error_l2, th_error_h1, th_error_ind;
  CHKERR MixedPoisson::getTagHandle(mField, "ERROR_L2_NORM", MB_TYPE_DOUBLE,
                                    th_error_l2);
  CHKERR MixedPoisson::getTagHandle(mField, "ERROR_H1_SEMINORM", MB_TYPE_DOUBLE,
                                    th_error_h1);
  CHKERR MixedPoisson::getTagHandle(mField, "ERROR_INDICATOR", MB_TYPE_DOUBLE,
                                    th_error_ind);
  CHKERR mField.get_moab().tag_set_data(th_error_l2, &ent, 1, &error_l2);
  CHKERR mField.get_moab().tag_set_data(th_error_h1, &ent, 1, &error_h1);
  CHKERR mField.get_moab().tag_set_data(th_error_ind, &ent, 1, &error_ind);

  int index = CommonData::ERROR_L2_NORM;
  constexpr std::array<int, 4> indices = {
      CommonData::ERROR_L2_NORM, CommonData::ERROR_H1_SEMINORM,
      CommonData::ERROR_INDICATOR, CommonData::TOTAL_NUMBER};
  std::array<double, 4> values;
  values[0] = error_l2;
  values[1] = error_h1;
  values[2] = error_ind;
  values[3] = 1.;
  CHKERR VecSetValues(commonDataPtr->petscVec, 4, indices.data(), values.data(),
                      ADD_VALUES);
  MoFEMFunctionReturn(0);
}
//! [OpError]

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