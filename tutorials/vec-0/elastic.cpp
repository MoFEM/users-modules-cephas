/**
 * \file lesson4_elastic.cpp
 * \example lesson4_elastic.cpp
 *
 * Plane stress elastic problem
 *
 */

#include <MoFEM.hpp>

using namespace MoFEM;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using BoundaryEle = PipelineManager::EdgeEle;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using BoundaryEle = FaceElementForcesAndSourcesCore;
};

//! [Define dimension]
constexpr int SPACE_DIM = 2; //< Space dimension of problem, mesh
//! [Define dimension]

using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using OpK = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpInternalForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;

using DomainNaturalBC =
    NaturalBC<DomainEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpBodyForce =
    DomainNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;

using BoundaryNaturalBCRhs =
    NaturalBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using BoundaryNaturalBCLhs =
    NaturalBC<BoundaryEleOp>::Assembly<PETSC>::BiLinearForm<GAUSS>;

using OpForce =
    BoundaryNaturalBCRhs::OpFlux<NaturalForceMeshsets, 1, SPACE_DIM>;

constexpr double young_modulus = 1;
constexpr double poisson_ratio = 0.3;
constexpr double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
constexpr double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

#include <ElasticSpring.hpp>
using OpSpringRhs =
    BoundaryNaturalBCRhs::OpFlux<ElasticExample::SpringBcType<BLOCKSET>, 1,
                                 SPACE_DIM>;
using OpSpringLhs =
    BoundaryNaturalBCLhs::OpFlux<ElasticExample::SpringBcType<BLOCKSET>, 1,
                                 SPACE_DIM>;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();

  static MoFEMErrorCode getMatDPtr(boost::shared_ptr<MatrixDouble> mat_D_ptr,
                                   double bulk_modulus_K,
                                   double shear_modulus_G);

  MoFEMErrorCode addMatBlockOps(
      boost::ptr_vector<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      std::string field_name, std::string block_name,
      boost::shared_ptr<MatrixDouble> mat_D_Ptr);
};

MoFEMErrorCode Example::getMatDPtr(boost::shared_ptr<MatrixDouble> mat_D_ptr,
                                   double bulk_modulus_K,
                                   double shear_modulus_G) {
  MoFEMFunctionBegin;
  //! [Calculate elasticity tensor]
  auto set_material_stiffness = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    double A = (SPACE_DIM == 2)
                   ? 2 * shear_modulus_G /
                         (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                   : 1;
    auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mat_D_ptr);
    t_D(i, j, k, l) = 2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l);
  };
  //! [Calculate elasticity tensor]

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  mat_D_ptr->resize(size_symm * size_symm, 1);
  set_material_stiffness();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Example::addMatBlockOps(
    boost::ptr_vector<ForcesAndSourcesCore::UserDataOperator> &pipeline,
    std::string field_name, std::string block_name,
    boost::shared_ptr<MatrixDouble> mat_D_Ptr) {
  MoFEMFunctionBegin;

  struct OpMatBlock : public DomainEleOp {
    OpMatBlock(std::string field_name, boost::shared_ptr<MatrixDouble> m,
               Range &&ents, double bulk_modulus_K, double shear_modulus_G)
        : DomainEleOp(field_name, DomainEleOp::OPROW), matDPtr(m), feEnts(ents),
          bulkModulusK(bulk_modulus_K), shearModulusG(shear_modulus_G) {
      std::fill(&(doEntities[MBEDGE]), &(doEntities[MBMAXTYPE]), false);
    }

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;
      if (!feEnts.empty()) {
        if (feEnts.find(getFEEntityHandle()) != feEnts.end()) {
          CHKERR Example::getMatDPtr(matDPtr, bulkModulusK, shearModulusG);
        }
      } else {
        CHKERR Example::getMatDPtr(matDPtr, bulkModulusK, shearModulusG);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<MatrixDouble> matDPtr;

    Range feEnts;
    double bulkModulusK;
    double shearModulusG;
  };

  pipeline.push_back(new OpMatBlock(field_name, mat_D_Ptr, Range(),
                                    shear_modulus_G, shear_modulus_G));

  auto add_op = [&](auto &&meshset_vec_ptr) {
    MoFEMFunctionBegin;
    for (auto m : meshset_vec_ptr) {
      MOFEM_TAG_AND_LOG("WORLD", Sev::inform, "MatBlock") << *m;
      std::vector<double> block_data;
      CHKERR m->getAttributes(block_data);
      if (block_data.size() != 2) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "Expected that block has two attribute");
      }
      auto get_block_ents = [&]() {
        Range ents;
        CHKERR
        mField.get_moab().get_entities_by_handle(m->meshset, ents, true);
        return ents;
      };

      double young_modulus = block_data[0];
      double poisson_ratio = block_data[1];
      double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
      double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

      MOFEM_TAG_AND_LOG("WORLD", Sev::inform, "MatBlock")
          << "E = " << young_modulus << " nu = " << poisson_ratio;

      pipeline.push_back(new OpMatBlock(field_name, mat_D_Ptr, get_block_ents(),
                                        bulk_modulus_K, shear_modulus_G));
    }
    MOFEM_LOG_CHANNEL("WORLD");
    MoFEMFunctionReturn(0);
  };

  CHKERR add_op(

      mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

          (boost::format("%s(.*)") % block_name).str()

              ))

  );

  MoFEMFunctionReturn(0);
}

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
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

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpSetHOWeightsOnFace());

  // Infernal forces
  auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
  auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
  auto mat_stress_ptr = boost::make_shared<MatrixDouble>();
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               mat_grad_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", mat_grad_ptr, mat_strain_ptr));

  auto mat_D_ptr = boost::make_shared<MatrixDouble>();
  CHKERR addMatBlockOps(pipeline_mng->getOpDomainRhsPipeline(), "U",
                        "MAT_ELASTIC", mat_D_ptr);
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", mat_strain_ptr, mat_stress_ptr, mat_D_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpInternalForce(
      "U", mat_stress_ptr,
      [](double, double, double) constexpr { return -1; }));

  // Body forces
  CHKERR DomainNaturalBC::AddFluxToPipeline<OpBodyForce>::add(
      pipeline_mng->getOpDomainRhsPipeline(), mField, "U", {}, "BODY_FORCE",
      Sev::inform);
  // Add force boundary condition
  CHKERR BoundaryNaturalBCRhs::AddFluxToPipeline<OpForce>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), mField, "U", {}, "FORCE",
      Sev::inform);

  auto u_ptr = boost::make_shared<MatrixDouble>();
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
  CHKERR BoundaryNaturalBCRhs::AddFluxToPipeline<OpSpringRhs>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), mField, "U", u_ptr, -1,
      "SPRING", Sev::inform);
  CHKERR BoundaryNaturalBCLhs::AddFluxToPipeline<OpSpringLhs>::add(
      pipeline_mng->getOpBoundaryLhsPipeline(), mField, "U", "U", "SPRING",
      Sev::inform);

  // Essential boundary condition
  CHKERR bc_mng->removeBlockDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");

  auto get_pre_proc_hook = [&]() {
    return EssentialPreProc<DisplacementCubitBcData>(
        mField, pipeline_mng->getDomainRhsFE(), {});
  };

  pipeline_mng->getDomainRhsFE()->preProcessHook = get_pre_proc_hook();
  pipeline_mng->getBoundaryRhsFE()->preProcessHook = get_pre_proc_hook();

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpSetHOWeightsOnFace());
  auto mat_D_ptr = boost::make_shared<MatrixDouble>();
  CHKERR addMatBlockOps(pipeline_mng->getOpDomainLhsPipeline(), "U",
                        "MAT_ELASTIC", mat_D_ptr);
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpK("U", "U", mat_D_ptr));

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
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
  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

  auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
  auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
  auto mat_stress_ptr = boost::make_shared<MatrixDouble>();

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               mat_grad_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", mat_grad_ptr, mat_strain_ptr));

  auto mat_D_ptr = boost::make_shared<MatrixDouble>();
  CHKERR addMatBlockOps(post_proc_fe->getOpPtrVector(), "U", "MAT_ELASTIC",
                        mat_D_ptr);
  post_proc_fe->getOpPtrVector().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", mat_strain_ptr, mat_stress_ptr, mat_D_ptr));

  auto u_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

  using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(

          post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

          {},

          {{"U", u_ptr}},

          {},

          {{"STRAIN", mat_strain_ptr}, {"STRESS", mat_stress_ptr}}

          )

  );

  pipeline_mng->getBoundaryRhsFE().reset();
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_elastic.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocessing results]

//! [Check]
MoFEMErrorCode Example::checkResults() {
  MOFEM_LOG_CHANNEL("WORLD");
  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  MoFEMFunctionBegin;
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpSetHOWeightsOnFace());

  auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
  auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
  auto mat_stress_ptr = boost::make_shared<MatrixDouble>();

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               mat_grad_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", mat_grad_ptr, mat_strain_ptr));

  auto mat_D_ptr = boost::make_shared<MatrixDouble>();
  CHKERR addMatBlockOps(pipeline_mng->getOpDomainRhsPipeline(), "U",
                        "MAT_ELASTIC", mat_D_ptr);
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", mat_strain_ptr, mat_stress_ptr, mat_D_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpInternalForce("U", mat_stress_ptr));

  CHKERR DomainNaturalBC::AddFluxToPipeline<OpBodyForce>::add(
      pipeline_mng->getOpDomainRhsPipeline(), mField, "U", {}, "BODY_FORCE",
      Sev::inform);

  // Add force boundary condition
  CHKERR BoundaryNaturalBCRhs::AddFluxToPipeline<OpForce>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), mField, "U", {}, "FORCE",
      Sev::inform);
  auto u_ptr = boost::make_shared<MatrixDouble>();
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
  CHKERR BoundaryNaturalBCRhs::AddFluxToPipeline<OpSpringRhs>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), mField, "U", u_ptr, 1, "SPRING",
      Sev::inform);

  auto integration_rule = [](int, int, int p_data) { return 2 * p_data; };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);

  auto dm = simple->getDM();
  auto res = smartCreateDMVector(dm);
  pipeline_mng->getDomainRhsFE()->ksp_f = res;
  pipeline_mng->getBoundaryRhsFE()->ksp_f = res;

  CHKERR VecZeroEntries(res);

  CHKERR mField.getInterface<FieldBlas>()->fieldScale(-1, "U");
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR mField.getInterface<FieldBlas>()->fieldScale(-1, "U");

  CHKERR VecGhostUpdateBegin(res, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateEnd(res, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecAssemblyBegin(res);
  CHKERR VecAssemblyEnd(res);

  double nrm2;
  CHKERR VecNorm(res, NORM_2, &nrm2);
  MOFEM_LOG_CHANNEL("WORLD");
  MOFEM_LOG_C("WORLD", Sev::inform, "residual = %3.4e\n", nrm2);
  constexpr double eps = 1e-8;
  if (nrm2 > eps)
    SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "Residual is not zero");

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
    MoFEM::Interface &m_field = core; ///< finite element database interface
    //! [Create MoFEM]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
