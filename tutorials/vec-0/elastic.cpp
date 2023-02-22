/**
 * @file elastic.cpp
 * @brief elastic example
 * @version 0.13.2
 * @date 2022-09-19
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <MoFEM.hpp>

using namespace MoFEM;

//! [Define dimension]
constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh
constexpr AssemblyType A = AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType I =
    IntegrationType::GAUSS; //< selected integration type

using EntData = EntitiesFieldData::EntData;
using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
using BoundaryEle =
    PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using OpK = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    I>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpInternalForce = FormsIntegrators<DomainEleOp>::Assembly<A>::LinearForm<
    I>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;

struct DomainBCs {};
struct BoundaryBCs {};

using DomainRhsBCs = NaturalBC<DomainEleOp>::Assembly<A>::LinearForm<I>;
using OpDomainRhsBCs = DomainRhsBCs::OpFlux<DomainBCs, 1, SPACE_DIM>;
using BoundaryRhsBCs = NaturalBC<BoundaryEleOp>::Assembly<A>::LinearForm<I>;
using OpBoundaryRhsBCs = BoundaryRhsBCs::OpFlux<BoundaryBCs, 1, SPACE_DIM>;
using BoundaryLhsBCs = NaturalBC<BoundaryEleOp>::Assembly<A>::BiLinearForm<I>;
using OpBoundaryLhsBCs = BoundaryLhsBCs::OpFlux<BoundaryBCs, 1, SPACE_DIM>;

#include <ElasticSpring.hpp>
#include <NaturalDomainBC.hpp>
#include <NaturalBoundaryBC.hpp>

constexpr double young_modulus = 1;
constexpr double poisson_ratio = 0.3;
constexpr double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
constexpr double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

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

  MoFEMErrorCode addMatBlockOps(
      boost::ptr_vector<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      std::string field_name, std::string block_name,
      boost::shared_ptr<MatrixDouble> mat_D_Ptr, Sev sev);
};

MoFEMErrorCode Example::addMatBlockOps(
    boost::ptr_vector<ForcesAndSourcesCore::UserDataOperator> &pipeline,
    std::string field_name, std::string block_name,
    boost::shared_ptr<MatrixDouble> mat_D_Ptr, Sev sev) {
  MoFEMFunctionBegin;

  struct OpMatBlocks : public DomainEleOp {
    OpMatBlocks(std::string field_name, boost::shared_ptr<MatrixDouble> m,
                double bulk_modulus_K, double shear_modulus_G,
                MoFEM::Interface &m_field, Sev sev,
                std::vector<const CubitMeshSets *> meshset_vec_ptr)
        : DomainEleOp(field_name, DomainEleOp::OPROW), matDPtr(m),
          bulkModulusKDefault(bulk_modulus_K),
          shearModulusGDefault(shear_modulus_G) {
      std::fill(&(doEntities[MBEDGE]), &(doEntities[MBMAXTYPE]), false);
      CHK_THROW_MESSAGE(extractBlockData(m_field, meshset_vec_ptr, sev),
                        "Can not get data from block");
    }

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;

      for (auto &b : blockData) {

        if (b.blockEnts.find(getFEEntityHandle()) != b.blockEnts.end()) {
          CHKERR getMatDPtr(matDPtr, b.bulkModulusK, b.shearModulusG);
          MoFEMFunctionReturnHot(0);
        }
      }

      CHKERR getMatDPtr(matDPtr, bulkModulusKDefault, shearModulusGDefault);
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<MatrixDouble> matDPtr;

    struct BlockData {
      double bulkModulusK;
      double shearModulusG;
      Range blockEnts;
    };

    double bulkModulusKDefault;
    double shearModulusGDefault;
    std::vector<BlockData> blockData;

    MoFEMErrorCode
    extractBlockData(MoFEM::Interface &m_field,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr,
                     Sev sev) {
      MoFEMFunctionBegin;

      for (auto m : meshset_vec_ptr) {
        MOFEM_TAG_AND_LOG("WORLD", sev, "MatBlock") << *m;
        std::vector<double> block_data;
        CHKERR m->getAttributes(block_data);
        if (block_data.size() != 2) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Expected that block has two attribute");
        }
        auto get_block_ents = [&]() {
          Range ents;
          CHKERR
          m_field.get_moab().get_entities_by_handle(m->meshset, ents, true);
          return ents;
        };

        double young_modulus = block_data[0];
        double poisson_ratio = block_data[1];
        double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
        double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

        MOFEM_TAG_AND_LOG("WORLD", sev, "MatBlock")
            << "E = " << young_modulus << " nu = " << poisson_ratio;

        blockData.push_back(
            {bulk_modulus_K, shear_modulus_G, get_block_ents()});
      }
      MOFEM_LOG_CHANNEL("WORLD");
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode getMatDPtr(boost::shared_ptr<MatrixDouble> mat_D_ptr,
                              double bulk_modulus_K, double shear_modulus_G) {
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
        t_D(i, j, k, l) =
            2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
            A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) * t_kd(i, j) *
                t_kd(k, l);
      };
      //! [Calculate elasticity tensor]
      constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
      mat_D_ptr->resize(size_symm * size_symm, 1);
      set_material_stiffness();
      MoFEMFunctionReturn(0);
    }
  };

  pipeline.push_back(new OpMatBlocks(
      field_name, mat_D_Ptr, bulk_modulus_K, shear_modulus_G, mField, sev,

      // Get blockset using regular expression
      mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

          (boost::format("%s(.*)") % block_name).str()

              ))

          ));

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

  CHKERR simple->addDataField("GEOMETRY", H1, AINSWORTH_LEGENDRE_BASE,
                              SPACE_DIM);

  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("GEOMETRY", 2);
  CHKERR simple->setUp();

  auto project_ho_geometry = [&]() {
    Projection10NodeCoordsOnField ent_method(mField, "GEOMETRY");
    return mField.loop_dofs("GEOMETRY", ent_method);
  };
  CHKERR project_ho_geometry();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_X",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Y",
                                           "U", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Z",
                                           "U", 2, 2);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "REMOVE_ALL", "U", 0, 3);

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpDomainRhsPipeline(), {H1}, "GEOMETRY");
  CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), {NOSPACE}, "GEOMETRY");

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
                        "MAT_ELASTIC", mat_D_ptr, Sev::inform);
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", mat_strain_ptr, mat_stress_ptr, mat_D_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpInternalForce("U", mat_stress_ptr,
                          [](double, double, double) constexpr { return -1; }));

  // Body forces
  CHKERR DomainRhsBCs::AddFluxToPipeline<OpDomainRhsBCs>::add(
      pipeline_mng->getOpDomainRhsPipeline(), mField, "U", Sev::inform);
  // Add force boundary condition
  CHKERR BoundaryRhsBCs::AddFluxToPipeline<OpBoundaryRhsBCs>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), mField, "U", -1, Sev::inform);
  CHKERR BoundaryLhsBCs::AddFluxToPipeline<OpBoundaryLhsBCs>::add(
      pipeline_mng->getOpBoundaryLhsPipeline(), mField, "U", Sev::verbose);

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

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpDomainLhsPipeline(), {H1}, "GEOMETRY");

  auto mat_D_ptr = boost::make_shared<MatrixDouble>();
  CHKERR addMatBlockOps(pipeline_mng->getOpDomainLhsPipeline(), "U",
                        "MAT_ELASTIC", mat_D_ptr, Sev::verbose);
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

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      post_proc_fe->getOpPtrVector(), {H1}, "GEOMETRY");

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
                        mat_D_ptr, Sev::verbose);
  post_proc_fe->getOpPtrVector().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", mat_strain_ptr, mat_stress_ptr, mat_D_ptr));

  auto u_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
  auto x_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", x_ptr));

  using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(

          post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

          {},

          {{"U", u_ptr}, {"GEOMETRY", x_ptr}},

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

  auto integration_rule = [](int, int, int p_data) { return 2 * p_data; };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpDomainRhsPipeline(), {H1}, "GEOMETRY");
  CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), {NOSPACE}, "GEOMETRY");

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
                        "MAT_ELASTIC", mat_D_ptr, Sev::verbose);
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", mat_strain_ptr, mat_stress_ptr, mat_D_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpInternalForce("U", mat_stress_ptr));

  CHKERR DomainRhsBCs::AddFluxToPipeline<OpDomainRhsBCs>::add(
      pipeline_mng->getOpDomainRhsPipeline(), mField, "U", Sev::verbose);
  CHKERR BoundaryRhsBCs::AddFluxToPipeline<OpBoundaryRhsBCs>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), mField, "U", 1, Sev::verbose);

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
