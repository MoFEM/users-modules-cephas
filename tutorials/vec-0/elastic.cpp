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
constexpr AssemblyType A = (SCHUR_ASSEMBLE)
                               ? AssemblyType::SCHUR
                               : AssemblyType::PETSC; //< selected assembly type

constexpr IntegrationType I =
    IntegrationType::GAUSS; //< selected integration type

using EntData = EntitiesFieldData::EntData;
using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
using BoundaryEle =
    PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEleOp = BoundaryEle::UserDataOperator;

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

using OpEssentialLhs = EssentialBC<BoundaryEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpEssentialLhs<DisplacementCubitBcData, 1, SPACE_DIM>;
using OpEssentialRhs = EssentialBC<BoundaryEleOp>::Assembly<A>::LinearForm<
    GAUSS>::OpEssentialRhs<DisplacementCubitBcData, 1, SPACE_DIM>;

template <int DIM> struct PostProcEleByDim;

template <> struct PostProcEleByDim<2> {
  using PostProcEleDomain = PostProcBrokenMeshInMoabBaseCont<DomainEle>;
  using PostProcEleBdy = PostProcBrokenMeshInMoabBaseCont<BoundaryEle>;
  using SideEle = PipelineManager::ElementsAndOpsByDim<2>::FaceSideEle;
};

template <> struct PostProcEleByDim<3> {
  using PostProcEleDomain = PostProcBrokenMeshInMoabBaseCont<BoundaryEle>;
  using PostProcEleBdy = PostProcBrokenMeshInMoabBaseCont<BoundaryEle>;
  using SideEle = PipelineManager::ElementsAndOpsByDim<3>::FaceSideEle;
};

using PostProcEleDomain = PostProcEleByDim<SPACE_DIM>::PostProcEleDomain;
using SideEle = PostProcEleByDim<SPACE_DIM>::SideEle;
using PostProcEleBdy = PostProcEleByDim<SPACE_DIM>::PostProcEleBdy;


#include <ElasticSpring.hpp>
#include <CalculateTraction.hpp>
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
      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      std::string field_name, std::string block_name,
      boost::shared_ptr<MatrixDouble> mat_D_Ptr, Sev sev);

};

MoFEMErrorCode Example::addMatBlockOps(
    boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
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
        if (block_data.size() < 2) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Expected that block has two attributes");
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
  auto pip = mField.getInterface<PipelineManager>();
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
  CHKERR bc_mng->removeBlockDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order + 1;
  };

  auto integration_rule_bc = [](int, int, int approx_order) {
    return 2 * approx_order + 1;
  };

  CHKERR pip->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pip->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pip->setBoundaryRhsIntegrationRule(integration_rule_bc);
  CHKERR pip->setBoundaryLhsIntegrationRule(integration_rule_bc);

  // Essential boundary condition.
  auto get_pre_proc_hook = [&]() {
    return EssentialPreProc<DisplacementCubitBcData>(mField,
                                                     pip->getDomainRhsFE(), {});
  };
  pip->getDomainRhsFE()->preProcessHook = get_pre_proc_hook();

    MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  auto pip = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainLhsPipeline(), {H1}, "GEOMETRY");
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainRhsPipeline(), {H1}, "GEOMETRY");
  CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpBoundaryRhsPipeline(), {NOSPACE}, "GEOMETRY");
  CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpBoundaryLhsPipeline(), {NOSPACE}, "GEOMETRY");

  auto mat_D_ptr = boost::make_shared<MatrixDouble>();

  // Assemble domain stiffness matrix
  CHKERR addMatBlockOps(pip->getOpDomainLhsPipeline(), "U", "MAT_ELASTIC",
                        mat_D_ptr, Sev::verbose);
  pip->getOpDomainLhsPipeline().push_back(new OpK("U", "U", mat_D_ptr));

  // Infernal forces
  auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
  auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
  auto mat_stress_ptr = boost::make_shared<MatrixDouble>();
  pip->getOpDomainRhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               mat_grad_ptr));
  pip->getOpDomainRhsPipeline().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", mat_grad_ptr, mat_strain_ptr));
  CHKERR addMatBlockOps(pip->getOpDomainRhsPipeline(), "U", "MAT_ELASTIC",
                        mat_D_ptr, Sev::inform);
  pip->getOpDomainRhsPipeline().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", mat_strain_ptr, mat_stress_ptr, mat_D_ptr));
  // Internal forces
  pip->getOpDomainRhsPipeline().push_back(
      new OpInternalForce("U", mat_stress_ptr,
                          [](double, double, double) constexpr { return -1; }));

  // Body forces
  CHKERR DomainRhsBCs::AddFluxToPipeline<OpDomainRhsBCs>::add(
      pip->getOpDomainRhsPipeline(), mField, "U", Sev::inform);

  // Add force boundary condition
  CHKERR BoundaryRhsBCs::AddFluxToPipeline<OpBoundaryRhsBCs>::add(
      pip->getOpBoundaryRhsPipeline(), mField, "U", -1, Sev::inform);
  // Add case for mix type of BCs 
  CHKERR BoundaryLhsBCs::AddFluxToPipeline<OpBoundaryLhsBCs>::add(
      pip->getOpBoundaryLhsPipeline(), mField, "U", Sev::verbose);

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
struct SetUpSchur {
  static boost::shared_ptr<SetUpSchur>
  createSetUpSchur(MoFEM::Interface &m_field);
  virtual MoFEMErrorCode setUp(SmartPetscObj<KSP> solver) = 0;
  virtual MoFEMErrorCode preProc() = 0;
  virtual MoFEMErrorCode postProc() = 0;

protected:
  SetUpSchur() = default;
};

MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>();
  auto solver = pip->createKSP();
  CHKERR KSPSetFromOptions(solver);

  auto dm = simple->getDM();
  auto D = createDMVector(dm);
  auto F = vectorDuplicate(D);

  auto setup_and_solve = [&]() {
    MoFEMFunctionBegin;
    BOOST_LOG_SCOPED_THREAD_ATTR("Timeline", attrs::timer());
    MOFEM_LOG("TIMER", Sev::inform) << "KSPSetUp";
    CHKERR KSPSetUp(solver);
    MOFEM_LOG("TIMER", Sev::inform) << "KSPSetUp <= Done";
    MOFEM_LOG("TIMER", Sev::inform) << "KSPSolve";
    CHKERR KSPSolve(solver, F, D);
    MOFEM_LOG("TIMER", Sev::inform) << "KSPSolve <= Done";
    MoFEMFunctionReturn(0);
  };

  MOFEM_LOG_CHANNEL("TIMER");
  MOFEM_LOG_TAG("TIMER", "timer");

  if (A == AssemblyType::SCHUR) {
    auto schur_ptr = SetUpSchur::createSetUpSchur(mField);
    CHKERR schur_ptr->setUp(solver);

    pip->getDomainLhsFE()->preProcessHook = [&]() {
      MoFEMFunctionBegin;
      if (schur_ptr)
        CHKERR schur_ptr->preProc();
      MoFEMFunctionReturn(0);
    };
    pip->getBoundaryLhsFE()->postProcessHook = [&]() {
      MoFEMFunctionBegin;
      if (schur_ptr)
        CHKERR schur_ptr->postProc();
      MoFEMFunctionReturn(0);
    };

    CHKERR setup_and_solve();
  } else {
    CHKERR setup_and_solve();
  }

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>();
  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  pip->getDomainRhsFE().reset();
  pip->getDomainLhsFE().reset();
  pip->getBoundaryRhsFE().reset();
  pip->getBoundaryLhsFE().reset();

  auto post_proc_mesh = boost::make_shared<moab::Core>();
  auto post_proc_begin = boost::make_shared<PostProcBrokenMeshInMoabBaseBegin>(
      mField, post_proc_mesh);
  auto post_proc_end = boost::make_shared<PostProcBrokenMeshInMoabBaseEnd>(
      mField, post_proc_mesh);

  auto calculate_stress_ops = [&](auto &pip) {
    AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1}, "GEOMETRY");
    auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
    auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
    auto mat_stress_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", mat_grad_ptr));
    pip.push_back(
        new OpSymmetrizeTensor<SPACE_DIM>("U", mat_grad_ptr, mat_strain_ptr));

    auto mat_D_ptr = boost::make_shared<MatrixDouble>();
    CHKERR addMatBlockOps(pip, "U", "MAT_ELASTIC", mat_D_ptr, Sev::verbose);
    pip.push_back(new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
        "U", mat_strain_ptr, mat_stress_ptr, mat_D_ptr));
    auto u_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
    auto x_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", x_ptr));
    return boost::make_tuple(u_ptr, x_ptr, mat_strain_ptr, mat_stress_ptr);
  };

  auto post_proc_domain = [&](auto post_proc_mesh) {
    auto post_proc_fe =
        boost::make_shared<PostProcEleDomain>(mField, post_proc_mesh);
    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto [u_ptr, x_ptr, mat_strain_ptr, mat_stress_ptr] =
        calculate_stress_ops(post_proc_fe->getOpPtrVector());

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {},

            {{"U", u_ptr}, {"GEOMETRY", x_ptr}},

            {},

            {{"STRAIN", mat_strain_ptr}, {"STRESS", mat_stress_ptr}}

            )

    );
    return post_proc_fe;
  };

  auto post_proc_boundary = [&](auto post_proc_mesh) {
    auto post_proc_fe =
        boost::make_shared<PostProcEleBdy>(mField, post_proc_mesh);
    AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {}, "GEOMETRY");
    auto op_loop_side =
        new OpLoopSide<SideEle>(mField, simple->getDomainFEName(), SPACE_DIM);
    // push ops to side element, through op_loop_side operator
    auto [u_ptr, x_ptr, mat_strain_ptr, mat_stress_ptr] =
        calculate_stress_ops(op_loop_side->getOpPtrVector());
    post_proc_fe->getOpPtrVector().push_back(op_loop_side);
    auto mat_traction_ptr = boost::make_shared<MatrixDouble>();
    post_proc_fe->getOpPtrVector().push_back(
        new ElasticExample::OpCalculateTraction(mat_stress_ptr,
                                                mat_traction_ptr));

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {},

            {{"U", u_ptr}, {"GEOMETRY", x_ptr}, {"T", mat_traction_ptr}},

            {},

            {{"STRAIN", mat_strain_ptr}, {"STRESS", mat_stress_ptr}}

            )

    );
    return post_proc_fe;
  };

  PetscBool post_proc_skin_only = PETSC_FALSE;
  if (SPACE_DIM == 3) {
    post_proc_skin_only = PETSC_TRUE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-post_proc_skin_only",
                               &post_proc_skin_only, PETSC_NULL);
  }
  if (post_proc_skin_only == PETSC_FALSE) {
    pip->getDomainRhsFE() = post_proc_domain(post_proc_mesh);
  }
  pip->getBoundaryRhsFE() = post_proc_boundary(post_proc_mesh);

  CHKERR DMoFEMPreProcessFiniteElements(simple->getDM(),
                                        post_proc_begin->getFEMethod());
  CHKERR pip->loopFiniteElements();
  CHKERR DMoFEMPostProcessFiniteElements(simple->getDM(),
                                         post_proc_end->getFEMethod());

  CHKERR post_proc_end->writeFile("out_elastic.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocessing results]

//! [Check]
MoFEMErrorCode Example::checkResults() {
  MOFEM_LOG_CHANNEL("WORLD");
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>();
  MoFEMFunctionBegin;
  pip->getDomainRhsFE().reset();
  pip->getDomainLhsFE().reset();
  pip->getBoundaryRhsFE().reset();
  pip->getBoundaryLhsFE().reset();

  auto integration_rule = [](int, int, int p_data) { return 2 * p_data + 1; };
  CHKERR pip->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pip->setBoundaryRhsIntegrationRule(integration_rule);

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainRhsPipeline(), {H1}, "GEOMETRY");
  CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpBoundaryRhsPipeline(), {}, "GEOMETRY");

  auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
  auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
  auto mat_stress_ptr = boost::make_shared<MatrixDouble>();

  pip->getOpDomainRhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               mat_grad_ptr));
  pip->getOpDomainRhsPipeline().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", mat_grad_ptr, mat_strain_ptr));

  auto mat_D_ptr = boost::make_shared<MatrixDouble>();
  CHKERR addMatBlockOps(pip->getOpDomainRhsPipeline(), "U", "MAT_ELASTIC",
                        mat_D_ptr, Sev::verbose);
  pip->getOpDomainRhsPipeline().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", mat_strain_ptr, mat_stress_ptr, mat_D_ptr));

  pip->getOpDomainRhsPipeline().push_back(
      new OpInternalForce("U", mat_stress_ptr));

  pip->getOpBoundaryRhsPipeline().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", mat_strain_ptr, mat_stress_ptr, mat_D_ptr));
  CHKERR DomainRhsBCs::AddFluxToPipeline<OpDomainRhsBCs>::add(
      pip->getOpDomainRhsPipeline(), mField, "U", Sev::verbose);
  CHKERR BoundaryRhsBCs::AddFluxToPipeline<OpBoundaryRhsBCs>::add(
      pip->getOpBoundaryRhsPipeline(), mField, "U", 1, Sev::verbose);

  auto dm = simple->getDM();
  auto res = createDMVector(dm);
  pip->getDomainRhsFE()->ksp_f = res;
  pip->getBoundaryRhsFE()->ksp_f = res;

  CHKERR VecZeroEntries(res);

  CHKERR mField.getInterface<FieldBlas>()->fieldScale(-1, "U");
  CHKERR pip->loopFiniteElements();
  CHKERR mField.getInterface<FieldBlas>()->fieldScale(-1, "U");

  CHKERR VecGhostUpdateBegin(res, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateEnd(res, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecAssemblyBegin(res);
  CHKERR VecAssemblyEnd(res);

  double nrm2;
  CHKERR VecNorm(res, NORM_2, &nrm2);
  MOFEM_LOG_CHANNEL("WORLD");
  MOFEM_LOG_C("WORLD", Sev::inform, "residual = %3.4e\n", nrm2);

  PetscBool test = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test", &test, PETSC_NULL);
  if (test == PETSC_TRUE) {

    auto post_proc_residual = [&](auto dm, auto f_res, auto out_name) {
      MoFEMFunctionBegin;
      auto post_proc_fe =
          boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(mField);
      using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;
      auto u_vec = boost::make_shared<MatrixDouble>();
      post_proc_fe->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_vec, f_res));
      post_proc_fe->getOpPtrVector().push_back(

          new OpPPMap(

              post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

              {},

              {{"RES", u_vec}},

              {}, {})

      );

      CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
                                      post_proc_fe);
      post_proc_fe->writeFile(out_name);
      MoFEMFunctionReturn(0);
    };

    CHKERR post_proc_residual(simple->getDM(), res, "res.h5m");

    constexpr double eps = 1e-8;
    if (nrm2 > eps)
      SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
              "Residual is not zero");
  }

  MoFEMFunctionReturn(0);
}
//! [Check]

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "TIMER"));

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

struct SetUpSchurImpl : public SetUpSchur {

  SetUpSchurImpl(MoFEM::Interface &m_field) : SetUpSchur(), mField(m_field) {
    if (S) {
      CHK_THROW_MESSAGE(
          MOFEM_DATA_INCONSISTENCY,
          "Is expected that schur matrix is not allocated. This is "
          "possible only is PC is set up twice");
    }
  }
  virtual ~SetUpSchurImpl() { S.reset(); }

  MoFEMErrorCode setUp(SmartPetscObj<KSP> solver);
  MoFEMErrorCode preProc();
  MoFEMErrorCode postProc();

private:
  MoFEMErrorCode setEntities();
  MoFEMErrorCode setUpSubDM();
  MoFEMErrorCode setOperator();
  MoFEMErrorCode setPC(PC pc);

  SmartPetscObj<Mat> S;

  MoFEM::Interface &mField;

  SmartPetscObj<DM> subDM;
  Range volEnts;
  Range subEnts;
};


MoFEMErrorCode SetUpSchurImpl::setUp(SmartPetscObj<KSP> solver) {
  MoFEMFunctionBegin;
  auto pip = mField.getInterface<PipelineManager>();
  PC pc;
  CHKERR KSPGetPC(solver, &pc);
  PetscBool is_pcfs = PETSC_FALSE;
  PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);
  if (is_pcfs) {
    if (S) {
      CHK_THROW_MESSAGE(
          MOFEM_DATA_INCONSISTENCY,
          "Is expected that schur matrix is not allocated. This is "
          "possible only is PC is set up twice");
    }
    CHKERR setEntities();
    CHKERR setUpSubDM();
    S = createDMMatrix(subDM);
    // CHKERR MatSetBlockSize(S, SPACE_DIM);
    // CHKERR MatSetOption(S, MAT_SYMMETRIC, PETSC_TRUE);
    CHKERR setOperator();
    CHKERR setPC(pc);
  } else {
    pip->getOpBoundaryLhsPipeline().push_front(new OpSchurAssembleBegin());
    pip->getOpBoundaryLhsPipeline().push_back(
        new OpSchurAssembleEnd<SCHUR_DSYSV>({}, {}, {}, {}, {}));
    pip->getOpDomainLhsPipeline().push_front(new OpSchurAssembleBegin());
    pip->getOpDomainLhsPipeline().push_back(
        new OpSchurAssembleEnd<SCHUR_DSYSV>({}, {}, {}, {}, {}));
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SetUpSchurImpl::setEntities() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  CHKERR mField.get_moab().get_entities_by_dimension(simple->getMeshset(),
                                                     SPACE_DIM, volEnts);
  CHKERR mField.get_moab().get_entities_by_handle(simple->getMeshset(),
                                                  subEnts);
  subEnts = subtract(subEnts, volEnts);
  MoFEMFunctionReturn(0);
};

MoFEMErrorCode SetUpSchurImpl::setUpSubDM() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  subDM = createDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(subDM, simple->getDM(), "SUB");
  CHKERR DMMoFEMSetSquareProblem(subDM, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(subDM, simple->getDomainFEName());
  auto sub_ents_ptr = boost::make_shared<Range>(subEnts);
  CHKERR DMMoFEMAddSubFieldRow(subDM, "U", sub_ents_ptr);
  CHKERR DMMoFEMAddSubFieldCol(subDM, "U", sub_ents_ptr);
  CHKERR DMSetUp(subDM);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SetUpSchurImpl::setOperator() {
  MoFEMFunctionBegin;
  auto pip = mField.getInterface<PipelineManager>();
  // Boundary
  pip->getOpBoundaryLhsPipeline().push_front(new OpSchurAssembleBegin());
  pip->getOpBoundaryLhsPipeline().push_back(new OpSchurAssembleEnd<SCHUR_DSYSV>(
      {"U"}, {boost::make_shared<Range>(volEnts)},
      {getDMSubData(subDM)->getSmartRowMap()}, {S}, {true}));
  // Domain
  pip->getOpDomainLhsPipeline().push_front(new OpSchurAssembleBegin());
  pip->getOpDomainLhsPipeline().push_back(new OpSchurAssembleEnd<SCHUR_DSYSV>(
      {"U"}, {boost::make_shared<Range>(volEnts)},
      {getDMSubData(subDM)->getSmartRowMap()}, {S}, {true}));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SetUpSchurImpl::setPC(PC pc) {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  SmartPetscObj<IS> vol_is;
  mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
      simple->getProblemName(), ROW, "U", 0, SPACE_DIM, vol_is, &volEnts);
  CHKERR PCFieldSplitSetIS(pc, NULL, vol_is);
  CHKERR PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, S);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SetUpSchurImpl::preProc() {
  MoFEMFunctionBegin;
  if (S) {
    CHKERR MatZeroEntries(S);
    MOFEM_LOG("TIMER", Sev::inform) << "Lhs Assemble Begin";
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SetUpSchurImpl::postProc() {
  MoFEMFunctionBegin;
  if (S) {
    CHKERR MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
  }
  MOFEM_LOG("TIMER", Sev::inform) << "Lhs Assemble End";
  MoFEMFunctionReturn(0);
}

boost::shared_ptr<SetUpSchur>
SetUpSchur::createSetUpSchur(MoFEM::Interface &m_field) {
  return boost::shared_ptr<SetUpSchur>(new SetUpSchurImpl(m_field));
}