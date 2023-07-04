#include <BasicFiniteElements.hpp>
#include <MoFEM.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using BoundaryEle = PipelineManager::EdgeEle;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using BoundaryEle = PipelineManager::FaceEle;
};

constexpr int SPACE_DIM = 3;
constexpr int BASE_DIM = 3;
constexpr AssemblyType A = AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType I =
    IntegrationType::GAUSS;                     //< selected integration type
constexpr CoordinateTypes coord_type = CARTESIAN;

using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using AssemblyDomainEleOp = FormsIntegrators<DomainEleOp>::Assembly<A>::OpBase;
using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<A>::OpBase;

//! [Domain Operators]
// OpCurlCurl - Once moved to Bilinear Forms in Core.
//! [Domain Operators]

//! [Boundary Operators]
using OpBoundarySourceRhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<3, 3>;

using OpMassStab = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<3, 3>;
//! [Boundary Operators]

int order = 2; ///< Order
// material parameters
double mu = 1.;       ///< magnetic constant  N / A2
double epsilon = 0.1; ///< regularization paramater

using ScalFun =
    boost::function<double(const double, const double, const double)>;

template <int SPACE_DIM>
using VecFun = boost::function<FTensor::Tensor1<double, SPACE_DIM>(
    const double, const double, const double)>;

struct OpCurlCurl : public AssemblyDomainEleOp {
  OpCurlCurl(const std::string row_field_name, const std::string col_field_name,
             ScalFun beta)
      : AssemblyDomainEleOp(row_field_name, col_field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        betaCoeff(beta) {
    sYmm = true;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {

    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getN().size2() / 3;
    const int nb_col_dofs = col_data.getN().size2() / 3;

    // locMat.resize(nb_row_dofs, nb_col_dofs, false);
    // locMat.clear();

    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    const double vol = getMeasure();
    size_t nb_base_functions = row_data.getN().size2() / 3;
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_curl_base = row_data.getFTensor2DiffN<3, SPACE_DIM>();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    const int nb_gauss_pts = getGaussPts().size2();

    for (int gg = 0; gg != nb_gauss_pts;
         gg++) { // nb_gauss_pts nbIntegrationPts

      const double alpha =
          t_w * vol * betaCoeff(t_coords(0), t_coords(1), t_coords(2));

      FTensor::Tensor1<double, SPACE_DIM> t_row_curl;

      // loop over rows base functions
      auto a_mat_ptr = &*locMat.data().begin();

      int rr = 0;
      for (; rr != nbRows; ++rr) {
        t_row_curl(i) = FTensor::levi_civita(j, i, k) * t_row_curl_base(j, k);

        FTensor::Tensor1<double, SPACE_DIM> t_col_curl;

        // FTensor::Tensor0<double *> t_local_mat(&locMat(rr, 0), 1);

        auto t_col_curl_base = col_data.getFTensor2DiffN<3, SPACE_DIM>(gg, 0);

        int cc = 0;
        for (; cc != nbCols; cc++) {
          t_col_curl(i) = FTensor::levi_civita(j, i, k) * t_col_curl_base(j, k);

          // locMat(rr, cc) += alpha * t_row_curl(i) * t_col_curl(i);
          // t_local_mat += alpha * t_row_curl(i) * t_col_curl(i);
          (*a_mat_ptr) += alpha * (t_row_curl(i) * t_col_curl(i));

          ++t_col_curl_base;
          // ++t_local_mat;
          ++a_mat_ptr;
        }
        ++t_row_curl_base;
      }
      for (; rr < nb_base_functions; ++rr)
        ++t_row_curl_base;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalFun betaCoeff;
};

struct OpStab : public AssemblyDomainEleOp {
  OpStab(const std::string row_field_name, const std::string col_field_name,
         ScalFun beta)
      : AssemblyDomainEleOp(row_field_name, col_field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        betaCoeff(beta) {
    sYmm = true;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {

    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getN().size2() / 3;
    const int nb_col_dofs = col_data.getN().size2() / 3;

    // locMat.resize(nb_row_dofs, nb_col_dofs, false);
    // locMat.clear();

    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'l', SPACE_DIM> l;
    const double vol = getMeasure();
    size_t nb_base_functions = row_data.getN().size2() / 3;
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor1N<3>();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    const int nb_gauss_pts = getGaussPts().size2();

    for (int gg = 0; gg != nb_gauss_pts;
         gg++) { // nb_gauss_pts nbIntegrationPts

      const double alpha =
          t_w * vol * betaCoeff(t_coords(0), t_coords(1), t_coords(2));

      // loop over rows base functions
      auto a_mat_ptr = &*locMat.data().begin();

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        auto t_col_base = col_data.getFTensor1N<3>();

        FTensor::Tensor0<double *> t_local_mat(&locMat(rr, 0), 1);

        int cc = 0;
        for (; cc != nbCols; cc++) {

          // locMat(rr, cc) += alpha * t_row_base(i) * t_col_base(i);
          // t_local_mat += alpha * t_row_base(i) * t_col_base(i);
          *a_mat_ptr += alpha * t_row_base(i) * t_col_base(i);

          ++t_col_base;
          ++a_mat_ptr;
          // ++t_local_mat;
        }
        ++t_row_base;
      }
      for (; rr < nb_base_functions; ++rr)
        ++t_row_base;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalFun betaCoeff;
};

struct OpBoundaryRhs : public AssemblyBoundaryEleOp {
  OpBoundaryRhs(const std::string field_name, VecFun<SPACE_DIM> source_fun,
                boost::shared_ptr<Range> ents_ptr = nullptr)
      : AssemblyBoundaryEleOp(field_name, field_name,
                              AssemblyBoundaryEleOp::OPROW, ents_ptr),
        sourceFun(source_fun) {}
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {

    MoFEMFunctionBegin;

    FTensor::Index<'i', SPACE_DIM> i;

    const int nb_row_dofs = data.getN().size2() / 3;
    locF.resize(nb_row_dofs, false);
    locF.clear();

    size_t nb_base_functions = data.getN().size2() / 3;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_coords = getFTensor1CoordsAtGaussPts();
    auto t_row_base = data.getFTensor1N<3>();

    const int nb_gauss_pts = getGaussPts().size2();

    for (int gg = 0; gg != nb_gauss_pts;
         gg++) { // nbIntegrationPts nb_gauss_pts
      const double alpha = t_w * vol;
      auto t_source = sourceFun(t_coords(0), t_coords(1), t_coords(2));

      int rr = 0;
      for (; rr != nb_row_dofs; ++rr) {
        locF[rr] += alpha * t_row_base(i) * t_source(i);
        ++t_row_base;
      }
      for (; rr < nb_base_functions; ++rr)
        ++t_row_base;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  VecFun<SPACE_DIM> sourceFun;
};

struct Magnetostatics {
public:
  Magnetostatics(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  // MoFEM interfaces
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  // boundary condition ranges
  Range naturalBcRange;
};

Magnetostatics::Magnetostatics(MoFEM::Interface &m_field) : mField(m_field) {}

//! [Read mesh]
MoFEMErrorCode Magnetostatics::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Setup problem]
MoFEMErrorCode Magnetostatics::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField("MAGNETIC_POTENTIAL", HCURL,
                                         DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simpleInterface->addBoundaryField("MAGNETIC_POTENTIAL", HCURL,
                                           DEMKOWICZ_JACOBI_BASE, 1);

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-my_order", &order, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder("MAGNETIC_POTENTIAL", order);

  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-mu", &mu, PETSC_NULL);

  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-epsilon", &epsilon,
                               PETSC_NULL);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Setup problem]

//! [Boundary condition]
MoFEMErrorCode Magnetostatics::boundaryCondition() {
  MoFEMFunctionBegin;

  auto natural_bc = [&]() {
    Range boundary_entities;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 9, "NATURALBC") == 0) {
        Range faces;
        CHKERR mField.get_moab().get_entities_by_type(bit->meshset, MBTRI,
                                                      faces, true);
        CHKERR mField.get_moab().get_adjacencies(
            faces, 1, true, boundary_entities, moab::Interface::UNION);
        boundary_entities.merge(faces);
      }
    }
    // auto bc_mng = mField.getInterface<BcManager>();
    // CHKERR bc_mng->pushMarkDOFsOnEntities(simpleInterface->getProblemName(),
    //                                       "MAGNETIC_POTENTIAL",
    //                                       "MAGNETIC_POTENTIAL", 0, 0);

    // auto &bc_map = bc_mng->getBcMapByBlockName();
    // for (auto b : bc_map) {
    //   if (std::regex_match(b.first, std::regex("(.*)NATURAL(.*)"))) {
    //     boundary_entities.merge(*(b.second->getBcEntsPtr()));
    //   }
    // }
    // // Add vertices to boundary entities
    // Range boundary_vertices;
    // CHKERR mField.get_moab().get_connectivity(boundary_entities,
    //                                           boundary_vertices, true);
    // boundary_entities.merge(boundary_vertices);

    return boundary_entities;
  };

  naturalBcRange = natural_bc();

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Assemble system]
MoFEMErrorCode Magnetostatics::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  // auto det_ptr = boost::make_shared<VectorDouble>();
  // auto jac_ptr = boost::make_shared<MatrixDouble>();
  // auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
  //     new OpCalculateHOJacForFace(jac_ptr));
  // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
  //     new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
  // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
  //     new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
  // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
  //     new OpSetInvJacHcurlFace(inv_jac_ptr));

  // Push Domain operators to the Pipeline that is responsible for calculating
  // LHS
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpDomainLhsPipeline(), {HCURL});
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpCurlCurl(
      "MAGNETIC_POTENTIAL", "MAGNETIC_POTENTIAL",
      [](const double, const double, const double) { return 1. / mu; }));
  // pipeline_mng->getOpDomainLhsPipeline().push_back(new OpStab(
  //     "MAGNETIC_POTENTIAL", "MAGNETIC_POTENTIAL",
  //     [](const double, const double, const double) { return epsilon / mu;
  //     }));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpMassStab(
      "MAGNETIC_POTENTIAL", "MAGNETIC_POTENTIAL",
      [](const double, const double, const double) { return epsilon / mu; }));

  // Push Domain operators to the Pipeline that is responsible for calculating
  // RHS
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpDomainRhsPipeline(), {HCURL});

  boost::function<FTensor::Tensor1<double, 3>(const double, const double,
                                              const double)>
      natural_fun = [](double x, double y, double z) {
        const double r = sqrt(x * x + y * y);
        return FTensor::Tensor1<double, 3>{(-y / r), (x / r), 0.};
      };

  // Push Boundary operators to the Pipeline that is responsible for calculating
  // RHS
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpHOSetCovariantPiolaTransformOnFace3D(HCURL));
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpBoundarySourceRhs("MAGNETIC_POTENTIAL", natural_fun,
                              boost::make_shared<Range>(naturalBcRange)));

  MoFEMFunctionReturn(0);
}
//! [Assemble system]

//! [Set integration rules]
MoFEMErrorCode Magnetostatics::setIntegrationRules() {
  MoFEMFunctionBegin;
  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto rule_lhs = [](int, int, int p) -> int { return 2 * p + 1; };
  auto rule_rhs = [](int, int, int p) -> int { return 2 * p + 1; };

  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

  auto boundary_rule_lhs = [](int, int, int p) -> int { return 2 * p; };
  auto boundary_rule_rhs = [](int, int, int p) -> int { return 2 * p; };

  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(boundary_rule_lhs);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(boundary_rule_rhs);

  MoFEMFunctionReturn(0);
}
//! [Set integration rules]

//! [Solve system]
MoFEMErrorCode Magnetostatics::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(ksp_solver);
  CHKERR KSPSetUp(ksp_solver);

  // Create RHS and solution vectors
  auto dm = simpleInterface->getDM();
  auto F = createDMVector(dm);
  auto D = vectorDuplicate(F);

  // Solve the system
  CHKERR KSPSolve(ksp_solver, F, D);

  // Scatter result data on the mesh
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve system]

//! [Output results]
MoFEMErrorCode Magnetostatics::outputResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      post_proc_fe->getOpPtrVector(), {HCURL});

  auto field_val_ptr = boost::make_shared<MatrixDouble>();
  auto induction_ptr = boost::make_shared<MatrixDouble>();

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateHVecVectorField<SPACE_DIM>("MAGNETIC_POTENTIAL",
                                                field_val_ptr));
  // Only implement for <3, 3>  OR <1, 2> (<BASE_DIM, SPACE_DIM>)
  post_proc_fe->getOpPtrVector().push_back(new OpCalculateHcurlVectorCurl<3, 3>(
      "MAGNETIC_POTENTIAL", induction_ptr));

  using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(
          post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

          OpPPMap::DataMapVec{},

          OpPPMap::DataMapMat{{"MAGNETIC_POTENTIAL", field_val_ptr},
                              {"MAGNETIC_INDUCTION_FIELD", induction_ptr}},
          //  {"MAGNETIC_INDUCTION_FIELD", induction_ptr}

          OpPPMap::DataMapMat{},

          OpPPMap::DataMapMat{}

          )

  );

  // post_proc.getOpPtrVector().push_back(new OpPostProcessCurl(
  //     blockData, post_proc.getPostProcMesh(), post_proc.getMapGaussPts()));
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_magneto_result.h5m");

  MoFEMFunctionReturn(0);
}
//! [Output results]

//! [Run program]
MoFEMErrorCode Magnetostatics::runProgram() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR setIntegrationRules();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();

  MoFEMFunctionReturn(0);
}
//! [Run program]

//! [Main]
int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Error handling
  try {
    // Register MoFEM discrete manager in PETSc
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    // Create MOAB instance
    moab::Core mb_instance;              // mesh database
    moab::Interface &moab = mb_instance; // mesh database interface

    // Create MoFEM instance
    MoFEM::Core core(moab);           // finite element database
    MoFEM::Interface &m_field = core; // finite element interface

    // Run the main analysis
    Magnetostatics ex(m_field);
    CHKERR ex.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
//! [Main]