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

constexpr int SPACE_DIM = EXECUTABLE_DIMENSION;

constexpr FieldSpace potential_space = SPACE_DIM == 2 ? H1 : HCURL;
constexpr size_t potential_field_dim = SPACE_DIM == 2 ? 1 : 3;
constexpr FieldApproximationBase approximation_base =
    SPACE_DIM == 2 ? AINSWORTH_LEGENDRE_BASE : DEMKOWICZ_JACOBI_BASE;

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
// using OpDomainSourceRhs = FormsIntegrators<DomainEleOp>::Assembly<
//     PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, SPACE_DIM>;
using OpDomainSourceRhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;
//! [Domain Operators]

//! [Boundary Operators]
using OpBoundarySourceRhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, 3>;

using OpBoundarySource2DRhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

using OpMassStab = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<BASE_DIM, 3>;

using OpMassStab2D = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
//! [Boundary Operators]

int order = 3; ///< Order
// material parameters
double mu = 1.;
double mu_0 = M_PI * 4e-7; ///< magnetic constant  N / A2; M_PI * 4e-7
double epsilon = 0.1;    ///< regularization paramater
double currentDensity = 5.;

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
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    const double vol = getMeasure();
    size_t nb_base_functions = row_data.getN().size2() / 3;
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_curl_base = row_data.getFTensor2DiffN<3, SPACE_DIM>();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha =
          t_w * vol * betaCoeff(t_coords(0), t_coords(1), t_coords(2));

      FTensor::Tensor1<double, 3> t_row_curl;

      // loop over rows base functions
      auto a_mat_ptr = &*locMat.data().begin();

      int rr = 0;
      for (; rr != nbRows; ++rr) {
        t_row_curl(i) = FTensor::levi_civita(j, i, k) * t_row_curl_base(j, k);

        FTensor::Tensor1<double, 3> t_col_curl;

        auto t_col_curl_base = col_data.getFTensor2DiffN<3, SPACE_DIM>(gg, 0);

        int cc = 0;
        for (; cc != nbCols; cc++) {
          t_col_curl(i) = FTensor::levi_civita(j, i, k) * t_col_curl_base(j, k);
          (*a_mat_ptr) += alpha * (t_row_curl(i) * t_col_curl(i));

          ++t_col_curl_base;
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

struct OpCurlCurl2D : public AssemblyDomainEleOp {
  OpCurlCurl2D(const std::string row_field_name,
               const std::string col_field_name, ScalFun beta)
      : AssemblyDomainEleOp(row_field_name, col_field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        betaCoeff(beta) {
    sYmm = true;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {

    MoFEMFunctionBegin;
    FTensor::Index<'i', 2> i;
    FTensor::Index<'j', 2> j;
    const double vol = getMeasure();
    size_t nb_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_grad = row_data.getFTensor1DiffN<SPACE_DIM>();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha =
          t_w * vol * betaCoeff(t_coords(0), t_coords(1), t_coords(2));

      FTensor::Tensor1<double, SPACE_DIM> t_row_curl;

      // loop over rows base functions
      auto a_mat_ptr = &*locMat.data().begin();

      int rr = 0;
      for (; rr != nbRows; ++rr) {
        t_row_curl(i) = FTensor::levi_civita(i, j) * t_row_grad(j);

        FTensor::Tensor1<double, SPACE_DIM> t_col_curl;

        auto t_col_grad = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        int cc = 0;
        for (; cc != nbCols; cc++) {
          t_col_curl(i) = FTensor::levi_civita(i, j) * t_col_grad(j);
          (*a_mat_ptr) += alpha * (t_row_curl(i) * t_col_curl(i));

          ++t_col_grad;
          ++a_mat_ptr;
        }
        ++t_row_grad;
      }
      for (; rr < nb_base_functions; ++rr)
        ++t_row_grad;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalFun betaCoeff;
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
  Range essentialBcRange;
  Range sourceRange;
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

  CHKERR simpleInterface->addDomainField("MAGNETIC_POTENTIAL", potential_space,
                                         approximation_base, 1);
  CHKERR simpleInterface->addBoundaryField(
      "MAGNETIC_POTENTIAL", potential_space, approximation_base, 1);

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-my_order", &order, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder("MAGNETIC_POTENTIAL", order);

  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-mu", &mu, PETSC_NULL);

  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-epsilon", &epsilon,
                               PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-current", &currentDensity,
                               PETSC_NULL);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Setup problem]

//! [Boundary condition]
MoFEMErrorCode Magnetostatics::boundaryCondition() {
  MoFEMFunctionBegin;

  auto boundary_natural_bc = [&]() {
    Range boundary_entities;

    if (SPACE_DIM == 3)
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

    if (SPACE_DIM == 2)
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
        if (bit->getName().compare(0, 9, "NATURALBC") == 0) {
          Range faces;
          CHKERR mField.get_moab().get_entities_by_type(bit->meshset, MBEDGE,
                                                        faces, true);
          CHKERR mField.get_moab().get_adjacencies(
              faces, 1, true, boundary_entities, moab::Interface::UNION);
          boundary_entities.merge(faces);
        }
      }

    return boundary_entities;
  };

  auto boundary_essential_bc = [&]() {
    Range boundary_entities;
    auto bc_mng = mField.getInterface<BcManager>();
    CHKERR bc_mng->pushMarkDOFsOnEntities(simpleInterface->getProblemName(),
                                          "ESSENTIAL", "MAGNETIC_POTENTIAL", 0,
                                          0);

    auto &bc_map = bc_mng->getBcMapByBlockName();
    // boundaryMarker = boost::make_shared<std::vector<char unsigned>>();
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)ESSENTIAL(.*)"))) {
        boundary_entities.merge(*(b.second->getBcEntsPtr()));
      }
    }
    // Add vertices to boundary entities
    Range boundary_vertices;
    CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                              boundary_vertices, true);
    boundary_entities.merge(boundary_vertices);

    return boundary_entities;
  };

  auto source_bc = [&]() {
    Range boundary_entities;

    if (SPACE_DIM == 3)
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
        if (bit->getName().compare(0, 8, "SOURCE_J") == 0) {
          Range faces;
          CHKERR mField.get_moab().get_entities_by_type(bit->meshset, MBTET,
                                                        faces, true);
          CHKERR mField.get_moab().get_adjacencies(
              faces, 1, true, boundary_entities, moab::Interface::UNION);
          boundary_entities.merge(faces);
        }
      }

    if (SPACE_DIM == 2)
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
        if (bit->getName().compare(0, 8, "SOURCE_J") == 0) {
          Range faces;
          CHKERR mField.get_moab().get_entities_by_type(bit->meshset, MBTRI,
                                                        faces, true);
          CHKERR mField.get_moab().get_adjacencies(
              faces, 1, true, boundary_entities, moab::Interface::UNION);
          boundary_entities.merge(faces);
        }
      }

    return boundary_entities;
  };

  naturalBcRange = boundary_natural_bc();
  essentialBcRange = boundary_essential_bc();
  sourceRange = source_bc();

  ParallelComm *pcomm =
      ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
  if (SPACE_DIM == 3)
    if (essentialBcRange.empty()) {
      Range tets;
      CHKERR mField.get_moab().get_entities_by_type(0, MBTET, tets);
      Skinner skin(&mField.get_moab());
      Range skin_faces; // skin faces from 3d ents
      CHKERR skin.find_skin(0, tets, false, skin_faces);
      skin_faces = subtract(skin_faces, naturalBcRange);
      Range proc_skin;
      CHKERR pcomm->filter_pstatus(skin_faces,
                                   PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                   PSTATUS_NOT, -1, &proc_skin);
      CHKERR mField.get_moab().get_adjacencies(
          proc_skin, 1, true, essentialBcRange, moab::Interface::UNION);
      essentialBcRange.merge(proc_skin);
    }
  if (SPACE_DIM == 2)
    if (essentialBcRange.empty()) {
      Range faces;
      CHKERR mField.get_moab().get_entities_by_type(0, MBTRI, faces);
      Skinner skin(&mField.get_moab());
      Range skin_edges; // skin faces from 3d ents
      CHKERR skin.find_skin(0, faces, false, skin_edges);
      skin_edges = subtract(skin_edges, naturalBcRange);
      Range proc_skin;
      CHKERR pcomm->filter_pstatus(skin_edges,
                                   PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                   PSTATUS_NOT, -1, &proc_skin);
      CHKERR mField.get_moab().get_adjacencies(
          proc_skin, 1, true, essentialBcRange, moab::Interface::UNION);
      essentialBcRange.merge(proc_skin);
    }

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Assemble system]
MoFEMErrorCode Magnetostatics::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
      simpleInterface->getProblemName(), "MAGNETIC_POTENTIAL",
      essentialBcRange);

  auto add_base_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    if (SPACE_DIM == 2) {
      auto det_ptr = boost::make_shared<VectorDouble>();
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
      pipeline.push_back(new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));
    }
    MoFEMFunctionReturn(0);
  };

  CHKERR add_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_base_ops(pipeline_mng->getOpDomainRhsPipeline());

  // Push Domain operators to the Pipeline that is responsible for calculating
  // LHS
  if (SPACE_DIM == 3) {
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pipeline_mng->getOpDomainLhsPipeline(), {HCURL});
    pipeline_mng->getOpDomainLhsPipeline().push_back(new OpCurlCurl(
        "MAGNETIC_POTENTIAL", "MAGNETIC_POTENTIAL",
        [](const double, const double, const double) { return 1. / mu; }));
    //     }));
    pipeline_mng->getOpDomainLhsPipeline().push_back(new OpMassStab(
        "MAGNETIC_POTENTIAL", "MAGNETIC_POTENTIAL",
        [](const double, const double, const double) { return epsilon / mu; }));
  }

  if (SPACE_DIM == 2) {
    pipeline_mng->getOpDomainLhsPipeline().push_back(new OpCurlCurl2D(
        "MAGNETIC_POTENTIAL", "MAGNETIC_POTENTIAL",
        [](const double, const double, const double) { return 1. / mu; }));
  }

  // Push Domain operators to the Pipeline that is responsible for calculating
  // RHS
  if (SPACE_DIM == 3)
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pipeline_mng->getOpDomainRhsPipeline(), {HCURL});

  auto source_fun = [&](const double x, const double y, const double z) {
    return 5.;
  };
  if (SPACE_DIM == 2)
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpDomainSourceRhs("MAGNETIC_POTENTIAL", source_fun,
                              boost::make_shared<Range>(sourceRange)));

  boost::function<FTensor::Tensor1<double, 3>(const double, const double,
                                              const double)>
      natural_fun = [](double x, double y, double z) {
        const double r = sqrt(x * x + y * y);
        return FTensor::Tensor1<double, 3>{(-y / r), (x / r), 0.};
      };

  auto boundary_fun = [&](const double x, const double y, const double z) {
    return 0.;
  };

  // Push Boundary operators to the Pipeline that is responsible for
  // calculating RHS
  if (SPACE_DIM == 3) {
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(
        new OpHOSetCovariantPiolaTransformOnFace3D(HCURL));
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(
        new OpBoundarySourceRhs("MAGNETIC_POTENTIAL", natural_fun,
                                boost::make_shared<Range>(naturalBcRange)));
  }

  if (SPACE_DIM == 2) {
    // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
    //     new OpBoundarySource2DRhs("MAGNETIC_POTENTIAL", boundary_fun,
    //                               boost::make_shared<Range>(naturalBcRange)));
  }

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
  if (SPACE_DIM == 3)
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {HCURL});

  if (SPACE_DIM == 2) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHOJacForFace(jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(inv_jac_ptr));
  }

  auto field_val_ptr = boost::make_shared<MatrixDouble>();
  auto field_val_ptr_2D = boost::make_shared<VectorDouble>();
  auto induction_ptr = boost::make_shared<MatrixDouble>();

  if (SPACE_DIM == 3)
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHVecVectorField<3, SPACE_DIM>("MAGNETIC_POTENTIAL",
                                                     field_val_ptr));

  if (SPACE_DIM == 2)
    post_proc_fe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        "MAGNETIC_POTENTIAL", field_val_ptr_2D));

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateHcurlVectorCurl<potential_field_dim, SPACE_DIM>(
          "MAGNETIC_POTENTIAL", induction_ptr));

  using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

  if (SPACE_DIM == 2)
    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(
            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            OpPPMap::DataMapVec{{"MAGNETIC_POTENTIAL_2D", field_val_ptr_2D}},

            OpPPMap::DataMapMat{{"MAGNETIC_INDUCTION_FIELD", induction_ptr}},

            OpPPMap::DataMapMat{},

            OpPPMap::DataMapMat{}

            )

    );

  if (SPACE_DIM == 3)
    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(
            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            OpPPMap::DataMapVec{},

            OpPPMap::DataMapMat{{"MAGNETIC_POTENTIAL", field_val_ptr},
                                {"MAGNETIC_INDUCTION_FIELD", induction_ptr}},

            OpPPMap::DataMapMat{},

            OpPPMap::DataMapMat{}

            )

    );

  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_magneto_result_" +
                                 boost::lexical_cast<std::string>(SPACE_DIM) +
                                 "D.h5m");

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