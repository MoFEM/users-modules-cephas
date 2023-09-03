/* The above code is a preprocessor directive in C++ that checks if the macro
"EXECUTABLE_DIMENSION" has been defined. If it has not been defined, it is set
to 3" */
// #ifndef EXECUTABLE_DIMENSION
// #define EXECUTABLE_DIMENSION 3
// #endif


#include <MoFEM.hpp>
#include <MatrixFunction.hpp>

// #ifdef PYTHON_SFD
// #include <boost/python.hpp>
// #include <boost/python/def.hpp>
// namespace bp = boost::python;
// #endif

using namespace MoFEM;

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

constexpr AssemblyType AT =
    (SCHUR_ASSEMBLE) ? AssemblyType::SCHUR
                     : AssemblyType::PETSC; //< selected assembly type

constexpr IntegrationType IT =
    IntegrationType::GAUSS; //< selected integration type
constexpr CoordinateTypes coord_type = CARTESIAN;

template <int DIM> struct ElementsAndOps;

template <> struct ElementsAndOps<2> : PipelineManager::ElementsAndOpsByDim<2> {
  static constexpr FieldSpace CONTACT_SPACE = HCURL;
};

template <> struct ElementsAndOps<3> : PipelineManager::ElementsAndOpsByDim<3> {
  static constexpr FieldSpace CONTACT_SPACE = HDIV;
};

constexpr FieldSpace ElementsAndOps<2>::CONTACT_SPACE;
constexpr FieldSpace ElementsAndOps<3>::CONTACT_SPACE;

/* The above code is defining an alias `EntData` for the type
`EntitiesFieldData::EntData`. This is a C++ syntax for creating a new name for
an existing type. */
using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;
using SkinPostProcEle = PostProcBrokenMeshInMoab<BoundaryEle>;

//! [Specialisation for assembly]

struct MonitorIncompressible : public FEMethod {

  MonitorIncompressible(SmartPetscObj<DM> dm,
          std::pair<boost::shared_ptr<PostProcEle>,
                    boost::shared_ptr<SkinPostProcEle>>
              pair_post_proc_fe,
          boost::shared_ptr<DomainEle> reaction_fe,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uz_scatter)
      : dM(dm), reactionFe(reaction_fe), uXScatter(ux_scatter),
        uYScatter(uy_scatter), uZScatter(uz_scatter) {
    postProcFe = pair_post_proc_fe.first;
    skinPostProcFe = pair_post_proc_fe.second;
  };

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    MoFEM::Interface *m_field_ptr;
    CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);

    auto make_vtk = [&]() {
      MoFEMFunctionBegin;
      if (postProcFe) {
        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe,
                                        getCacheWeakPtr());
        CHKERR postProcFe->writeFile("out_incomp_elasticity" +
                                     boost::lexical_cast<std::string>(ts_step) +
                                     ".h5m");
      }
      if (skinPostProcFe) {
        CHKERR DMoFEMLoopFiniteElements(dM, "bFE", skinPostProcFe,
                                        getCacheWeakPtr());
        CHKERR skinPostProcFe->writeFile(
            "out_skin_incomp_elasticity_" + boost::lexical_cast<std::string>(ts_step) +
            ".h5m");
      }
      MoFEMFunctionReturn(0);
    };

    auto calculate_reaction = [&]() {
      MoFEMFunctionBegin;
      auto r = createDMVector(dM);
      reactionFe->f = r;
      CHKERR VecZeroEntries(r);
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", reactionFe, getCacheWeakPtr());

#ifndef NDEBUG
      auto post_proc_residual = [&](auto dm, auto f_res, auto out_name) {
        MoFEMFunctionBegin;
        auto post_proc_fe =
            boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(
                *m_field_ptr);
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

        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", post_proc_fe);
        post_proc_fe->writeFile("res.h5m");
        MoFEMFunctionReturn(0);
      };

      CHKERR post_proc_residual(dM, r, "reaction");
#endif // NDEBUG

      MoFEMFunctionReturn(0);
    };

    auto print_max_min = [&](auto &tuple, const std::string msg) {
      MoFEMFunctionBegin;
      CHKERR VecScatterBegin(std::get<1>(tuple), ts_u, std::get<0>(tuple),
                             INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecScatterEnd(std::get<1>(tuple), ts_u, std::get<0>(tuple),
                           INSERT_VALUES, SCATTER_FORWARD);
      double max, min;
      CHKERR VecMax(std::get<0>(tuple), PETSC_NULL, &max);
      CHKERR VecMin(std::get<0>(tuple), PETSC_NULL, &min);
      MOFEM_LOG_C("INCOMP_ELASTICITY", Sev::inform,
                  "%s time %3.4e min %3.4e max %3.4e", msg.c_str(), ts_t, min,
                  max);
      MoFEMFunctionReturn(0);
    };

    CHKERR make_vtk();
    if (reactionFe)
      CHKERR calculate_reaction();
    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");
    if constexpr (SPACE_DIM == 3)
      CHKERR print_max_min(uZScatter, "Uz");

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<SkinPostProcEle> skinPostProcFe;
  boost::shared_ptr<DomainEle> reactionFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;
};

// Assemble to A matrix, by default, however, some terms are assembled only to
// preconditioning.

template <>
typename MoFEM::OpBaseImpl<AT, DomainEleOp>::MatSetValuesHook
    MoFEM::OpBaseImpl<AT, DomainEleOp>::matSetValuesHook =
        [](ForcesAndSourcesCore::UserDataOperator *op_ptr,
           const EntitiesFieldData::EntData &row_data,
           const EntitiesFieldData::EntData &col_data, MatrixDouble &m) {
          return MatSetValues<AssemblyTypeSelector<AT>>(
              op_ptr->getKSPA(), row_data, col_data, m, ADD_VALUES);
        };

template <>
typename MoFEM::OpBaseImpl<AT, BoundaryEleOp>::MatSetValuesHook
    MoFEM::OpBaseImpl<AT, BoundaryEleOp>::matSetValuesHook =
        [](ForcesAndSourcesCore::UserDataOperator *op_ptr,
           const EntitiesFieldData::EntData &row_data,
           const EntitiesFieldData::EntData &col_data, MatrixDouble &m) {
          return MatSetValues<AssemblyTypeSelector<AT>>(
              op_ptr->getKSPA(), row_data, col_data, m, ADD_VALUES);
        };

/**
 * @brief Element used to specialise assembly
 *
 */
struct BoundaryEleOpStab : public BoundaryEleOp {
  using BoundaryEleOp::BoundaryEleOp;
};

/**
 * @brief Specialise assembly for Stabilised matrix
 *
 * @tparam
 */
template <>
typename MoFEM::OpBaseImpl<AT, BoundaryEleOpStab>::MatSetValuesHook
    MoFEM::OpBaseImpl<AT, BoundaryEleOpStab>::matSetValuesHook =
        [](ForcesAndSourcesCore::UserDataOperator *op_ptr,
           const EntitiesFieldData::EntData &row_data,
           const EntitiesFieldData::EntData &col_data, MatrixDouble &m) {
          return MatSetValues<AssemblyTypeSelector<AT>>(
              op_ptr->getKSPB(), row_data, col_data, m, ADD_VALUES);
        };
//! [Specialisation for assembly]

constexpr FieldSpace CONTACT_SPACE = ElementsAndOps<SPACE_DIM>::CONTACT_SPACE;

//! [Operators used for contact]
using OpSpringLhs = FormsIntegrators<BoundaryEleOp>::Assembly<AT>::BiLinearForm<
    IT>::OpMass<1, SPACE_DIM>;
using OpSpringRhs = FormsIntegrators<BoundaryEleOp>::Assembly<AT>::LinearForm<
    IT>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Operators used for contact]

//! [Operators used for RHS incompressible elasticity]
using OpDomainGradTimesTensor = FormsIntegrators<DomainEleOp>::Assembly<
    AT>::LinearForm<GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;

using OpDivDeltaUTimesP = FormsIntegrators<DomainEleOp>::Assembly<AT>::LinearForm<
    GAUSS>::OpMixDivTimesU<1, SPACE_DIM, SPACE_DIM>;

using OpBaseTimesScalarValues = FormsIntegrators<DomainEleOp>::Assembly<
    AT>::LinearForm<GAUSS>::OpBaseTimesScalar<1>;

//! [Operators used for RHS incompressible elasticity]

//! [Operators used for RHS incompressible elasticity]
using OpMassPressure = FormsIntegrators<DomainEleOp>::Assembly<AT>::BiLinearForm<
        GAUSS>::OpMass<1, 1>;
//! [Operators used for RHS incompressible elasticity]

using BoundaryNaturalBC =
    NaturalBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpForce = BoundaryNaturalBC::OpFlux<NaturalForceMeshsets, 1, SPACE_DIM>;

PetscBool is_quasi_static = PETSC_TRUE;

int order = 2;
int geom_order = 1;
double young_modulus = 100;
double poisson_ratio = 0.25;
double rho = 0.0;
double spring_stiffness = 0.0;
double vis_spring_stiffness = 0.0;
double alpha_damping = 0;
double mu = 1.;
double lambda = 1.;

double scale = 1.;

PetscBool isDiscontinuousPressure = PETSC_FALSE;
PetscBool isATPetscFieldsplit = PETSC_FALSE;

namespace ContactOps {
double cn_contact = 0.1;
}; // namespace ContactOps

#include <HenckyOps.hpp>
using namespace HenckyOps;
#include <ContactOps.hpp>
#include <PostProcContact.hpp>
#include <ContactNaturalBC.hpp>

using DomainRhsBCs = NaturalBC<DomainEleOp>::Assembly<AT>::LinearForm<IT>;
using OpDomainRhsBCs =
    DomainRhsBCs::OpFlux<ContactOps::DomainBCs, 1, SPACE_DIM>;
using BoundaryRhsBCs = NaturalBC<BoundaryEleOp>::Assembly<AT>::LinearForm<IT>;
using OpBoundaryRhsBCs =
    BoundaryRhsBCs::OpFlux<ContactOps::BoundaryBCs, 1, SPACE_DIM>;
using BoundaryLhsBCs = NaturalBC<BoundaryEleOp>::Assembly<AT>::BiLinearForm<IT>;
using OpBoundaryLhsBCs =
    BoundaryLhsBCs::OpFlux<ContactOps::BoundaryBCs, 1, SPACE_DIM>;

using namespace ContactOps;

struct Incompressible {

  Incompressible(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode tsSolve();
  MoFEMErrorCode checkResults();

  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<DomainEle> reactionFe;

#ifdef PYTHON_SFD
  boost::shared_ptr<SDFPython> sdfPythonPtr;
#endif

  struct ScaledTimeScale : public MoFEM::TimeScale {
    using MoFEM::TimeScale::TimeScale;
    double getScale(const double time) {
      return scale * MoFEM::TimeScale::getScale(time);
    };
  };
};

template <int DIM>
struct OpCalculateLameStress : public ForcesAndSourcesCore::UserDataOperator {
  OpCalculateLameStress(double m_u, boost::shared_ptr<MatrixDouble> stress_ptr,
                        boost::shared_ptr<MatrixDouble> strain_ptr,
                        boost::shared_ptr<VectorDouble> pressure_ptr)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST), mU(m_u),
        stressPtr(stress_ptr), strainPtr(strain_ptr),
        pressurePtr(pressure_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    // Define Indicies
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;

    // Define Kronecker Delta
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<double>();

    // Number of Gauss points
    const size_t nb_gauss_pts = getGaussPts().size2();

    stressPtr->resize((DIM * (DIM + 1)) / 2, nb_gauss_pts);
    auto t_stress =
      getFTensor2SymmetricFromMat<DIM>(*(stressPtr));
    auto t_strain =
      getFTensor2SymmetricFromMat<DIM>(*(strainPtr));
    auto t_pressure = getFTensor0FromVec(*(pressurePtr));

    const double l_mu = mU;
    // Extract matrix from data matrix
    for (auto gg = 0; gg != nb_gauss_pts; ++gg) {

      t_stress(i, j) = t_pressure * t_kd(i, j) + 2. * l_mu * t_strain(i, j);

      ++t_strain;
      ++t_stress;
      ++t_pressure;
    }

    MoFEMFunctionReturn(0);
  }

private:
  double mU;
  boost::shared_ptr<MatrixDouble> stressPtr;
  boost::shared_ptr<MatrixDouble> strainPtr;
  boost::shared_ptr<VectorDouble> pressurePtr;
};

//! [Run problem]
MoFEMErrorCode Incompressible::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR bC();
  CHKERR OPs();
  CHKERR tsSolve();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Set up problem]
MoFEMErrorCode Incompressible::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-geom_order", &geom_order,
                            PETSC_NULL);
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_discontinuous_pressure",
                             &isDiscontinuousPressure, PETSC_NULL);
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_a_t_petsc_fieldsplit",
                             &isATPetscFieldsplit, PETSC_NULL);                             
                             
  MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "Order " << order;
  MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "Geom order " << geom_order;

  // Select base
  enum bases { AINSWORTH, DEMKOWICZ, LASBASETOPT };
  const char *list_bases[LASBASETOPT] = {"ainsworth", "demkowicz"};
  PetscInt choice_base_value = AINSWORTH;
  CHKERR PetscOptionsGetEList(PETSC_NULL, NULL, "-base", list_bases,
                              LASBASETOPT, &choice_base_value, PETSC_NULL);

  FieldApproximationBase base;
  switch (choice_base_value) {
  case AINSWORTH:
    base = AINSWORTH_LEGENDRE_BASE;
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform)
        << "Set AINSWORTH_LEGENDRE_BASE for displacements";
    break;
  case DEMKOWICZ:
    base = DEMKOWICZ_JACOBI_BASE;
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform)
        << "Set DEMKOWICZ_JACOBI_BASE for displacements";
    break;
  default:
    base = LASTBASE;
    break;
  }

  // Note: For tets we have only H1 Ainsworth base, for Hex we have only H1
  // Demkowicz base. We need to implement Demkowicz H1 base on tet.
  CHKERR simple->addDataField("GEOMETRY", H1, base, SPACE_DIM);
  CHKERR simple->setFieldOrder("GEOMETRY", geom_order);

  // Adding fields related to incompressible elasticiy
  // Add displacement domain and boundary fields
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
  CHKERR simple->setFieldOrder("U", order);

  // Add pressure domain and boundary fields
  // Choose either Crouzeix-Raviart element:
  if (isDiscontinuousPressure) {
    CHKERR simple->addDomainField("P", L2, base, 1);
    CHKERR simple->setFieldOrder("P", order - 2);
  } else {
    // ... or Taylor-Hood element:
    CHKERR simple->addDomainField("P", H1, base, 1);
    CHKERR simple->setFieldOrder("P", order - 1);
  }

  // Add geometry data field
  CHKERR simple->addDataField("GEOMETRY", H1, base, SPACE_DIM);
  CHKERR simple->setFieldOrder("GEOMETRY", geom_order);

  auto get_skin = [&]() {
    Range body_ents;
    CHKERR mField.get_moab().get_entities_by_dimension(0, SPACE_DIM, body_ents);
    Skinner skin(&mField.get_moab());
    Range skin_ents;
    CHKERR skin.find_skin(0, body_ents, false, skin_ents);
    return skin_ents;
  };

  auto filter_blocks = [&](auto skin) {
    Range contact_range;
    for (auto m :
         mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

             (boost::format("%s(.*)") % "INCOMP_ELASTICITY").str()

                 ))

    ) {
      MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform)
          << "Find contact block set:  " << m->getName();
      auto meshset = m->getMeshset();
      CHKERR mField.get_moab().get_entities_by_dimension(meshset, SPACE_DIM - 1,
                                                         contact_range, true);

      MOFEM_LOG("SYNC", Sev::inform)
          << "Nb entities in contact surface: " << contact_range.size();
      MOFEM_LOG_SYNCHRONISE(mField.get_comm());
      CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
          contact_range);
      skin = intersect(skin, contact_range);
    }
    return skin;
  };

  auto filter_true_skin = [&](auto skin) {
    Range boundary_ents;
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    CHKERR pcomm->filter_pstatus(skin, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                 PSTATUS_NOT, -1, &boundary_ents);
    return boundary_ents;
  };

  CHKERR simple->setUp();

  auto project_ho_geometry = [&]() {
    Projection10NodeCoordsOnField ent_method(mField, "GEOMETRY");
    return mField.loop_dofs("GEOMETRY", ent_method);
  };
  CHKERR project_ho_geometry();

  MoFEMFunctionReturn(0);
} //! [Set up problem]

//! [Create common data]
MoFEMErrorCode Incompressible::createCommonData() {
  MoFEMFunctionBegin;

  auto get_options = [&]() {
    MoFEMFunctionBegin;
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-scale", &scale, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-young_modulus",
                                 &young_modulus, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-poisson_ratio",
                                 &poisson_ratio, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-rho", &rho, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-cn", &cn_contact,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-spring_stiffness",
                                 &spring_stiffness, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-vis_spring_stiffness",
                                 &vis_spring_stiffness, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-alpha_damping",
                                 &alpha_damping, PETSC_NULL);

    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "Young modulus " << young_modulus;
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "Poisson_ratio " << poisson_ratio;
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "Density " << rho;
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "cn_contact " << cn_contact;
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform)
        << "Spring stiffness " << spring_stiffness;
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform)
        << "Vis spring_stiffness " << vis_spring_stiffness;

    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "alpha_damping " << alpha_damping;

    PetscBool is_scale = PETSC_TRUE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_scale", &is_scale,
                               PETSC_NULL);
    // if (is_scale) {
    //   scale /= young_modulus;
    // }
    mu = young_modulus / (2. + 2. * poisson_ratio);
    const double lambda_denom = (1. + poisson_ratio ) * (1. - 2. * poisson_ratio);
    lambda = young_modulus * poisson_ratio / lambda_denom;
    cerr << "young_modulus " << young_modulus << "\n";
    
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "Scale " << scale;

    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_quasi_static",
                               &is_quasi_static, PETSC_NULL);
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform)
        << "Is quasi-static: " << (is_quasi_static ? "true" : "false");

    MoFEMFunctionReturn(0);
  };

  CHKERR get_options();

#ifdef PYTHON_SFD
  sdfPythonPtr = boost::make_shared<SDFPython>();
  CHKERR sdfPythonPtr->sdfInit("sdf.py");
  sdfPythonWeakPtr = sdfPythonPtr;
#endif

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Incompressible::bC() {
  MoFEMFunctionBegin;
  auto bc_mng = mField.getInterface<BcManager>();
  auto simple = mField.getInterface<Simple>();
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  
  for (auto f : {"U"}) {
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                             "REMOVE_X", f, 0, 0);
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                             "REMOVE_Y", f, 1, 1);
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                             "REMOVE_Z", f, 2, 2);
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                             "REMOVE_ALL", f, 0, 3);
  }

  // Note remove has to be always before push. Then node marking will be
  // corrupted.
  CHKERR bc_mng->pushMarkDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");
  
  auto time_scale = boost::make_shared<TimeScale>();
  CHKERR BoundaryNaturalBC::AddFluxToPipeline<OpForce>::add(
        pipeline_mng->getOpBoundaryRhsPipeline(), mField, "U", {},
        "FORCE", Sev::inform);

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pip]
MoFEMErrorCode Incompressible::OPs() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  auto *pip_mng = mField.getInterface<PipelineManager>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto time_scale = boost::make_shared<ScaledTimeScale>();
  auto body_force_time_scale =
      boost::make_shared<ScaledTimeScale>("body_force_hist.txt");

  auto integration_rule_vol = [](int, int, int approx_order) {
    return 2 * approx_order + geom_order - 1;
  };
  auto integration_rule_boundary = [](int, int, int approx_order) {
    return 2 * approx_order + geom_order - 1;
  };

  auto add_domain_base_ops = [&](auto &pip) {
    MoFEMFunctionBegin;
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1, HDIV},
                                                          "GEOMETRY");
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs = [&](auto &pip) {
    MoFEMFunctionBegin;
    
    //-------------------------------------------------------------------------

    //! [Operators used for incompressible elasticity]
    using OpGradSymTensorGrad =
        FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
            IT>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
    using OpMixScalarTimesDiv = FormsIntegrators<DomainEleOp>::Assembly<
        PETSC>::BiLinearForm<IT>::OpMixScalarTimesDiv<SPACE_DIM, coord_type>;
    //! [Operators used for incompressible elasticity]

    auto mat_D_ptr = boost::make_shared<MatrixDouble>();
    constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
    mat_D_ptr->resize(size_symm * size_symm, 1);

    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;

    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();

    auto t_mat = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mat_D_ptr);
    t_mat(i, j, k, l) = -2. * mu * ((t_kd(i, k) ^ t_kd(j, l)) / 4.);

    pip.push_back(new OpMixScalarTimesDiv(
        "P", "U", [](const double, const double, const double) { return -1.; },
        true, false));
    pip.push_back(new OpGradSymTensorGrad("U", "U", mat_D_ptr));

    auto get_lambda_reciprocal =
          [&](const double, const double, const double) {
            double rec_lambda = 1./lambda;
            if(isinf(rec_lambda))
              rec_lambda = 0.;
            
            return rec_lambda;
          };
      pip.push_back(new OpMassPressure("P", "P", get_lambda_reciprocal));

      if (AT != AssemblyType::SCHUR) {
        double eps_stab = 1e-4;
        CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-eps_stab", &eps_stab,
                                     PETSC_NULL);
        pip.push_back(new OpMassPressure(
            "P", "P", [eps_stab](double, double, double) { return eps_stab; }));
      }

    //-------------------------------------------------------------------------

    //! [Only used for dynamics]
    // using OpMass = FormsIntegrators<DomainEleOp>::Assembly<AT>::BiLinearForm<
    //     GAUSS>::OpMass<1, SPACE_DIM>;
    // //! [Only used for dynamics]
    // if (is_quasi_static == PETSC_FALSE) {

    //   auto *pip_mng = mField.getInterface<PipelineManager>();
    //   auto fe_domain_lhs = pip_mng->getDomainLhsFE();

    //   auto get_inertia_and_mass_damping =
    //       [this, fe_domain_lhs](const double, const double, const double) {
    //         return (rho * scale) * fe_domain_lhs->ts_aa +
    //                (alpha_damping * scale) * fe_domain_lhs->ts_a;
    //       };
    //   pip.push_back(new OpMass("U", "U", get_inertia_and_mass_damping));
    // } else {

    //   auto *pip_mng = mField.getInterface<PipelineManager>();
    //   auto fe_domain_lhs = pip_mng->getDomainLhsFE();

    //   auto get_inertia_and_mass_damping =
    //       [this, fe_domain_lhs](const double, const double, const double) {
    //         return (alpha_damping * scale) * fe_domain_lhs->ts_a;
    //       };
    //   pip.push_back(new OpMass("U", "U", get_inertia_and_mass_damping));
    // }

    // CHKERR HenckyOps::opFactoryDomainLhs<SPACE_DIM, AT, IT, DomainEleOp>(
    //     mField, pip, "U", "MAT_ELASTIC", Sev::verbose, scale);

    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    CHKERR DomainRhsBCs::AddFluxToPipeline<OpDomainRhsBCs>::add(
        pip, mField, "U", {body_force_time_scale}, Sev::inform);

    //! [Only used for dynamics]
    using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
        AT>::LinearForm<IT>::OpBaseTimesVector<1, SPACE_DIM, 1>;
    //! [Only used for dynamics]

    // only in case of dynamics
    // if (is_quasi_static == PETSC_FALSE) {
    //   auto mat_acceleration = boost::make_shared<MatrixDouble>();
    //   pip.push_back(new OpCalculateVectorFieldValuesDotDot<SPACE_DIM>(
    //       "U", mat_acceleration));
    //   pip.push_back(
    //       new OpInertiaForce("U", mat_acceleration, [](double, double, double) {
    //         return rho * scale;
    //       }));
    // }

    // // only in case of viscosity
    // if (alpha_damping > 0) {
    //   auto mat_velocity = boost::make_shared<MatrixDouble>();
    //   pip.push_back(
    //       new OpCalculateVectorFieldValuesDot<SPACE_DIM>("U", mat_velocity));
    //   pip.push_back(
    //       new OpInertiaForce("U", mat_velocity, [](double, double, double) {
    //         return alpha_damping * scale;
    //       }));
    // }

    auto pressure_ptr = boost::make_shared<VectorDouble>();
    pip.push_back(new OpCalculateScalarFieldValues("P", pressure_ptr));

    auto div_u_ptr = boost::make_shared<VectorDouble>();
    pip.push_back(
        new OpCalculateDivergenceVectorFieldValues<SPACE_DIM>("U", div_u_ptr));

    auto grad_u_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", grad_u_ptr));

    auto strain_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpSymmetrizeTensor<SPACE_DIM>("U", grad_u_ptr, strain_ptr));

    auto get_four_mu =
          [&](const double, const double, const double) {
            return - 2. * mu;
          };

    auto minus_one = [&](const double, const double, const double) {
      return -1.;
    };

    pip.push_back(new OpDomainGradTimesTensor(
        "U", strain_ptr, get_four_mu));

    pip.push_back(new OpDivDeltaUTimesP(
        "U", pressure_ptr, minus_one));

    pip.push_back(new OpBaseTimesScalarValues(
        "P", div_u_ptr, minus_one));
    
    auto get_lambda_reciprocal =
          [&](const double, const double, const double) {
            double rec_lambda = 1./lambda;
              if(isinf(rec_lambda))
                rec_lambda = 0.;

            return rec_lambda;
          };

    pip.push_back(new OpBaseTimesScalarValues(
        "P", pressure_ptr, get_lambda_reciprocal));

    // CHKERR HenckyOps::opFactoryDomainRhs<SPACE_DIM, AT, IT, DomainEleOp>(
    //     mField, pip, "U", "MAT_ELASTIC", Sev::inform, scale);

    // CHKERR ContactOps::opFactoryDomainRhs<SPACE_DIM, AT, IT, DomainEleOp>(
    //     pip, "SIGMA", "U");

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_base_ops = [&](auto &pip) {
    MoFEMFunctionBegin;
    CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(pip, {HDIV},
                                                              "GEOMETRY");
    // We have to integrate on curved face geometry, thus integration weight
    // have to adjusted.
    pip.push_back(new OpSetHOWeightsOnSubDim<SPACE_DIM>());
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_lhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    //! [Operators used for contact]
    using OpSpringLhs = FormsIntegrators<BoundaryEleOp>::Assembly<
        AT>::BiLinearForm<IT>::OpMass<1, SPACE_DIM>;
    //! [Operators used for contact]

    // Add Natural BCs to LHS
    CHKERR BoundaryLhsBCs::AddFluxToPipeline<OpBoundaryLhsBCs>::add(
        pip, mField, "U", Sev::inform);

    if (spring_stiffness > 0 || vis_spring_stiffness > 0) {

      auto *pip_mng = mField.getInterface<PipelineManager>();
      auto fe_boundary_lhs = pip_mng->getBoundaryLhsFE();

      pip.push_back(new OpSpringLhs(
          "U", "U",

          [this, fe_boundary_lhs](double, double, double) {
            return spring_stiffness * scale +
                   (vis_spring_stiffness * scale) * fe_boundary_lhs->ts_a;
          }

          ));
    }

    // CHKERR ContactOps::opFactoryBoundaryLhs<SPACE_DIM, AT, GAUSS,
    //                                         BoundaryEleOp>(pip, "SIGMA", "U");
    // CHKERR ContactOps::opFactoryBoundaryToDomainLhs<SPACE_DIM, AT, GAUSS,
    //                                                 DomainEle>(
    //     mField, pip, simple->getDomainFEName(), "SIGMA", "U", "GEOMETRY",
    //     integration_rule_vol);

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    //! [Operators used for contact]
    using OpSpringRhs = FormsIntegrators<BoundaryEleOp>::Assembly<
        AT>::LinearForm<IT>::OpBaseTimesVector<1, SPACE_DIM, 1>;
    //! [Operators used for contact]

    // Add Natural BCs to RHS
    CHKERR BoundaryRhsBCs::AddFluxToPipeline<OpBoundaryRhsBCs>::add(
        pip, mField, "U", {time_scale}, Sev::inform);

    if (spring_stiffness > 0 || vis_spring_stiffness > 0) {
      auto u_disp = boost::make_shared<MatrixDouble>();
      auto dot_u_disp = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_disp));
      pip.push_back(
          new OpCalculateVectorFieldValuesDot<SPACE_DIM>("U", dot_u_disp));
      pip.push_back(
          new OpSpringRhs("U", u_disp, [this](double, double, double) {
            return spring_stiffness * scale;
          }));
      pip.push_back(
          new OpSpringRhs("U", dot_u_disp, [this](double, double, double) {
            return vis_spring_stiffness * scale;
          }));
    }

    CHKERR ContactOps::opFactoryBoundaryRhs<SPACE_DIM, AT, GAUSS,
                                            BoundaryEleOp>(pip, "SIGMA", "U");

    MoFEMFunctionReturn(0);
  };

  CHKERR add_domain_base_ops(pip_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_base_ops(pip_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_ops_lhs(pip_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_ops_rhs(pip_mng->getOpDomainRhsPipeline());

  CHKERR add_boundary_base_ops(pip_mng->getOpBoundaryLhsPipeline());
  CHKERR add_boundary_base_ops(pip_mng->getOpBoundaryRhsPipeline());
  // CHKERR add_boundary_ops_lhs(pip_mng->getOpBoundaryLhsPipeline());
  // CHKERR add_boundary_ops_rhs(pip_mng->getOpBoundaryRhsPipeline());

  CHKERR pip_mng->setDomainRhsIntegrationRule(integration_rule_vol);
  CHKERR pip_mng->setDomainLhsIntegrationRule(integration_rule_vol);
  CHKERR pip_mng->setBoundaryRhsIntegrationRule(integration_rule_boundary);
  CHKERR pip_mng->setBoundaryLhsIntegrationRule(integration_rule_boundary);

  reactionFe = boost::make_shared<DomainEle>(mField);

  MoFEMFunctionReturn(0);
}
//! [Push operators to pip]

//! [Solve]
struct SetUpSchur {
  static boost::shared_ptr<SetUpSchur>
  createSetUpSchur(MoFEM::Interface &m_field);
  virtual MoFEMErrorCode setUp(SmartPetscObj<TS> solver) = 0;

protected:
  SetUpSchur() = default;
};

MoFEMErrorCode Incompressible::tsSolve() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pip_mng = mField.getInterface<PipelineManager>();
  ISManager *is_manager = mField.getInterface<ISManager>();

  auto set_section_monitor = [&](auto solver) {
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

  auto scatter_create = [&](auto D, auto coeff) {
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

  auto create_post_process_elements = [&]() {
    auto pp_fe = boost::make_shared<PostProcEle>(mField);
    auto &pip = pp_fe->getOpPtrVector();

    auto get_four_mu =
          [&](const double, const double, const double) {
            return 4. * mu;
          };

    auto get_lambda_reciprocal =
          [&](const double, const double, const double) {
            double rec_lambda = -1./lambda;
            if(isinf(rec_lambda))
                rec_lambda = 0.;
            return rec_lambda;
          };

    auto push_vol_ops = [this](auto &pip) {
      MoFEMFunctionBegin;
      CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1},
                                                            "GEOMETRY");

      MoFEMFunctionReturn(0);
    };

    auto push_vol_post_proc_ops = [this](auto &pp_fe, auto &&p) {
      MoFEMFunctionBegin;

      auto &pip = pp_fe->getOpPtrVector();

      // auto [common_plastic_ptr, common_henky_ptr] = p;

      using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

      auto x_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(
          new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", x_ptr));
      auto u_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

      auto pressure_ptr = boost::make_shared<VectorDouble>();
      pip.push_back(new OpCalculateScalarFieldValues("P", pressure_ptr));

      auto div_u_ptr = boost::make_shared<VectorDouble>();
      pip.push_back(new OpCalculateDivergenceVectorFieldValues<SPACE_DIM>(
          "U", div_u_ptr));

      auto grad_u_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
          "U", grad_u_ptr));

      auto strain_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", grad_u_ptr, strain_ptr));

      auto stress_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateLameStress<SPACE_DIM>(mu, stress_ptr,
                                                         strain_ptr, pressure_ptr));

      pip.push_back(

          new OpPPMap(

              pp_fe->getPostProcMesh(), pp_fe->getMapGaussPts(),

              {{"P", pressure_ptr}},

              {{"U", u_ptr}, {"GEOMETRY", x_ptr}},

              {},

              {{"STRAIN", strain_ptr}, {"STRESS", stress_ptr}}

              )

      );

      MoFEMFunctionReturn(0);
    };

    auto vol_post_proc = [this, push_vol_post_proc_ops, push_vol_ops]() {
      PetscBool post_proc_vol = PETSC_FALSE;
      CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-post_proc_vol",
                                 &post_proc_vol, PETSC_NULL);
      if (post_proc_vol == PETSC_FALSE)
        return boost::shared_ptr<PostProcEle>();
      auto pp_fe = boost::make_shared<PostProcEle>(mField);
      CHK_MOAB_THROW(
          push_vol_post_proc_ops(pp_fe, push_vol_ops(pp_fe->getOpPtrVector())),
          "push_vol_post_proc_ops");
      return pp_fe;
    };

    auto skin_post_proc = [this, push_vol_post_proc_ops, push_vol_ops]() {
      PetscBool post_proc_skin = PETSC_TRUE;
      CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-post_proc_skin",
                                 &post_proc_skin, PETSC_NULL);
      if (post_proc_skin == PETSC_FALSE)
        return boost::shared_ptr<SkinPostProcEle>();

      auto simple = mField.getInterface<Simple>();
      auto pp_fe = boost::make_shared<SkinPostProcEle>(mField);
      auto op_side = new OpLoopSide<DomainEle>(
          mField, simple->getDomainFEName(), SPACE_DIM, Sev::verbose);
      pp_fe->getOpPtrVector().push_back(op_side);
      CHK_MOAB_THROW(push_vol_post_proc_ops(
                         pp_fe, push_vol_ops(op_side->getOpPtrVector())),
                     "push_vol_post_proc_ops");
      return pp_fe;
    };

    return std::make_pair(vol_post_proc(), skin_post_proc());
  };

  auto set_time_monitor = [&](auto dm, auto solver) {
    MoFEMFunctionBegin;
    boost::shared_ptr<MonitorIncompressible> monitor_ptr(
        new MonitorIncompressible(dm, create_post_process_elements(), reactionFe, uXScatter,
                    uYScatter, uZScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  auto set_essential_bc = [&]() {
    MoFEMFunctionBegin;
    // This is low level pushing finite elements (pipelines) to solver
    auto ts_ctx_ptr = getDMTsCtx(simple->getDM());
    auto pre_proc_ptr = boost::make_shared<FEMethod>();
    auto post_proc_rhs_ptr = boost::make_shared<FEMethod>();
    auto post_proc_lhs_ptr = boost::make_shared<FEMethod>();

    // Add boundary condition scaling
    auto time_scale = boost::make_shared<TimeScale>();

    auto get_bc_hook_rhs = [&]() {
      EssentialPreProc<DisplacementCubitBcData> hook(mField, pre_proc_ptr,
                                                     {time_scale}, false);
      return hook;
    };
    pre_proc_ptr->preProcessHook = get_bc_hook_rhs();

    auto get_post_proc_hook_rhs = [&]() {
      return EssentialPostProcRhs<DisplacementCubitBcData>(
          mField, post_proc_rhs_ptr, 1.);
    };
    auto get_post_proc_hook_lhs = [&]() {
      return EssentialPostProcLhs<DisplacementCubitBcData>(
          mField, post_proc_lhs_ptr, 1.);
    };
    post_proc_rhs_ptr->postProcessHook = get_post_proc_hook_rhs();

    ts_ctx_ptr->getPreProcessIFunction().push_front(pre_proc_ptr);
    ts_ctx_ptr->getPreProcessIJacobian().push_front(pre_proc_ptr);
    ts_ctx_ptr->getPostProcessIFunction().push_back(post_proc_rhs_ptr);
    if (AT != AssemblyType::SCHUR) {
      post_proc_lhs_ptr->postProcessHook = get_post_proc_hook_lhs();
      ts_ctx_ptr->getPostProcessIJacobian().push_back(post_proc_lhs_ptr);
    }
    MoFEMFunctionReturn(0);
  };

  auto set_fieldsplit_preconditioner_ksp = [&](auto ksp) {
    MoFEMFunctionBeginHot;
    PC pc;
    CHKERR KSPGetPC(ksp, &pc);
    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);
    if (is_pcfs == PETSC_TRUE) {
      auto bc_mng = mField.getInterface<BcManager>();
      auto name_prb = simple->getProblemName();
      SmartPetscObj<IS> is_u;
      CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
          name_prb, ROW, "U", 0, SPACE_DIM, is_u);
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL, is_u);
    }
    MoFEMFunctionReturnHot(0);
  };

  auto set_fieldsplit_preconditioner_ts = [&](auto solver) {
    MoFEMFunctionBeginHot;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    CHKERR set_fieldsplit_preconditioner_ksp(ksp);
    MoFEMFunctionReturnHot(0);
  };


  auto set_schur_pc = [&](auto solver) {
    boost::shared_ptr<SetUpSchur> schur_ptr;
    if (AT == AssemblyType::SCHUR) {
      schur_ptr = SetUpSchur::createSetUpSchur(mField);
      CHKERR schur_ptr->setUp(solver);
    } else {
      if(isATPetscFieldsplit)
        CHKERR set_fieldsplit_preconditioner_ts(solver);
    }
    return schur_ptr;
  };

  auto dm = simple->getDM();
  auto D = createDMVector(dm);

  ContactOps::CommonData::createTotalTraction(mField);

  uXScatter = scatter_create(D, 0);
  uYScatter = scatter_create(D, 1);
  if (SPACE_DIM == 3)
    uZScatter = scatter_create(D, 2);

  // Add extra finite elements to SNES solver pipelines to resolve essential
  // boundary conditions
  CHKERR set_essential_bc();

  if (is_quasi_static == PETSC_TRUE) {
    auto solver = pip_mng->createTSIM();
    CHKERR TSSetFromOptions(solver);

    auto D = createDMVector(dm);
    
    CHKERR set_section_monitor(solver);
    CHKERR set_time_monitor(dm, solver);
    CHKERR TSSetSolution(solver, D);
    auto schur_pc_ptr = set_schur_pc(solver);

    CHKERR TSSetUp(solver);
    CHKERR TSSolve(solver, NULL);
  } else {
    auto solver = pip_mng->createTSIM2();
    CHKERR TSSetFromOptions(solver);

    auto dm = simple->getDM();
    auto D = createDMVector(dm);
    auto DD = vectorDuplicate(D);
    auto schur_pc_ptr = set_schur_pc(solver);

    CHKERR set_section_monitor(solver);
    CHKERR set_time_monitor(dm, solver);
    CHKERR TS2SetSolution(solver, D, DD);
    CHKERR TSSetUp(solver);
    CHKERR TSSolve(solver, NULL);
  }

  ContactOps::CommonData::totalTraction.reset();

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Check]
MoFEMErrorCode Incompressible::checkResults() { return 0; }
//! [Check]

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

#ifdef PYTHON_SFD
  Py_Initialize();
#endif

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for CONTACT
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "INCOMP_ELASTICITY"));
  LogManager::setLog("INCOMP_ELASTICITY");
  MOFEM_LOG_TAG("INCOMP_ELASTICITY", "Indent");

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

    //! [Load mesh]
    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();
    CHKERR simple->loadFile("");
    //! [Load mesh]

    //! [CONTACT]
    Incompressible ex(m_field);
    CHKERR ex.runProblem();
    //! [CONTACT]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();

#ifdef PYTHON_SFD
  if (Py_FinalizeEx() < 0) {
    exit(120);
  }
#endif

  return 0;
}

struct SetUpSchurImpl : public SetUpSchur {

  SetUpSchurImpl(MoFEM::Interface &m_field) : SetUpSchur(), mField(m_field) {}

  virtual ~SetUpSchurImpl() {
    A.reset();
    P.reset();
    S.reset();
  }

  MoFEMErrorCode setUp(SmartPetscObj<TS> solver);

private:
  MoFEMErrorCode setEntities();
  MoFEMErrorCode setOperator();
  MoFEMErrorCode setPC(PC pc);

  SmartPetscObj<DM> createSubDM();

  SmartPetscObj<Mat> A;
  SmartPetscObj<Mat> P;
  SmartPetscObj<Mat> S;

  MoFEM::Interface &mField;

  SmartPetscObj<DM> subDM;
};

MoFEMErrorCode SetUpSchurImpl::setUp(SmartPetscObj<TS> solver) {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>();
  auto dm = simple->getDM();

  SNES snes;
  CHKERR TSGetSNES(solver, &snes);
  KSP ksp;
  CHKERR SNESGetKSP(snes, &ksp);
  CHKERR KSPSetFromOptions(ksp);

  PC pc;
  CHKERR KSPGetPC(ksp, &pc);

  PetscBool is_pcfs = PETSC_FALSE;
  PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);
  if (is_pcfs) {

    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "Setup Schur pc";

    if (A || P || S) {
      CHK_THROW_MESSAGE(
          MOFEM_DATA_INCONSISTENCY,
          "Is expected that schur matrix is not allocated. This is "
          "possible only is PC is set up twice");
    }

    A = createDMMatrix(dm);
    P = matDuplicate(A, MAT_DO_NOT_COPY_VALUES);
    subDM = createSubDM();
    S = createDMMatrix(subDM);
    CHKERR MatSetBlockSize(S, SPACE_DIM);

    auto ts_ctx_ptr = getDMTsCtx(dm);
    CHKERR TSSetIJacobian(solver, A, P, TsSetIJacobian, ts_ctx_ptr.get());

    CHKERR setOperator();
    CHKERR setPC(pc);

  } else {
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "No Schur pc";

    pip->getOpBoundaryLhsPipeline().push_front(new OpSchurAssembleBegin());
    pip->getOpBoundaryLhsPipeline().push_back(
        new OpSchurAssembleEnd<SCHUR_DGESV>({}, {}, {}, {}, {}));
    pip->getOpDomainLhsPipeline().push_front(new OpSchurAssembleBegin());
    pip->getOpDomainLhsPipeline().push_back(
        new OpSchurAssembleEnd<SCHUR_DGESV>({}, {}, {}, {}, {}));

    auto post_proc_schur_lhs_ptr = boost::make_shared<FEMethod>();
    post_proc_schur_lhs_ptr->postProcessHook = [this,
                                                post_proc_schur_lhs_ptr]() {
      MoFEMFunctionBegin;
      CHKERR EssentialPostProcLhs<DisplacementCubitBcData>(
          mField, post_proc_schur_lhs_ptr, 1.)();
      MoFEMFunctionReturn(0);
    };
    auto ts_ctx_ptr = getDMTsCtx(dm);
    ts_ctx_ptr->getPostProcessIJacobian().push_back(post_proc_schur_lhs_ptr);
  }
  MoFEMFunctionReturn(0);
}

SmartPetscObj<DM> SetUpSchurImpl::createSubDM() {
  auto simple = mField.getInterface<Simple>();
  auto sub_dm = createDM(mField.get_comm(), "DMMOFEM");
  auto set_up = [&]() {
    MoFEMFunctionBegin;
    CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "SUB");
    CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
    CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
    CHKERR DMMoFEMAddSubFieldRow(sub_dm, "U");
    CHKERR DMSetUp(sub_dm);
    MoFEMFunctionReturn(0);
  };
  CHK_THROW_MESSAGE(set_up(), "sey up dm");
  return sub_dm;
}

MoFEMErrorCode SetUpSchurImpl::setOperator() {
  MoFEMFunctionBegin;

  double eps_stab = 1e-4;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-eps_stab", &eps_stab,
                               PETSC_NULL);

  using B =
      FormsIntegrators<BoundaryEleOpStab>::Assembly<SCHUR>::BiLinearForm<IT>;
  using OpMassStab = B::OpMass<1, SPACE_DIM * SPACE_DIM>;

  auto pip = mField.getInterface<PipelineManager>();
  // Boundary
  auto dm_is = getDMSubData(subDM)->getSmartRowIs();
  auto ao_up = createAOMappingIS(dm_is, PETSC_NULL);

  pip->getOpBoundaryLhsPipeline().push_front(new OpSchurAssembleBegin());
  pip->getOpBoundaryLhsPipeline().push_back(
      new OpMassStab("P", "P",
                     [eps_stab](double, double, double) { return eps_stab; }));
  pip->getOpBoundaryLhsPipeline().push_back(new OpSchurAssembleEnd<SCHUR_DGESV>(
      {"P"}, {nullptr}, {ao_up}, {S}, {false}, false));

  // Domain
  pip->getOpDomainLhsPipeline().push_front(new OpSchurAssembleBegin());
  pip->getOpDomainLhsPipeline().push_back(new OpSchurAssembleEnd<SCHUR_DGESV>(
      {"P"}, {nullptr}, {ao_up}, {S}, {false}, false));

  auto pre_proc_schur_lhs_ptr = boost::make_shared<FEMethod>();
  auto post_proc_schur_lhs_ptr = boost::make_shared<FEMethod>();

  pre_proc_schur_lhs_ptr->preProcessHook = [this]() {
    MoFEMFunctionBegin;
    CHKERR MatZeroEntries(A);
    CHKERR MatZeroEntries(P);
    CHKERR MatZeroEntries(S);
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::verbose) << "Lhs Assemble Begin";
    MoFEMFunctionReturn(0);
  };

  post_proc_schur_lhs_ptr->postProcessHook = [this, ao_up,
                                              post_proc_schur_lhs_ptr]() {
    MoFEMFunctionBegin;
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::verbose) << "Lhs Assemble End";

    *post_proc_schur_lhs_ptr->matAssembleSwitch = false;

    auto print_mat_norm = [this](auto a, std::string prefix) {
      MoFEMFunctionBegin;
      double nrm;
      CHKERR MatNorm(a, NORM_FROBENIUS, &nrm);
      MOFEM_LOG("INCOMP_ELASTICITY", Sev::noisy) << prefix << " norm = " << nrm;
      MoFEMFunctionReturn(0);
    };

    CHKERR MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    CHKERR EssentialPostProcLhs<DisplacementCubitBcData>(
        mField, post_proc_schur_lhs_ptr, 1, A)();

    CHKERR MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
    CHKERR MatAXPY(P, 1, A, SAME_NONZERO_PATTERN);

    CHKERR MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);

    CHKERR EssentialPostProcLhs<DisplacementCubitBcData>(
        mField, post_proc_schur_lhs_ptr, 1, S, ao_up)();

#ifndef NDEBUG
    CHKERR print_mat_norm(A, "A");
    CHKERR print_mat_norm(P, "P");
    CHKERR print_mat_norm(S, "S");
#endif // NDEBUG

    MOFEM_LOG("INCOMP_ELASTICITY", Sev::verbose) << "Lhs Assemble Finish";
    MoFEMFunctionReturn(0);
  };

  auto simple = mField.getInterface<Simple>();
  auto ts_ctx_ptr = getDMTsCtx(simple->getDM());
  ts_ctx_ptr->getPreProcessIJacobian().push_front(pre_proc_schur_lhs_ptr);
  ts_ctx_ptr->getPostProcessIJacobian().push_back(post_proc_schur_lhs_ptr);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SetUpSchurImpl::setPC(PC pc) {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  SmartPetscObj<IS> is;
  mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
      simple->getProblemName(), ROW, "P", 0, 1, is);
  CHKERR PCFieldSplitSetIS(pc, NULL, is);
  CHKERR PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, S);
  MoFEMFunctionReturn(0);
}

boost::shared_ptr<SetUpSchur>
SetUpSchur::createSetUpSchur(MoFEM::Interface &m_field) {
  return boost::shared_ptr<SetUpSchur>(new SetUpSchurImpl(m_field));
}
