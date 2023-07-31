/**
 * \file contact.cpp
 * \CONTACT contact.cpp
 *
 * CONTACT of contact problem
 *
 * @copyright Copyright (c) 2023
 */

/* The above code is a preprocessor directive in C++ that checks if the macro
"EXECUTABLE_DIMENSION" has been defined. If it has not been defined, it is set
to 3" */
#ifndef EXECUTABLE_DIMENSION
#define EXECUTABLE_DIMENSION 3
#endif

#include <MoFEM.hpp>
#include <MatrixFunction.hpp>

#ifdef PYTHON_SFD
#include <boost/python.hpp>
#include <boost/python/def.hpp>
namespace bp = boost::python;
#endif

using namespace MoFEM;

constexpr AssemblyType AT = AssemblyType::SCHUR; //< selected assembly type
constexpr IntegrationType IT =
    IntegrationType::GAUSS;                      //< selected integration type

template <int DIM> struct ElementsAndOps;

template <> struct ElementsAndOps<2> : PipelineManager::ElementsAndOpsByDim<2> {
  static constexpr FieldSpace CONTACT_SPACE = HCURL;
};

template <> struct ElementsAndOps<3> : PipelineManager::ElementsAndOpsByDim<3> {
  static constexpr FieldSpace CONTACT_SPACE = HDIV;
};

constexpr FieldSpace ElementsAndOps<2>::CONTACT_SPACE;
constexpr FieldSpace ElementsAndOps<3>::CONTACT_SPACE;

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

/* The above code is defining an alias `EntData` for the type
`EntitiesFieldData::EntData`. This is a C++ syntax for creating a new name for
an existing type. */
using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;

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

struct BoundaryEleOpStab : public BoundaryEleOp {
  using BoundaryEleOp::BoundaryEleOp;
};

template <>
typename MoFEM::OpBaseImpl<AT, BoundaryEleOpStab>::MatSetValuesHook
    MoFEM::OpBaseImpl<AT, BoundaryEleOpStab>::matSetValuesHook =
        [](ForcesAndSourcesCore::UserDataOperator *op_ptr,
           const EntitiesFieldData::EntData &row_data,
           const EntitiesFieldData::EntData &col_data, MatrixDouble &m) {
          return MatSetValues<AssemblyTypeSelector<AT>>(
              op_ptr->getKSPB(), row_data, col_data, m, ADD_VALUES);
        };

constexpr FieldSpace CONTACT_SPACE = ElementsAndOps<SPACE_DIM>::CONTACT_SPACE;

//! [Operators used for contact]
using OpSpringLhs = FormsIntegrators<BoundaryEleOp>::Assembly<AT>::BiLinearForm<
    IT>::OpMass<1, SPACE_DIM>;
using OpSpringRhs = FormsIntegrators<BoundaryEleOp>::Assembly<AT>::LinearForm<
    IT>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Operators used for contact]

//! [Only used for dynamics]
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<AT>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<AT>::LinearForm<
    IT>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Only used for dynamics]

namespace ContactOps {

double cn_contact = 0.1;

struct DomainBCs {};
struct BoundaryBCs {};

using DomainRhsBCs = NaturalBC<DomainEleOp>::Assembly<AT>::LinearForm<IT>;
using OpDomainRhsBCs = DomainRhsBCs::OpFlux<DomainBCs, 1, SPACE_DIM>;
using BoundaryRhsBCs = NaturalBC<BoundaryEleOp>::Assembly<AT>::LinearForm<IT>;
using OpBoundaryRhsBCs = BoundaryRhsBCs::OpFlux<BoundaryBCs, 1, SPACE_DIM>;
using BoundaryLhsBCs = NaturalBC<BoundaryEleOp>::Assembly<AT>::BiLinearForm<IT>;
using OpBoundaryLhsBCs = BoundaryLhsBCs::OpFlux<BoundaryBCs, 1, SPACE_DIM>;

}; // namespace ContactOps

constexpr bool is_quasi_static = true;

int order = 2;
int geom_order = 1;
double young_modulus = 100;
double poisson_ratio = 0.25;
double rho = 0.0;
double spring_stiffness = 0.5;
double alpha_damping = 0;

#include <HenckyOps.hpp>
using namespace HenckyOps;
#include <ContactOps.hpp>
#include <PostProcContact.hpp>
#include <ContactNaturalDomainBC.hpp>
#include <ContactNaturalBoundaryBC.hpp>

using namespace ContactOps;

struct Contact {

  Contact(MoFEM::Interface &m_field) : mField(m_field) {}

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
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;

#ifdef PYTHON_SFD
  boost::shared_ptr<SDFPython> sdfPythonPtr;
#endif
};

//! [Run problem]
MoFEMErrorCode Contact::runProblem() {
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
MoFEMErrorCode Contact::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-geom_order", &geom_order,
                            PETSC_NULL);
  MOFEM_LOG("CONTACT", Sev::inform) << "Order " << order;
  MOFEM_LOG("CONTACT", Sev::inform) << "Geom order " << geom_order;

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
    MOFEM_LOG("CONTACT", Sev::inform)
        << "Set AINSWORTH_LEGENDRE_BASE for displacements";
    break;
  case DEMKOWICZ:
    base = DEMKOWICZ_JACOBI_BASE;
    MOFEM_LOG("CONTACT", Sev::inform)
        << "Set DEMKOWICZ_JACOBI_BASE for displacements";
    break;
  default:
    base = LASTBASE;
    break;
  }

  // Note: For tets we have only H1 Ainsworth base, for Hex we have only H1
  // Demkowicz base. We need to implement Demkowicz H1 base on tet.
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
  CHKERR simple->addDomainField("SIGMA", CONTACT_SPACE, DEMKOWICZ_JACOBI_BASE,
                                SPACE_DIM);
  CHKERR simple->addBoundaryField("SIGMA", CONTACT_SPACE, DEMKOWICZ_JACOBI_BASE,
                                  SPACE_DIM);
  CHKERR simple->addDataField("GEOMETRY", H1, base, SPACE_DIM);

  CHKERR simple->setFieldOrder("U", order);
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

             (boost::format("%s(.*)") % "CONTACT").str()

                 ))

    ) {
      MOFEM_LOG("CONTACT", Sev::inform)
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

  auto boundary_ents = filter_true_skin(filter_blocks(get_skin()));
  CHKERR simple->setFieldOrder("SIGMA", 0);
  CHKERR simple->setFieldOrder("SIGMA", order - 1, &boundary_ents);

  CHKERR simple->setUp();

  auto project_ho_geometry = [&]() {
    Projection10NodeCoordsOnField ent_method(mField, "GEOMETRY");
    return mField.loop_dofs("GEOMETRY", ent_method);
  };
  CHKERR project_ho_geometry();

  MoFEMFunctionReturn(0);
} //! [Set up problem]

//! [Create common data]
MoFEMErrorCode Contact::createCommonData() {
  MoFEMFunctionBegin;

  auto get_options = [&]() {
    MoFEMFunctionBegin;
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-young_modulus",
                                 &young_modulus, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-poisson_ratio",
                                 &poisson_ratio, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-rho", &rho, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-cn", &cn_contact,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-spring_stiffness",
                                 &spring_stiffness, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-alpha_damping",
                                 &alpha_damping, PETSC_NULL);

    MOFEM_LOG("CONTACT", Sev::inform) << "Young modulus " << young_modulus;
    MOFEM_LOG("CONTACT", Sev::inform) << "Poisson_ratio " << poisson_ratio;
    MOFEM_LOG("CONTACT", Sev::inform) << "Density " << rho;
    MOFEM_LOG("CONTACT", Sev::inform) << "cn_contact " << cn_contact;
    MOFEM_LOG("CONTACT", Sev::inform)
        << "spring_stiffness " << spring_stiffness;
    MOFEM_LOG("CONTACT", Sev::inform) << "alpha_damping " << alpha_damping;

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
MoFEMErrorCode Contact::bC() {
  MoFEMFunctionBegin;
  auto bc_mng = mField.getInterface<BcManager>();
  auto simple = mField.getInterface<Simple>();

  for (auto f : {"U", "SIGMA"}) {
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                             "REMOVE_X", f, 0, 0);
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                             "REMOVE_Y", f, 1, 1);
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                             "REMOVE_Z", f, 2, 2);
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                             "REMOVE_ALL", f, 0, 3);
  }

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX_X",
                                           "SIGMA", 0, 0, false, true);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX_Y",
                                           "SIGMA", 1, 1, false, true);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX_Z",
                                           "SIGMA", 2, 2, false, true);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX_ALL",
                                           "SIGMA", 0, 3, false, true);
  CHKERR bc_mng->removeBlockDOFsOnEntities(
      simple->getProblemName(), "NO_CONTACT", "SIGMA", 0, 3, false, true);

  // Note remove has to be always before push. Then node marking will be
  // corrupted.
  CHKERR bc_mng->pushMarkDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");
  boundaryMarker =
      bc_mng->getMergedBlocksMarker(vector<string>{"FIX_", "ROTATE_"});

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pip]
MoFEMErrorCode Contact::OPs() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  auto *pip_mng = mField.getInterface<PipelineManager>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto time_scale = boost::make_shared<TimeScale>();

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

  auto henky_common_data_ptr = boost::make_shared<HenckyOps::CommonData>();
  henky_common_data_ptr->matDPtr = boost::make_shared<MatrixDouble>();
  henky_common_data_ptr->matGradPtr = boost::make_shared<MatrixDouble>();

  auto add_domain_ops_lhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    if (!is_quasi_static) {
      auto get_inertia_and_mass_damping = [this](const double, const double,
                                                 const double) {
        auto *pip_mng = mField.getInterface<PipelineManager>();
        auto &fe_domain_lhs = pip_mng->getDomainLhsFE();
        return rho * fe_domain_lhs->ts_aa + alpha_damping * fe_domain_lhs->ts_a;
      };
      pip.push_back(new OpMass("U", "U", get_inertia_and_mass_damping));
    } else if (alpha_damping > 0) {
      auto get_mass_damping = [this](const double, const double, const double) {
        auto *pip_mng = mField.getInterface<PipelineManager>();
        auto &fe_domain_lhs = pip_mng->getDomainLhsFE();
        return alpha_damping * fe_domain_lhs->ts_a;
      };
    }

    CHKERR HenckyOps::opFactoryDomainLhs<SPACE_DIM, AT, IT, DomainEleOp>(
        mField, pip, "U", "MAT_ELASTIC", Sev::verbose);

    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    CHKERR DomainRhsBCs::AddFluxToPipeline<OpDomainRhsBCs>::add(
        pip, mField, "U", {time_scale}, Sev::inform);

    // only in case of dynamics
    if (!is_quasi_static) {
      auto mat_acceleration = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldValuesDotDot<SPACE_DIM>(
          "U", mat_acceleration));
      pip.push_back(new OpInertiaForce(
          "U", mat_acceleration, [](double, double, double) { return rho; }));
    }
    if (alpha_damping > 0) {
      auto mat_velocity = boost::make_shared<MatrixDouble>();
      pip.push_back(
          new OpCalculateVectorFieldValuesDot<SPACE_DIM>("U", mat_velocity));
      pip.push_back(
          new OpInertiaForce("U", mat_velocity, [](double, double, double) {
            return alpha_damping;
          }));
    }

    CHKERR
    HenckyOps::opFactoryDomainRhs<SPACE_DIM, AT, IT, DomainEleOp>(
        mField, pip, "U", "MAT_ELASTIC", Sev::inform);

    CHKERR ContactOps::opFactoryDomainRhs<SPACE_DIM, AT, IT, DomainEleOp>(
        pip, "SIGMA", "U");

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

    if (spring_stiffness > 0)
      pip.push_back(new OpSpringLhs(
          "U", "U",

          [this](double, double, double) { return spring_stiffness; }

          ));

    CHKERR ContactOps::opFactoryBoundaryLhs<SPACE_DIM, AT, GAUSS,
                                            BoundaryEleOp>(pip, "SIGMA", "U");
    CHKERR ContactOps::opFactoryBoundaryToDomainLhs<SPACE_DIM, AT, GAUSS,
                                                    DomainEle>(
        mField, pip, simple->getDomainFEName(), "SIGMA", "U", "GEOMETRY",
        integration_rule_vol);

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    if (spring_stiffness > 0) {
      auto u_disp = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_disp));
      pip.push_back(
          new OpSpringRhs("U", u_disp, [this](double, double, double) {
            return spring_stiffness;
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
  CHKERR add_boundary_ops_lhs(pip_mng->getOpBoundaryLhsPipeline());
  CHKERR add_boundary_ops_rhs(pip_mng->getOpBoundaryRhsPipeline());

  CHKERR pip_mng->setDomainRhsIntegrationRule(integration_rule_vol);
  CHKERR pip_mng->setDomainLhsIntegrationRule(integration_rule_vol);
  CHKERR pip_mng->setBoundaryRhsIntegrationRule(integration_rule_boundary);
  CHKERR pip_mng->setBoundaryLhsIntegrationRule(integration_rule_boundary);

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

MoFEMErrorCode Contact::tsSolve() {
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

  auto set_time_monitor = [&](auto dm, auto solver) {
    MoFEMFunctionBegin;
    boost::shared_ptr<Monitor> monitor_ptr(
        new Monitor(dm, uXScatter, uYScatter, uZScatter));
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
      return EssentialPreProcRhs<DisplacementCubitBcData>(
          mField, post_proc_rhs_ptr, 1.);
    };
    auto get_post_proc_hook_lhs = [&]() {
      return EssentialPreProcLhs<DisplacementCubitBcData>(
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

  auto set_schur_pc = [&](auto solver) {
    boost::shared_ptr<SetUpSchur> schur_ptr;
    if (AT == AssemblyType::SCHUR) {
      schur_ptr = SetUpSchur::createSetUpSchur(mField);
      CHKERR schur_ptr->setUp(solver);
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

  if (is_quasi_static) {
    auto solver = pip_mng->createTSIM();
    CHKERR TSSetFromOptions(solver);

    auto D = createDMVector(dm);
    auto schur_pc_ptr = set_schur_pc(solver);

    CHKERR set_section_monitor(solver);
    CHKERR set_time_monitor(dm, solver);
    CHKERR TSSetSolution(solver, D);
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
MoFEMErrorCode Contact::checkResults() { return 0; }
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
      LogManager::createSink(LogManager::getStrmWorld(), "CONTACT"));
  LogManager::setLog("CONTACT");
  MOFEM_LOG_TAG("CONTACT", "Indent");

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
    Contact ex(m_field);
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
}

struct SetUpSchurImpl : public SetUpSchur {

  SetUpSchurImpl(MoFEM::Interface &m_field) : SetUpSchur(), mField(m_field) {}

  virtual ~SetUpSchurImpl() {
    AT.reset();
    P.reset();
    S.reset();
  }

  MoFEMErrorCode setUp(SmartPetscObj<TS> solver);

private:
  MoFEMErrorCode setEntities();
  MoFEMErrorCode setOperator();
  MoFEMErrorCode setPC(PC pc);

  SmartPetscObj<DM> createSubDM();

  SmartPetscObj<Mat> AT;
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

    MOFEM_LOG("CONTACT", Sev::inform) << "Setup Schur pc";

    if (AT || P || S) {
      CHK_THROW_MESSAGE(
          MOFEM_DATA_INCONSISTENCY,
          "Is expected that schur matrix is not allocated. This is "
          "possible only is PC is set up twice");
    }

    AT = createDMMatrix(dm);
    P = matDuplicate(AT, MAT_DO_NOT_COPY_VALUES);
    CHKERR MatSetBlockSize(AT, SPACE_DIM);
    CHKERR MatSetBlockSize(P, SPACE_DIM);
    subDM = createSubDM();
    S = createDMMatrix(subDM);
    CHKERR MatSetBlockSize(S, SPACE_DIM);

    auto ts_ctx_ptr = getDMTsCtx(dm);
    CHKERR TSSetIJacobian(solver, AT, P, TsSetIJacobian, ts_ctx_ptr.get());

    CHKERR setOperator();
    CHKERR setPC(pc);

  } else {
    MOFEM_LOG("CONTACT", Sev::inform) << "No Schur pc";

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
      CHKERR EssentialPreProcLhs<DisplacementCubitBcData>(
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
  using OpMassStab = B::OpMass<3, SPACE_DIM * SPACE_DIM>;

  auto pip = mField.getInterface<PipelineManager>();
  // Boundary
  auto dm_is = getDMSubData(subDM)->getSmartRowIs();
  auto ao_up = createAOMappingIS(dm_is, PETSC_NULL);

  pip->getOpBoundaryLhsPipeline().push_front(new OpSchurAssembleBegin());
  pip->getOpBoundaryLhsPipeline().push_back(
      new OpMassStab("SIGMA", "SIGMA",
                     [eps_stab](double, double, double) { return eps_stab; }));
  pip->getOpBoundaryLhsPipeline().push_back(new OpSchurAssembleEnd<SCHUR_DGESV>(
      {"SIGMA"}, {nullptr}, {ao_up}, {S}, {false}, false));

  // Domain
  pip->getOpDomainLhsPipeline().push_front(new OpSchurAssembleBegin());
  pip->getOpDomainLhsPipeline().push_back(new OpSchurAssembleEnd<SCHUR_DGESV>(
      {"SIGMA"}, {nullptr}, {ao_up}, {S}, {false}, false));

  auto pre_proc_schur_lhs_ptr = boost::make_shared<FEMethod>();
  auto post_proc_schur_lhs_ptr = boost::make_shared<FEMethod>();

  pre_proc_schur_lhs_ptr->preProcessHook = [this]() {
    MoFEMFunctionBegin;
    CHKERR MatZeroEntries(AT);
    CHKERR MatZeroEntries(P);
    CHKERR MatZeroEntries(S);
    MOFEM_LOG("CONTACT", Sev::verbose) << "Lhs Assemble Begin";
    MoFEMFunctionReturn(0);
  };

  post_proc_schur_lhs_ptr->postProcessHook = [this, ao_up,
                                              post_proc_schur_lhs_ptr]() {
    MoFEMFunctionBegin;
    MOFEM_LOG("CONTACT", Sev::verbose) << "Lhs Assemble End";

    *post_proc_schur_lhs_ptr->matAssembleSwitch = false;

    auto print_mat_norm = [this](auto a, std::string prefix) {
      MoFEMFunctionBegin;
      double nrm;
      CHKERR MatNorm(a, NORM_FROBENIUS, &nrm);
      MOFEM_LOG("CONTACT", Sev::noisy) << prefix << " norm = " << nrm;
      MoFEMFunctionReturn(0);
    };

    CHKERR MatAssemblyBegin(AT, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(AT, MAT_FINAL_ASSEMBLY);
    CHKERR EssentialPreProcLhs<DisplacementCubitBcData>(
        mField, post_proc_schur_lhs_ptr, 1, AT)();

    CHKERR MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
    CHKERR MatAXPY(P, 1, AT, SAME_NONZERO_PATTERN);

    CHKERR MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);

    CHKERR EssentialPreProcLhs<DisplacementCubitBcData>(
        mField, post_proc_schur_lhs_ptr, 1, S, ao_up)();

#ifndef NDEBUG
    CHKERR print_mat_norm(AT, "AT");
    CHKERR print_mat_norm(P, "P");
    CHKERR print_mat_norm(S, "S");
#endif // NDEBUG

    MOFEM_LOG("CONTACT", Sev::verbose) << "Lhs Assemble Finish";
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
      simple->getProblemName(), ROW, "SIGMA", 0, SPACE_DIM, is);
  CHKERR PCFieldSplitSetIS(pc, NULL, is);
  CHKERR PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, S);
  MoFEMFunctionReturn(0);
}

boost::shared_ptr<SetUpSchur>
SetUpSchur::createSetUpSchur(MoFEM::Interface &m_field) {
  return boost::shared_ptr<SetUpSchur>(new SetUpSchurImpl(m_field));
}
