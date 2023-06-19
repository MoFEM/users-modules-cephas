/**
 * \file contact.cpp
 * \CONTACT contact.cpp
 *
 * CONTACT of contact problem
 *
 * @copyright Copyright (c) 2023
 */

/* The above code is a preprocessor directive in C++ that checks if the macro
"EXECUTABLE_DIMENSION" has been defined. If it has not been defined, it replaces
the " */
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

constexpr AssemblyType A = AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType I =
    IntegrationType::GAUSS;                     //< selected integration type

template <int DIM> struct ElementsAndOps {};

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

using AssemblyDomainEleOp = FormsIntegrators<DomainEleOp>::Assembly<A>::OpBase;
using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<A>::OpBase;
constexpr FieldSpace CONTACT_SPACE = ElementsAndOps<SPACE_DIM>::CONTACT_SPACE;

//! [Operators used for contact]
using OpSpringLhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<I>::OpMass<1, SPACE_DIM>;
using OpSpringRhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<I>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Operators used for contact]

//! [Only used for dynamics]
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<I>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Only used for dynamics]

//! [Essential boundary conditions]
using OpEssentialLhs = EssentialBC<BoundaryEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpEssentialLhs<DisplacementCubitBcData, 1, SPACE_DIM>;
using OpEssentialRhs = EssentialBC<BoundaryEleOp>::Assembly<A>::LinearForm<
    GAUSS>::OpEssentialRhs<DisplacementCubitBcData, 1, SPACE_DIM>;
//! [Essential boundary conditions]

namespace ContactOps {

double cn_contact = 0.1;

struct DomainBCs {};
struct BoundaryBCs {};

using DomainRhsBCs = NaturalBC<DomainEleOp>::Assembly<A>::LinearForm<I>;
using OpDomainRhsBCs = DomainRhsBCs::OpFlux<DomainBCs, 1, SPACE_DIM>;
using BoundaryRhsBCs = NaturalBC<BoundaryEleOp>::Assembly<A>::LinearForm<I>;
using OpBoundaryRhsBCs = BoundaryRhsBCs::OpFlux<BoundaryBCs, 1, SPACE_DIM>;
using BoundaryLhsBCs = NaturalBC<BoundaryEleOp>::Assembly<A>::BiLinearForm<I>;
using OpBoundaryLhsBCs = BoundaryLhsBCs::OpFlux<BoundaryBCs, 1, SPACE_DIM>;

}; // namespace ContactOps

constexpr bool is_quasi_static = true;

int order = 2;
int geom_order = 1;
double young_modulus = 100;
double poisson_ratio = 0.25;
double rho = 0.0;
double spring_stiffness = 0.1;
double alpha_dumping = 0;

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
  MoFEMErrorCode postProcess();
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
  CHKERR postProcess();
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
  // CHKERR simple->setFieldOrder("SIGMA", 0);
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
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-alpha_dumping",
                                 &alpha_dumping, PETSC_NULL);

    MOFEM_LOG("CONTACT", Sev::inform) << "Young modulus " << young_modulus;
    MOFEM_LOG("CONTACT", Sev::inform) << "Poisson_ratio " << poisson_ratio;
    MOFEM_LOG("CONTACT", Sev::inform) << "Density " << rho;
    MOFEM_LOG("CONTACT", Sev::inform) << "cn_contact " << cn_contact;
    MOFEM_LOG("CONTACT", Sev::inform)
        << "spring_stiffness " << spring_stiffness;
    MOFEM_LOG("CONTACT", Sev::inform) << "alpha_dumping " << alpha_dumping;

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
    pip.push_back(new OpSetBc("U", true, boundaryMarker));

    CHKERR HenckyOps::opFactoryDomainLhs<DomainEleOp, PETSC, I>(
        mField, pip, "U", "MAT_ELASTIC", Sev::verbose);
    CHKERR opFactoryDomainLhs<DomainEleOp, PETSC, I>(pip, "SIGMA", "U");

    if (!is_quasi_static) {
      auto get_inertia_and_mass_dumping = [this](const double, const double,
                                                 const double) {
        auto *pip_mng = mField.getInterface<PipelineManager>();
        auto &fe_domain_lhs = pip_mng->getDomainLhsFE();
        return rho * fe_domain_lhs->ts_aa + alpha_dumping * fe_domain_lhs->ts_a;
      };
      pip.push_back(new OpMass("U", "U", get_inertia_and_mass_dumping));
    } else if (alpha_dumping > 0) {
      auto get_mass_dumping = [this](const double, const double, const double) {
        auto *pip_mng = mField.getInterface<PipelineManager>();
        auto &fe_domain_lhs = pip_mng->getDomainLhsFE();
        return alpha_dumping * fe_domain_lhs->ts_a;
      };
      pip.push_back(new OpMass("U", "U", get_mass_dumping));
    }


    pip.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;
    pip.push_back(new OpSetBc("U", true, boundaryMarker));

    CHKERR DomainRhsBCs::AddFluxToPipeline<OpDomainRhsBCs>::add(
        pip, mField, "U", {time_scale}, Sev::inform);

    CHKERR HenckyOps::opFactoryDomainRhs<DomainEleOp, PETSC, I>(
        mField, pip, "U", "MAT_ELASTIC", Sev::inform);
    CHKERR ContactOps::opFactoryDomainRhs<DomainEleOp, PETSC, I>(pip, "SIGMA",
                                                                 "U");
    // only in case of dynamics
    if (!is_quasi_static) {
      auto mat_acceleration = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldValuesDotDot<SPACE_DIM>(
          "U", mat_acceleration));
      pip.push_back(new OpInertiaForce(
          "U", mat_acceleration, [](double, double, double) { return rho; }));
    }
    if (alpha_dumping > 0) {
      auto mat_velocity = boost::make_shared<MatrixDouble>();
      pip.push_back(
          new OpCalculateVectorFieldValuesDot<SPACE_DIM>("U", mat_velocity));
      pip.push_back(
          new OpInertiaForce("U", mat_velocity, [](double, double, double) {
            return alpha_dumping;
          }));
    }
    pip.push_back(new OpUnSetBc("U"));
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
    CHKERR EssentialBC<BoundaryEleOp>::Assembly<A>::BiLinearForm<GAUSS>::
        AddEssentialToPipeline<OpEssentialLhs>::add(
            mField, pip, simple->getProblemName(), "U");
    pip.push_back(new OpSetBc("U", true, boundaryMarker));
    CHKERR BoundaryLhsBCs::AddFluxToPipeline<OpBoundaryLhsBCs>::add(
        pip, mField, "U", Sev::inform);

    CHKERR opFactoryBoundaryLhs(pip, "SIGMA", "U");

    if (spring_stiffness > 0)
      pip.push_back(new OpSpringLhs(
          "U", "U",

          [this](double, double, double) { return spring_stiffness; }

          ));

    pip.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    auto u_mat_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_mat_ptr));
    CHKERR EssentialBC<BoundaryEleOp>::Assembly<A>::LinearForm<GAUSS>::
        AddEssentialToPipeline<OpEssentialRhs>::add(
            mField, pip, simple->getProblemName(), "U", u_mat_ptr,
            {boost::make_shared<TimeScale>()}); // note displacements have no
                                                // scaling

    pip.push_back(new OpSetBc("U", true, boundaryMarker));
    CHKERR BoundaryRhsBCs::AddFluxToPipeline<OpBoundaryRhsBCs>::add(
        pip, mField, "U", {time_scale}, Sev::inform);
    CHKERR opFactoryBoundaryRhs(pip, "SIGMA", "U");

    if (spring_stiffness > 0) {
      auto u_disp = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_disp));
      pip.push_back(
          new OpSpringRhs("U", u_disp, [this](double, double, double) {
            return spring_stiffness;
          }));
    }

    pip.push_back(new OpUnSetBc("U"));
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

  auto integration_rule_vol = [](int, int, int approx_order) {
    return 2 * approx_order + geom_order - 1;
  };
  CHKERR pip_mng->setDomainRhsIntegrationRule(integration_rule_vol);
  CHKERR pip_mng->setDomainLhsIntegrationRule(integration_rule_vol);
  auto integration_rule_boundary = [](int, int, int approx_order) {
    return 2 * approx_order + geom_order - 1;
  };
  CHKERR pip_mng->setBoundaryRhsIntegrationRule(integration_rule_boundary);
  CHKERR pip_mng->setBoundaryLhsIntegrationRule(integration_rule_boundary);

  MoFEMFunctionReturn(0);
}
//! [Push operators to pip]

//! [Solve]
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

  auto set_fieldsplit_preconditioner = [&](auto solver) {
    MoFEMFunctionBegin;

    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    PC pc;
    CHKERR KSPGetPC(ksp, &pc);
    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);

    // Setup fieldsplit (block) solver - optional: yes/no
    if (is_pcfs == PETSC_TRUE) {
      auto bc_mng = mField.getInterface<BcManager>();
      auto name_prb = simple->getProblemName();
      auto is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_X", "U", 0, 0);
      is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_Y", "U", 1, 1, is_all_bc);
      is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_Z", "U", 2, 2, is_all_bc);
      is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_ALL", "U", 0, 2, is_all_bc);
      int is_all_bc_size;
      CHKERR ISGetSize(is_all_bc, &is_all_bc_size);
      MOFEM_LOG("CONTACT", Sev::inform)
          << "Field split block size " << is_all_bc_size;
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL,
                               is_all_bc); // boundary block
    }

    MoFEMFunctionReturn(0);
  };

  auto dm = simple->getDM();
  auto D = createDMVector(dm);

  ContactOps::CommonData::createTotalTraction(mField);

  uXScatter = scatter_create(D, 0);
  uYScatter = scatter_create(D, 1);
  if (SPACE_DIM == 3)
    uZScatter = scatter_create(D, 2);

  if (is_quasi_static) {
    auto solver = pip_mng->createTSIM();
    auto D = createDMVector(dm);
    CHKERR set_section_monitor(solver);
    CHKERR set_time_monitor(dm, solver);
    CHKERR TSSetSolution(solver, D);
    CHKERR TSSetFromOptions(solver);
    CHKERR set_fieldsplit_preconditioner(solver);
    CHKERR TSSetUp(solver);
    CHKERR TSSolve(solver, NULL);
  } else {
    auto solver = pip_mng->createTSIM2();
    auto dm = simple->getDM();
    auto D = createDMVector(dm);
    auto DD = vectorDuplicate(D);
    CHKERR set_section_monitor(solver);
    CHKERR set_time_monitor(dm, solver);
    CHKERR TS2SetSolution(solver, D, DD);
    CHKERR TSSetFromOptions(solver);
    CHKERR set_fieldsplit_preconditioner(solver);
    CHKERR TSSetUp(solver);
    CHKERR TSSolve(solver, NULL);
  }

  ContactOps::CommonData::totalTraction.reset();

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Contact::postProcess() { return 0; }
//! [Postprocess results]

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
  MOFEM_LOG_TAG("CONTACT", "indent");

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
