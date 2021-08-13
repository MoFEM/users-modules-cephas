/**
 * \file lesson4_elastic.cpp
 * \example lesson4_elastic.cpp
 *
 * Plane stress elastic problem
 *
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

constexpr int SPACE_DIM = 2; //< Space dimension of problem, mesh

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;
using EdgeEle = EdgeElementForcesAndSourcesCoreBase;
using EdgeEleOp = EdgeEle::UserDataOperator;
using PostProcEle = PostProcFaceOnRefinedMesh;

using OpK = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;

using OpP = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradGrad<1, 1, SPACE_DIM>;
using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;

using OpInternalForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;

using OpDomainGradTimesGravAcceleration = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, 1, SPACE_DIM, 0>;

// boundary opperator aliasing

using OpBoundaryMass = FormsIntegrators<EdgeEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpBoundarySource = FormsIntegrators<EdgeEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

constexpr double young_modulus = 1000.;
constexpr double poisson_ratio = 0.3;
constexpr double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
constexpr double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

#include <OpPostProcElastic.hpp>
using namespace Tutorial;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_2;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();

  MatrixDouble invJac;
  boost::shared_ptr<MatrixDouble> matGradPtr;
  boost::shared_ptr<MatrixDouble> matStrainPtr;
  boost::shared_ptr<MatrixDouble> matStressPtr;
  boost::shared_ptr<MatrixDouble> matDPtr;
  boost::shared_ptr<MatrixDouble> bodyForceMatPtr;
  boost::shared_ptr<MatrixDouble> gravityDirectionMatPtr;

  boost::shared_ptr<VectorDouble> waterPressurePtr;
};

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  //! [Calculate elasticity tensor]
  auto set_material_stiffness = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    MoFEMFunctionBegin;
    constexpr double A =
        (SPACE_DIM == 2) ? 2 * shear_modulus_G /
                               (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                         : 1;
    auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*matDPtr);
    t_D(i, j, k, l) = 2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l);
    MoFEMFunctionReturn(0);
  };
  //! [Calculate elasticity tensor]

  //! [Define gravity vector]
  auto set_body_force = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    MoFEMFunctionBegin;
    auto t_force = getFTensor1FromMat<SPACE_DIM, 0>(*bodyForceMatPtr);
    double unit_weight = 1.;
    CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-unit_weight", &unit_weight,
                               PETSC_NULL);
    t_force(i) = 0;
    if (SPACE_DIM == 2) {
      t_force(1) = -unit_weight;
    } else if (SPACE_DIM == 3) {
      t_force(2) = -unit_weight;
    }


        // t_force(i) = 0;

        MoFEMFunctionReturn(0);
  };
  //! [Define gravity vector]

   auto set_grav_direction = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    MoFEMFunctionBegin;
    auto t_force = getFTensor1FromMat<SPACE_DIM, 0>(*gravityDirectionMatPtr);
  
    t_force(i) = 0;
    if (SPACE_DIM == 2) {
      t_force(1) = -1;
    } else if (SPACE_DIM == 3) {
      t_force(2) = -1;
    }

    // t_force(i) = 0;

    MoFEMFunctionReturn(0);
  };

  //! [Initialise containers for commonData]
  matGradPtr = boost::make_shared<MatrixDouble>();
  matStrainPtr = boost::make_shared<MatrixDouble>();
  matStressPtr = boost::make_shared<MatrixDouble>();
  matDPtr = boost::make_shared<MatrixDouble>();

  bodyForceMatPtr = boost::make_shared<MatrixDouble>();
  gravityDirectionMatPtr = boost::make_shared<MatrixDouble>();
  waterPressurePtr = boost::make_shared<VectorDouble>();

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  matDPtr->resize(size_symm * size_symm, 1);

  bodyForceMatPtr->resize(SPACE_DIM, 1);
  gravityDirectionMatPtr->resize(SPACE_DIM, 1);
  //! [Initialise containers for commonData]

  CHKERR
  set_material_stiffness();
  CHKERR set_body_force();
  CHKERR set_grav_direction();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR createCommonData();
  // CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();
  // CHKERR checkResults();
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
  // CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);
  // CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);
  CHKERR simple->addDomainField("P", H1, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addBoundaryField("P", H1, AINSWORTH_LEGENDRE_BASE, 1);
  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  int order_pressure = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order_pressure", &order_pressure,
                            PETSC_NULL);
  // CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("P", order_pressure);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  //  auto get_ents_on_mesh_skin = [&]() {
  //   Range boundary_entities;
  //   for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
  //     std::string entity_name = it->getName();
  //     if (entity_name.compare(0, 7, "FIX_P_1") == 0) {
  //       CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
  //                                                  boundary_entities, true);
  //     }
  //   }
  //   // Add vertices to boundary entities
  //   Range boundary_vertices;
  //   CHKERR mField.get_moab().get_connectivity(boundary_entities,
  //                                             boundary_vertices, true);
  //   boundary_entities.merge(boundary_vertices);

  //   return boundary_entities;
  // };

  // auto get_ents_on_flux_boundary = [&]() {
  //   Range boundary_entities;
  //   for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
  //     std::string entity_name = it->getName();
  //     if (entity_name.compare(0, 7, "FIX_Q_2") == 0) {
  //       CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
  //                                                  boundary_entities, true);
  //     }
  //   }
  //   // Add vertices to boundary entities
  //   Range boundary_vertices;
  //   CHKERR mField.get_moab().get_connectivity(boundary_entities,
  //                                             boundary_vertices, true);
  //   boundary_entities.merge(boundary_vertices);

  //   return boundary_entities;
  // };
  // auto mark_boundary_dofs = [&](Range &&skin_edges) {
  //   auto problem_manager = mField.getInterface<ProblemsManager>();
  //   auto marker_ptr = boost::make_shared<std::vector<unsigned char>>();
  //   problem_manager->markDofs(simple->getProblemName(), ROW,
  //                             skin_edges, *marker_ptr);
  //   return marker_ptr;
  // };

  // Get global local vector of marked DOFs. Is global, since is set for all
  // DOFs on processor. Is local since only DOFs on processor are in the
  // vector. To access DOFs use local indices.
  cerr << "\n\nCHECK!\n\n";
  // boundaryMarker = mark_boundary_dofs(get_ents_on_flux_boundary());

  //   MoFEMFunctionReturn(0);
  // }

  auto fix_disp = [&](const std::string blockset_name) {
    Range fix_ents;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blockset_name.length(), blockset_name) ==
          0) {
        CHKERR mField.get_moab().get_entities_by_handle(it->meshset, fix_ents,
                                                        true);
      }
    }
    return fix_ents;
  };

  auto remove_ents = [&](const Range &&ents, const int lo, const int hi) {
    auto prb_mng = mField.getInterface<ProblemsManager>();
    auto simple = mField.getInterface<Simple>();
    MoFEMFunctionBegin;
    Range verts;
    CHKERR mField.get_moab().get_connectivity(ents, verts, true);
    verts.merge(ents);
    if (SPACE_DIM == 3) {
      Range adj;
      CHKERR mField.get_moab().get_adjacencies(ents, 1, false, adj,
                                               moab::Interface::UNION);
      verts.merge(adj);
    };

    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "U", verts,
                                         lo, hi);
    MoFEMFunctionReturn(0);
  };

  auto bc_mng = mField.getInterface<BcManager>();
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX_X",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX_Y",
                                           "U", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX_Z",
                                           "U", 2, 2);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX_ALL",
                                           "U", 0, 3);

  auto remove_dofs_from_problem = [&](Range &&skin_edges) {
    MoFEMFunctionBegin;
    auto problem_manager = mField.getInterface<ProblemsManager>();
    CHKERR problem_manager->removeDofsOnEntities(simple->getProblemName(), "P",
                                                 skin_edges, 0, 1);
    MoFEMFunctionReturn(0);
  };
  // CHKERR remove_dofs_from_problem(get_ents_on_mesh_skin());
  MoFEMFunctionReturn(0);
}

//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  if (SPACE_DIM == 2) {
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetInvJacH1ForFace(invJac));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpSetInvJacH1ForFace(invJac));

  }
  double conductivity = 0.05;
  CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-conductivity", &conductivity,
                             PETSC_NULL);
  auto beta_conductivity = [=](const double, const double,
                                           const double) {
    return conductivity;
  };
  // pipeline_mng->getOpDomainLhsPipeline().push_back(new OpK("U", "U", matDPtr));

  auto simple = mField.getInterface<Simple>();
  auto mark_boundary_dofs = [&](Range &&skin_edges) {
    auto problem_manager = mField.getInterface<ProblemsManager>();
    auto marker_ptr = boost::make_shared<std::vector<unsigned char>>();
    problem_manager->markDofs(simple->getProblemName(), ROW, skin_edges,
                              *marker_ptr);
    return marker_ptr;
  };

  // std::string boundary_pressure_1 = "FIX_P_1";
  // std::string boundary_pressure_2 = "FIX_P_2";
  // std::string boundary_flux_1 = "FIX_Q_1";
  // std::string boundary_flux_2 = "FIX_Q_2";

  // if (name_1.compare(name_2)) {
  //   cerr << "This will never print\n";
  // }

  auto get_ents_on_flux_boundary =
      [&](std::string block_name) {
        Range boundary_entities;
         for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(
                                    mField, BLOCKSET, it)) {
          std::string entity_name = it->getName();
          if (entity_name.compare(block_name)==0){
          //  cerr <<entity_name<<endl;
                        // if (entity_name.compare(0, 7, "FIX_Q_2") == 0) {
                        CHKERR it->getMeshsetIdEntitiesByDimension(
                            mField.get_moab(), 1, boundary_entities, true);
          }
        }

  // Add vertices to boundary entities
  Range boundary_vertices;
  CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                            boundary_vertices, true);
  boundary_entities.merge(boundary_vertices);

  return boundary_entities;
  };


  std::string boundary_pressure_1 = "FIX_P_1";
  std::string boundary_pressure_2 = "FIX_P_2";
  std::string boundary_flux_1 = "FIX_Q_1";
  std::string boundary_flux_2 = "FIX_Q_2";

Range boundary_pressure_range_1;
Range boundary_pressure_range_2;
boundary_pressure_range_1 = get_ents_on_flux_boundary(boundary_pressure_1);
boundary_pressure_range_2 = get_ents_on_flux_boundary(boundary_pressure_2);

boundary_pressure_range_1.merge(boundary_pressure_range_2);
auto pass_range = [](Range &passing_range){
  return passing_range;
  };

boundaryMarker = mark_boundary_dofs(pass_range(boundary_pressure_range_1));

pipeline_mng->getOpDomainLhsPipeline().push_back(
    new OpSetBc("P", true, boundaryMarker));
// true means dont fill any values on that
//                                  // boundary as the values are prescribed.
pipeline_mng->getOpDomainLhsPipeline().push_back(
    new OpP("P", "P", beta_conductivity));
pipeline_mng->getOpDomainLhsPipeline().push_back(
    new OpUnSetBc("P")); // releases the true condtion of OpSetBc

// pipeline_mng->getOpDomainRhsPipeline().push_back(new OpBodyForce(
//     "U", bodyForceMatPtr, [](double, double, double) { return 1.; }));

double water_table = -0.;
CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-water_table", &water_table,
                           PETSC_NULL);
double specific_weight_water = 9.81;
CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-specific_weight_water",
                           &specific_weight_water, PETSC_NULL);

// pipeline_mng->getOpDomainRhsPipeline().push_back(
//     new OpDomainRhsHydrostaticStress<SPACE_DIM>("U", specific_weight_water,
//                                                 water_table));

// Boundary opperators

double prescribed_pressure_1 = 1.;
CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-prescribed_pressure_1",
                           &prescribed_pressure_1, PETSC_NULL);
auto beta = [=](const double, const double, const double) {

  return prescribed_pressure_1;
};
double prescribed_pressure_2 = 1.;
CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-prescribed_pressure_2",
                           &prescribed_pressure_2, PETSC_NULL);
auto beta_pressure = [=](const double, const double, const double) {
  return prescribed_pressure_2;
};
double prescribed_flux_1 = 1.;
CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-prescribed_flux_1",
                           &prescribed_pressure_1, PETSC_NULL);
auto beta_flux_1 = [=](const double, const double, const double) {
  return prescribed_pressure_1;
};
double prescribed_flux_2 = 1.;
CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-prescribed_flux_2",
                           &prescribed_pressure_2, PETSC_NULL);
auto beta_flux_2 = [=](const double, const double, const double) {
  return prescribed_flux_2;
};
auto beta_1 = [](const double, const double, const double) { return 1.; };

//  auto set_boundary =
//      [&]() {
//        MoFEMFunctionBegin;
//  pipeline_mng->getOpBoundaryLhsPipeline().push_back(
//      new OpSetBc("P", false, boundaryMarker));
//  pipeline_mng->getOpBoundaryLhsPipeline().push_back(
//      new OpBoundaryMass("P", "P", beta_1));
//  pipeline_mng->getOpBoundaryLhsPipeline().push_back(
//      new OpUnSetBc("P"));

//  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
//      new OpSetBc("P", false, boundaryMarker));
//  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
//      new OpBoundarySource("P", beta));
//  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
//      new OpUnSetBc("P"));

// std::string boundary_flux_1 = "FIX_Q_1";
// std::string boundary_flux_2 = "FIX_Q_2";
pipeline_mng->getOpBoundaryLhsPipeline().push_back(new OpBoundaryMass(
    "P", "P", beta_1,
    boost::make_shared<Range>(get_ents_on_flux_boundary(boundary_pressure_1))));
pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpBoundarySource(
    "P", beta,
    boost::make_shared<Range>(get_ents_on_flux_boundary(boundary_pressure_1))));
pipeline_mng->getOpBoundaryLhsPipeline().push_back(
    new OpBoundaryMass("P", "P", beta_1,
                       boost::make_shared<Range>(
                           get_ents_on_flux_boundary(boundary_pressure_2))));
pipeline_mng->getOpBoundaryRhsPipeline().push_back(
    new OpBoundarySource("P", beta_pressure,
                         boost::make_shared<Range>(get_ents_on_flux_boundary(
                             boundary_pressure_2))));

pipeline_mng->getOpBoundaryRhsPipeline().push_back(
    new OpBoundarySource("P", beta_flux_1,
                         boost::make_shared<Range>(get_ents_on_flux_boundary(
                             boundary_flux_1)))); // for flux 1

pipeline_mng->getOpBoundaryRhsPipeline().push_back(
    new OpBoundarySource("P", beta_flux_2,
                         boost::make_shared<Range>(get_ents_on_flux_boundary(
                             boundary_flux_2)))); // for flux 2


// pipeline_mng->getOpDomainRhsPipeline().push_back(
//     new OpSetBc("P", true, boundaryMarker));
// pipeline_mng->getOpDomainRhsPipeline().push_back(
//     new OpDomainGradTimesGravAcceleration("P", gravityDirectionMatPtr,
//                                           beta_conductivity));
// pipeline_mng->getOpDomainRhsPipeline().push_back(new OpUnSetBc("P"));
// auto remove_dofs_from_problem = [&](Range &&skin_edges) {
//   MoFEMFunctionBegin;
//   auto problem_manager = mField.getInterface<ProblemsManager>();
//   CHKERR problem_manager->removeDofsOnEntities(simple->getProblemName(), "P",
//                                                skin_edges, 0, 1);
//   MoFEMFunctionReturn(0);
// };
// CHKERR remove_dofs_from_problem(
//     get_ents_on_flux_boundary(boundary_pressure_2));

// pipeline_mng->getOpBoundaryRhsPipeline().push_back(
//     new OpSetBc("P", true, boundaryMarker));
// pipeline_mng->getOpBoundaryRhsPipeline().push_back(
//     new OpBoundarySource("P", beta));
// pipeline_mng->getOpBoundaryRhsPipeline().push_back(
//     new OpUnSetBc("P"));
auto integration_rule = [](int, int, int approx_order) {
  return 2 * (approx_order);
};
CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);
//  };
//! [Push operators to pipeline]
MoFEMFunctionReturn(0);
}
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

  double specific_weight_water = 9.81;
  CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-specific_weight_water",
                             &specific_weight_water, PETSC_NULL);
  double water_table = -0.;
  CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-water_table", &water_table,
                             PETSC_NULL);

  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  post_proc_fe->generateReferenceElementMesh();
  if (SPACE_DIM == 2) {
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    post_proc_fe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
  }
  // post_proc_fe->getOpPtrVector().push_back(
  //     new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
  //                                                              matGradPtr));
  // post_proc_fe->getOpPtrVector().push_back(
  //     new OpSymmetrizeTensor<SPACE_DIM>("U", matGradPtr, matStrainPtr));
  // post_proc_fe->getOpPtrVector().push_back(
  //     new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
  //         "U", matStrainPtr, matStressPtr, matDPtr));
  // post_proc_fe->getOpPtrVector().push_back(new OpPostProcElastic<SPACE_DIM>(
  //     "U", post_proc_fe->postProcMesh, post_proc_fe->mapGaussPts, matStrainPtr,
  //     matStressPtr, water_table, specific_weight_water));
  // post_proc_fe->addFieldValuesPostProc("U");
  post_proc_fe->addFieldValuesPostProc("P");
  post_proc_fe->addFieldValuesGradientPostProc("P");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_elastic.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() {
  MOFEM_LOG_CHANNEL("WORLD");
  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  MoFEMFunctionBegin;
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getDomainLhsFE().reset();

  if (SPACE_DIM == 2) {
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpSetInvJacH1ForFace(invJac));
  }
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               matGradPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", matGradPtr, matStrainPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", matStrainPtr, matStressPtr, matDPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpInternalForce("U", matStressPtr));
  (*bodyForceMatPtr) *= -1;
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpBodyForce(
      "U", bodyForceMatPtr, [](double, double, double) { return 1.; }));

  auto integration_rule = [](int, int, int p_data) { return 2 * (p_data - 1); };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);

  auto dm = simple->getDM();
  auto res = smartCreateDMVector(dm);
  pipeline_mng->getDomainRhsFE()->ksp_f = res;

  CHKERR VecZeroEntries(res);
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR VecGhostUpdateBegin(res, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateEnd(res, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecAssemblyBegin(res);
  CHKERR VecAssemblyEnd(res);

  double nrm2;
  CHKERR VecNorm(res, NORM_2, &nrm2);
  MOFEM_LOG_C("WORLD", Sev::verbose, "residual = %3.4e\n", nrm2);
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
