/**
 * \file Thermal_elastic_2D.cpp
 * \example Thermal_elastic_2D.cpp
 *
 * Using PipelineManager interface calculate thermo elastic problem. Example show how
 * to make coupling between elasticity and thermal problem using two fields.
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
static char help[] = "...\n\n";

#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <Thermal_2d.hpp>

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;

constexpr int SPACE_DIM = 2; //< Space dimension of problem, mesh

// Adding all the required forms integrator for thermal problem
using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;
using EdgeElem = EdgeElementForcesAndSourcesCoreBase;
using EdgeEleOp = EdgeElem::UserDataOperator;

using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<1, 1, 2>;
using OpBoundaryMass = FormsIntegrators<EdgeEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpBoundarySource = FormsIntegrators<EdgeEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;
using OpDomainMass = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpHdivU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixDivTimesScalar<2>;

// Adding all the required forms integrator for elastic problem
using OpK = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;
using OpInternalForce =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
        GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;

// These Youngs modulus and poisson ratio should be an input

constexpr double young_modulus = 50e3;
constexpr double poisson_ratio = 0.;
constexpr double coeff_expansion = 1e-5;
constexpr double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
constexpr double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

using namespace Thermal2DOperators;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  // Object to mark boundary entities for the assembling of domain elements
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_1;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_2;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_3;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_4;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_5;

  // Elasticity pointers
  boost::shared_ptr<MatrixDouble> matGradPtr;
  boost::shared_ptr<MatrixDouble> matStrainPtr;
  boost::shared_ptr<MatrixDouble> matStressPtr;
  boost::shared_ptr<MatrixDouble> matDPtr;
  boost::shared_ptr<MatrixDouble> thDPtr;
  boost::shared_ptr<MatrixDouble> bodyForceMatPtr;

  // Heat flux boundary
  Range fluxBoundaryConditions_1;
  Range fluxBoundaryConditions_2;

  // Objects needed for solution updates in Newton's method (dynamics) and for calling also temperature field
  boost::shared_ptr<DataAtGaussPoints> previousUpdate;
  boost::shared_ptr<VectorDouble> fieldValuePtr;
  boost::shared_ptr<MatrixDouble> fieldGradPtr;
  boost::shared_ptr<VectorDouble> fieldDotPtr;

  MatrixDouble invJac;


  // Object needed for postprocessing
  boost::shared_ptr<FaceEle> postProc;

  // Boundary entities marked for fieldsplit (block) solver - optional
  Range boundaryEntitiesForFieldsplit;
};

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  auto set_matrial_stiffens = [&]() {
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

  auto set_thermal_strain = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;

    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    MoFEMFunctionBegin;                    
    auto alpha = coeff_expansion; 
    constexpr double A =
        (SPACE_DIM == 2) ? 2 * shear_modulus_G /
                               (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                         : 1;
    auto t_DT = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*thDPtr);
    t_DT(i, j, k, l) = (2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l)) * alpha;
                          
    MoFEMFunctionReturn(0);
  };

  auto set_body_force = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    MoFEMFunctionBegin;
    auto t_force = getFTensor1FromMat<SPACE_DIM, 0>(*bodyForceMatPtr);
    t_force(i) = 0;
    t_force(1) = -1;
    MoFEMFunctionReturn(0);
  };

  matGradPtr = boost::make_shared<MatrixDouble>();
  matStrainPtr = boost::make_shared<MatrixDouble>();
  matStressPtr = boost::make_shared<MatrixDouble>();
  matDPtr = boost::make_shared<MatrixDouble>();
  thDPtr = boost::make_shared<MatrixDouble>();
  bodyForceMatPtr = boost::make_shared<MatrixDouble>();

  previousUpdate = boost::shared_ptr<DataAtGaussPoints>(new DataAtGaussPoints());
  fieldValuePtr = boost::shared_ptr<VectorDouble>(previousUpdate,&previousUpdate->fieldValue);

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  matDPtr->resize(size_symm * size_symm, 1);
  thDPtr->resize(size_symm * size_symm, 1);
  bodyForceMatPtr->resize(SPACE_DIM, 1);

  CHKERR set_matrial_stiffens();
  CHKERR set_thermal_strain();
  CHKERR set_body_force();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();

  MoFEMFunctionReturn(0);
}

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  // Add field
  CHKERR simpleInterface->addDomainField("U", H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, SPACE_DIM);
  CHKERR simpleInterface->addDomainField("TEMP", H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simpleInterface->addBoundaryField("U", H1,
                                           AINSWORTH_BERNSTEIN_BEZIER_BASE, SPACE_DIM);
  CHKERR simpleInterface->addBoundaryField("TEMP", H1,
                                           AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  
  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder("U", order);
  CHKERR simpleInterface->setFieldOrder("TEMP", order);
  CHKERR simpleInterface->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Applying BCs]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;
  //! [Apply essential BCs for elastic problem]
    // Start Elastic BC
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
        // auto simple = mField.getInterface<Simple>();
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
        CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(verts);
        CHKERR prb_mng->removeDofsOnEntities(simpleInterface->getProblemName(), "U", verts,
                                            lo, hi);
        MoFEMFunctionReturn(0);
    };

    CHKERR remove_ents(fix_disp("FIX_X"), 0, 0);
    CHKERR remove_ents(fix_disp("FIX_Y"), 1, 1);
    CHKERR remove_ents(fix_disp("FIX_Z"), 2, 2);
    CHKERR remove_ents(fix_disp("FIX_ALL"), 0, 3);
    //! [Apply essential BCs for elastic problem]

    //! [Apply BCs for thermal problem]  
auto get_ents_on_mesh_skin_1 = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      if (entity_name.compare(0, 5, "FIX_X") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities, true);
      }
    }
    // Add vertices to boundary entities
    Range boundary_vertices;
    CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                              boundary_vertices, true);
    boundary_entities.merge(boundary_vertices);

    return boundary_entities;
  };

  //   // Remove DOFs as homogeneous boundary condition is used
  // CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
  //     simpleInterface->getProblemName(), domainField, get_ents_on_mesh_skin_1());

    auto get_ents_on_mesh_skin_5 = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      if (entity_name.compare(0, 20, "BOUNDARY_CONDITION_2") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities, true);
      }
    }
    // Add vertices to boundary entities
    Range boundary_vertices;
    CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                              boundary_vertices, true);
    boundary_entities.merge(boundary_vertices);

    return boundary_entities;
  };

  //   // Remove DOFs as homogeneous boundary condition is used
  // CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
  //     simpleInterface->getProblemName(), domainField, get_ents_on_mesh_skin_5());

    // BC NEUMANN 1
    auto get_ents_on_mesh_skin_2 = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      if (entity_name.compare(0, 6, "FLUX_1") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities, true);
      }
    }
    // Add vertices to boundary entities
    Range boundary_vertices;
    CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                              boundary_vertices, true);
    boundary_entities.merge(boundary_vertices);

    return boundary_entities;
  };

    // BC NEUMANN 2
    auto get_ents_on_mesh_skin_3 = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      if (entity_name.compare(0, 6, "FLUX_2") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities, true);
      }
    }
    // Add vertices to boundary entities
    Range boundary_vertices;
    CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                              boundary_vertices, true);
    boundary_entities.merge(boundary_vertices);

    return boundary_entities;
  };


  //BC for DOMAIN
    auto get_ents_on_mesh_skin_4 = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      Range boundary_entities_loop;
      if (entity_name.compare(0, 5, "FIX_X") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities_loop, true);
        boundary_entities.merge(boundary_entities_loop);                                           
      }
      if (entity_name.compare(0, 20, "BOUNDARY_CONDITION_2") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities_loop, true);
        boundary_entities.merge(boundary_entities_loop);                                           
      }      
      if (entity_name.compare(0, 6, "FLUX_1") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities_loop, true);
        boundary_entities.merge(boundary_entities_loop);  
      }
      if (entity_name.compare(0, 6, "FLUX_2") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities_loop, true);
        boundary_entities.merge(boundary_entities_loop);  
      }                
    }
    // Add vertices to boundary entities
    Range boundary_vertices;
    CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                              boundary_vertices, true);
    boundary_entities.merge(boundary_vertices);

    return boundary_entities;
  };

  auto mark_boundary_dofs = [&](Range &&skin_edges) {
    auto problem_manager = mField.getInterface<ProblemsManager>();
    auto marker_ptr = boost::make_shared<std::vector<unsigned char>>();
    problem_manager->markDofs(simpleInterface->getProblemName(), ROW,
                              skin_edges, *marker_ptr);
    return marker_ptr;
  };

  // Get global local vector of marked DOFs. Is global, since is set for all
  // DOFs on processor. Is local since only DOFs on processor are in the
  // vector. To access DOFs use local indices.
  boundaryMarker_1 = mark_boundary_dofs(get_ents_on_mesh_skin_1());
  boundaryMarker_2 = mark_boundary_dofs(get_ents_on_mesh_skin_2());
  boundaryMarker_3 = mark_boundary_dofs(get_ents_on_mesh_skin_3());
  boundaryMarker_4 = mark_boundary_dofs(get_ents_on_mesh_skin_4());
  boundaryMarker_5 = mark_boundary_dofs(get_ents_on_mesh_skin_5());
  fluxBoundaryConditions_1 = get_ents_on_mesh_skin_2();
  fluxBoundaryConditions_2 = get_ents_on_mesh_skin_3();

  EntityHandle meshset_skin1;
  CHKERR mField.get_moab().create_meshset(MESHSET_SET, meshset_skin1);
  CHKERR mField.get_moab().add_entities(meshset_skin1, get_ents_on_mesh_skin_1());
  // CHKERR mField.get_moab().write_mesh("bottom.vtk", &meshset_skin1,1);

    EntityHandle meshset_skin5;
  CHKERR mField.get_moab().create_meshset(MESHSET_SET, meshset_skin5);
  CHKERR mField.get_moab().add_entities(meshset_skin5, get_ents_on_mesh_skin_5());
  // CHKERR mField.get_moab().write_mesh("top.vtk", &meshset_skin5,1);


  MoFEMFunctionReturn(0);
}
//! [Applying BCs]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto &domain_lhs = pipeline_mng->getDomainLhsFE(); 
  auto &domain_rhs = pipeline_mng->getDomainRhsFE(); 
  auto &boundary_lhs = pipeline_mng->getBoundaryRhsFE(); 
  auto &boundary_rhs = pipeline_mng->getBoundaryRhsFE(); 

  double D = 1.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-D", &D, PETSC_NULL);
  auto D_mat = [D](const double, const double, const double) { return D; };
  auto q_unit = [](const double, const double, const double) { return 1; };

  double source = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-source", &source, PETSC_NULL);
  auto sourceTermFunction =[source](const double, const double, const double) { return source; };
  
  double bc_temp1 = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-bc_temp1", &bc_temp1, PETSC_NULL);
  auto bc_1 =[bc_temp1](const double, const double, const double) { return bc_temp1; };

  double bc_temp2 = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-bc_temp2", &bc_temp2, PETSC_NULL);
  auto bc_2 =[bc_temp2](const double, const double, const double) { return bc_temp2; };

  double bc_flux1 = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-bc_flux1", &bc_flux1, PETSC_NULL);
  auto flux_1 =[&](const double, const double, const double) { 
     auto fe_ent = boundary_rhs->getFEEntityHandle();
    if(fluxBoundaryConditions_1.find(fe_ent)!=fluxBoundaryConditions_1.end()) {
      return bc_flux1; 
    } else {
      return 0.;
    }
  };

  double bc_flux2 = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-bc_flux2", &bc_flux2, PETSC_NULL);
  auto flux_2 =[&](const double, const double, const double) { 
    const auto fe_ent = boundary_rhs->getFEEntityHandle();
    if(fluxBoundaryConditions_2.find(fe_ent)!=fluxBoundaryConditions_2.end()) {
      return bc_flux2; 
    } else {
      return 0.;
    }
  };

  auto integration_rule = [](int, int, int p_data) { return 2 * p_data; };

    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetInvJacH1ForFace(invJac));

    // Push operator to get TEMP from Integration Points and pass at pointer 
    pipeline_mng->getOpDomainLhsPipeline().push_back(
    new OpCalculateScalarFieldValues("TEMP", fieldValuePtr)); 

    // Push back the LHS  for conduction
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetBc("TEMP", true, boundaryMarker_4));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainGradGrad("TEMP", "TEMP", D_mat));   
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpUnSetBc("TEMP"));
    
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpSetBc("TEMP", true, boundaryMarker_4));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpDomainSource("TEMP", sourceTermFunction));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpUnSetBc("TEMP"));

    pipeline_mng->getOpDomainLhsPipeline().push_back(new OpK("U", "U", matDPtr));   

    // Start coupling term

    pipeline_mng->getOpDomainLhsPipeline().push_back(new OpKut("U", "TEMP", thDPtr, previousUpdate));

    // end coupling

    // Body force operator
    double set_body_force = 0.;
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-set_body_force", &set_body_force, PETSC_NULL);
    auto body_force =[&](const double, const double, const double) { return set_body_force; };     
    pipeline_mng->getOpDomainRhsPipeline().push_back(new OpBodyForce(
        "U", bodyForceMatPtr, body_force)); 

    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);


//  Operator for coupling LHS boundary term
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpSetBc("TEMP", true, boundaryMarker_4));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryLhs_tm("U", "TEMP", thDPtr, previousUpdate));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(new OpUnSetBc("TEMP"));

    // Start Code for non zero Dirichelet conditions
   // Push operators in boundary pipeline LHS for Dirichelet

   pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpSetBc("TEMP", false, boundaryMarker_1));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryMass("TEMP", "TEMP", q_unit));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(new OpUnSetBc("TEMP"));
  
       // Push operators in boundary pipeline LHS for Dirichelet
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpSetBc("TEMP", false, boundaryMarker_5));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryMass("TEMP", "TEMP", q_unit));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(new OpUnSetBc("TEMP"));
  
   // Push operators in boundary pipeline RHS for Dirichelet
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(
        new OpSetBc("TEMP", false, boundaryMarker_1));
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(
        new OpBoundaryRhs("TEMP", bc_1));
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpUnSetBc("TEMP"));
  
   // Push operators in boundary pipeline RHS for Dirichelet
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(
        new OpSetBc("TEMP", false, boundaryMarker_5));
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(
        new OpBoundaryRhs("TEMP", bc_2));
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpUnSetBc("TEMP")); 
  

    //   // Push operator to get TEMP from Integration Points and pass at pointer 
    // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
    // new OpCalculateScalarFieldValues("TEMP", fieldValuePtr)); 

    // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
    //     new OpBoundaryRhsThermoMech("U", thDPtr, previousUpdate));


// End Code for non zero Dirichelet conditions

    // // Push operators to the Pipeline that is responsible for calculating RHS of
    // // boundary elements
    // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
    //     new OpSetBc("TEMP", false, boundaryMarker_2));
    // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
    //     new OpBoundarySource("TEMP", flux_1));
    // pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpUnSetBc("TEMP"));
  

    // // Push operators to the Pipeline that is responsible for calculating RHS of
    // // boundary elements
    // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
    //     new OpSetBc("TEMP", false, boundaryMarker_3));
    // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
    //     new OpBoundarySource("TEMP", flux_2));
    // pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpUnSetBc("TEMP"));
  
    CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
    CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

  MoFEMFunctionReturn(0);
}

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPSetUp(solver);

  auto dm = simpleInterface->getDM();
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
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  if (SPACE_DIM) {
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    post_proc_fe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
  }
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               matGradPtr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", matGradPtr, matStrainPtr));

      // Push operator to get TEMP from Integration Points and pass at pointer 
    post_proc_fe->getOpPtrVector().push_back(
    new OpCalculateScalarFieldValues("TEMP", fieldValuePtr)); 

  post_proc_fe->getOpPtrVector().push_back(
      new OpTensorTimesSymmetricTensorNew<SPACE_DIM, SPACE_DIM>(
          "U", matStrainPtr, matStressPtr, matDPtr, thDPtr, previousUpdate));
  // post_proc_fe->getOpPtrVector().push_back(
  //   new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
  //       "U", matStrainPtr, matStressPtr, matDPtr));
  post_proc_fe->getOpPtrVector().push_back(new OpPostProcElastic<SPACE_DIM>(
      "U", post_proc_fe->postProcMesh, post_proc_fe->mapGaussPts, matStrainPtr,
      matStressPtr));
  post_proc_fe->addFieldValuesPostProc("U");
  post_proc_fe->addFieldValuesPostProc("TEMP"); 
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_result.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

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

  return 0;
}