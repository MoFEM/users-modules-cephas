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

namespace MDynamicsFunctions {

struct MDynamics {
  int order;
  PetscBool isQuasiStatic;
  MoFEM::Interface &mField;

  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
  boost::ptr_map<std::string, NodalForce> nodal_forces;
  boost::ptr_map<std::string, EdgeForce> edge_forces;

  boost::shared_ptr<DirichletDisplacementBc> dirichletBcPtr; // should be spatial
  // boost::shared_ptr<DirichletSpatialPositionsBc> dirichletBcPtr;

  boost::shared_ptr<NonlinearElasticElement> nonLinearElasticElementPtr;
  boost::shared_ptr<ElasticMaterials> elasticMaterialsPtr;

  MDynamics(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode tsSolve();
  MoFEMErrorCode postProcess();

  MoFEMErrorCode getEntsOnMeshSkin(Range &bc);

  struct LoadPreprocMethods : public FEMethod {
    MoFEM::Interface &mField;
    LoadPreprocMethods(MoFEM::Interface &m_field) : mField(m_field) {}
  };
  
};

MoFEMErrorCode MDynamics::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();

  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);
  CHKERR simple->setFieldOrder("U", order);

  CHKERR simple->addDataField("MESH_NODE_POSITIONS", H1,
                              AINSWORTH_LEGENDRE_BASE, 3);
  CHKERR simple->setFieldOrder("MESH_NODE_POSITIONS", 1);
  auto all_edges_ptr = boost::make_shared<Range>();
  CHKERR mField.get_moab().get_entities_by_type(0, MBEDGE, *all_edges_ptr,
                                                true);
  CHKERR simple->setFieldOrder("MESH_NODE_POSITIONS", 2, all_edges_ptr.get());

  Range skin_edges;
  CHKERR getEntsOnMeshSkin(skin_edges);

  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MDynamics::createCommonData() {
  MoFEMFunctionBegin;
  isQuasiStatic = PETSC_FALSE;
  order = 2;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "-is_quasi_static", &isQuasiStatic,
                             PETSC_NULL);
  CHKERR PetscOptionsGetInt(PETSC_NULL, "-order", &order, PETSC_NULL);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MDynamics::bC() {
  MoFEMFunctionBegin;

  auto bc_mng = mField.getInterface<BcManager>();
  auto simple = mField.getInterface<Simple>();

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_X",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Y",
                                           "U", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Z",
                                           "U", 2, 2);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "REMOVE_ALL", "U", 0, 3);

  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_X", "U",
                                        0, 0);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_Y", "U",
                                        1, 1);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_Z", "U",
                                        2, 2);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_ALL",
                                        "U", 0, 3);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "ROTATE",
                                        "U", 0, 3);

  auto &bc_map = bc_mng->getBcMapByBlockName();
  if (bc_map.size()) {
    boundaryMarker = boost::make_shared<std::vector<char unsigned>>();
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)_FIX_(.*)")) ||
          std::regex_match(b.first, std::regex("(.*)_ROTATE_(.*)"))) {
        boundaryMarker->resize(b.second->bcMarkers.size(), 0);
        for (int i = 0; i != b.second->bcMarkers.size(); ++i) {
          (*boundaryMarker)[i] |= b.second->bcMarkers[i];
        }
      }
    }
  }


  //old boundary conditions implementations
    CHKERR MetaNeumannForces::addNeumannBCElements(m_field, "U");
    CHKERR MetaNodalForces::addElement(m_field, "U");
    CHKERR MetaEdgeForces::addElement(m_field, "U");
    FluidPressure fluid_pressure_fe(m_field);
    fluid_pressure_fe.addNeumannFluidPressureBCElements("U");
    dirichletBcPtr = boost::make_shared<DirichletDisplacementBc>(mField, "U");
    dirichlet_bc_ptr->methodsOp.push_back(new TimeForceScale("-load_history", false));

    MoFEMFunctionReturn(0);
}

MoFEMErrorCode MDynamics::OPs() {
  MoFEMFunctionBegin;
    auto *pipeline_mng = mField.getInterface<PipelineManager>();

    auto integration_rule_vol = [](int, int, int approx_order) {
      return 3 * approx_order;
    };
    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule_vol);
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule_vol);
    auto integration_rule_boundary = [](int, int, int approx_order) {
      return 3 * approx_order;
    };
    CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(
        integration_rule_boundary);
    CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(
        integration_rule_boundary);

    pipeline_mng->getOpDomainLhsPipeline();
    pipeline_mng->getOpDomainRhsPipeline();
    pipeline_mng->getOpBoundaryLhsPipeline();
    pipeline_mng->getOpBoundaryRhsPipeline();

  nonLinearElasticElementPtr = boost::make_shared<NonlinearElasticElement>(mField, 2);
  elasticMaterialsPtr = boost::make_shared<ElasticMaterials>(mField);

  CHKERR elasticMaterialsPtr->setBlocks(
      nonLinearElasticElementPtr->setOfBlocks);

  CHKERR addHOOpsVol("MESH_NODE_POSITIONS",
                     nonLinearElasticElementPtr->getLoopFeRhs(), true, false,
                     false, false);
  CHKERR addHOOpsVol("MESH_NODE_POSITIONS",
                     nonLinearElasticElementPtr->getLoopFeLhs(), true, false,
                     false, false);
  CHKERR addHOOpsVol("MESH_NODE_POSITIONS",
                     nonLinearElasticElementPtr->getLoopFeEnergy(), true, false,
                     false, false);
  CHKERR nonLinearElasticElementPtr->setOperators("U", "MESH_NODE_POSITIONS",
                                                  false, true);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MDynamics::tsSolve() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MDynamics::postProcess() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MDynamics::getEntsOnMeshSkin(Range &boundary_ents) {
  MoFEMFunctionBeginHot;
  
  Range body_ents;
  CHKERR mField.get_moab().get_entities_by_dimension(0, 3, body_ents);
  Skinner skin(&mField.get_moab());
  Range skin_ents;
  CHKERR skin.find_skin(0, body_ents, false, skin_ents);

  // filter not owned entities, those are not on boundary
  // Range boundary_ents;
  ParallelComm *pcomm =
      ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
  if (pcomm == NULL) {
    SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
            "Communicator not created");
  }

  CHKERR pcomm->filter_pstatus(skin_ents, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                               PSTATUS_NOT, -1, &boundary_ents);

  MoFEMFunctionReturnHot(0);
}

} // namespace MDynamicsFunctions