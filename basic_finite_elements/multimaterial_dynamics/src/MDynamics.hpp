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

  boost::shared_ptr<DirichletDisplacementBc>
      dirichletBcPtr; // should be spatial
  // boost::shared_ptr<DirichletSpatialPositionsBc> dirichletBcPtr;

  boost::shared_ptr<NonlinearElasticElement> elasticElementPtr;
  boost::shared_ptr<ElasticMaterials> elasticMaterialsPtr;

  using BcDataTuple = std::tuple<VectorDouble, VectorDouble,
                                 boost::shared_ptr<MethodForForceScaling>>;

  std::map<int, BcDataTuple> bodyForceMap;
  std::map<int, BcDataTuple> dispConstraintMap;

  boost::shared_ptr<PostProcEle> postProc;
  boost::shared_ptr<PostProcSkinEle> postProcSkin;

  MDynamics(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode postProcessSetup();
  MoFEMErrorCode tsSolve();

  MoFEMErrorCode getEntsOnMeshSkin(Range &bc);

  struct LoadPreprocMethods : public FEMethod {
    MoFEM::Interface &mField;
    LoadPreprocMethods(MoFEM::Interface &m_field) : mField(m_field) {}
    MoFEMErrorCode preProcess() { return 0; }
    MoFEMErrorCode postProcess() { return 0; }
    MoFEMErrorCode operator()() { return 0; }
  };
};

struct MDmonitor : public FEMethod {

  MDmonitor(SmartPetscObj<DM> dm) : dM(dm){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  int saveEveryNthStep = 1;

  MoFEMErrorCode postProcess();

public:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProc;
  boost::shared_ptr<PostProcSkinEle> postProcSkin;
};

boost::shared_ptr<MDmonitor> monitorPtr;

MoFEMErrorCode MDynamics::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();

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
    MOFEM_LOG("WORLD", Sev::inform)
        << "Set AINSWORTH_LEGENDRE_BASE for displacements";
    break;
  case DEMKOWICZ:
    base = DEMKOWICZ_JACOBI_BASE;
    MOFEM_LOG("WORLD", Sev::inform)
        << "Set DEMKOWICZ_JACOBI_BASE for displacements";
    break;
  default:
    base = LASTBASE;
    break;
  }

  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
  CHKERR simple->setFieldOrder("U", order);

  CHKERR simple->addDataField("MESH_NODE_POSITIONS", H1, base, 3);
  CHKERR simple->setFieldOrder("MESH_NODE_POSITIONS", 2);

  Range skin_edges;
  CHKERR getEntsOnMeshSkin(skin_edges);


    // old boundary conditions implementations
  CHKERR MetaNeumannForces::addNeumannBCElements(mField, "U");
  CHKERR MetaNodalForces::addElement(mField, "U");
  CHKERR MetaEdgeForces::addElement(mField, "U");

  simple->getOtherFiniteElements().push_back("FORCE_FE");
  simple->getOtherFiniteElements().push_back("PRESSURE_FE");

  // CHKERR simple->setUp();

  CHKERR simple->defineFiniteElements();
  // if (!simple->skeletonFields.empty())
  //   CHKERR simple->setSkeletonAdjacency();
  CHKERR simple->defineProblem(PETSC_TRUE);
  CHKERR simple->buildFields();
  CHKERR simple->buildFiniteElements();
  CHKERR simple->buildProblem();

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

  auto dm = mField.getInterface<Simple>()->getDM();
  auto bc_mng = mField.getInterface<BcManager>();
  auto simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

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
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "ROTATE", "U",
                                        0, 3);

  boundaryMarker =
      bc_mng->getMergedBlocksMarker(vector<string>{"FIX_", "ROTATE"});

  auto get_id_history_param = [&](string base_name, int id) {
    char load_hist_file[255] = "hist.in";
    PetscBool ctg_flag = PETSC_FALSE;
    string param_name_with_id = "-" + base_name + "_" + to_string(id);
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL,
                                 param_name_with_id.c_str(), load_hist_file,
                                 255, &ctg_flag);
    if (ctg_flag)
      return param_name_with_id;
    return string("");
  };

  auto get_adj_ents = [&](const Range &ents) {
    Range verts;
    CHKERR mField.get_moab().get_connectivity(ents, verts, true);
    for (size_t d = 1; d < 3; ++d)
      CHKERR mField.get_moab().get_adjacencies(ents, d, false, verts,
                                               moab::Interface::UNION);
    verts.merge(ents);
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(verts);
    return verts;
  };

  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
    const std::string block_name = "BODY_FORCE";
    if (it->getName().compare(0, block_name.size(), block_name) == 0) {
      std::vector<double> attr;
      CHKERR it->getAttributes(attr);
      if (attr.size() > 3) {

        const int id = it->getMeshsetId();

        Range bc_ents;
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 3,
                                                   bc_ents, true);
        auto bc_ents_ptr = boost::make_shared<Range>(get_adj_ents(bc_ents));

        VectorDouble accel({attr[0], attr[1], attr[2]});
        VectorDouble density({attr[3], 0., 0.});
        // if accelerogram is provided then change the acceleration, otherwise use whatever is in that block TODO:
        boost::shared_ptr<MethodForForceScaling> method_for_scaling;
        auto param_name_for_scaling = get_id_history_param("accelerogram", id);

        if (!param_name_for_scaling.empty())
          method_for_scaling = boost::shared_ptr<MethodForForceScaling>(
              new TimeAccelerogram(param_name_for_scaling));

        bodyForceMap[id] = std::make_tuple(accel, density, method_for_scaling);
        auto &acc = std::get<0>(bodyForceMap.at(id));
        auto &rho = std::get<1>(bodyForceMap.at(id));
        auto &method = std::get<2>(bodyForceMap.at(id));

        auto get_scale_body = [&](double, double, double) {
          auto *pipeline_mng = mField.getInterface<PipelineManager>();
          FTensor::Index<'i', 3> i;

          auto fe_domain_rhs = pipeline_mng->getDomainRhsFE();
          if (method)
            CHKERR MethodForForceScaling::applyScale(fe_domain_rhs.get(),
                                                     method, acc);

          FTensor::Tensor1<double, 3> t_source(acc(0), acc(1), acc(2));
          t_source(i) *= rho(0);
          return t_source;
        };

        auto get_rho = [&](double, double, double) {
          auto *pipeline_mng = mField.getInterface<PipelineManager>();
          auto &fe_domain_lhs = pipeline_mng->getDomainLhsFE();
          return rho(0) * fe_domain_lhs->ts_aa;
        };

        pipeline_mng->getOpDomainRhsPipeline().push_back(
            new OpBodyForce("U", get_scale_body, bc_ents_ptr));

        if (!isQuasiStatic) {
          pipeline_mng->getOpDomainLhsPipeline().push_back(
              new OpMass("U", "U", get_rho, bc_ents_ptr));
          auto mat_acceleration = boost::make_shared<MatrixDouble>();
          pipeline_mng->getOpDomainRhsPipeline().push_back(
              new OpCalculateVectorFieldValuesDotDot<3>("U", mat_acceleration));
          pipeline_mng->getOpDomainRhsPipeline().push_back(new OpInertiaForce(
              "U", mat_acceleration,
              [&](double, double, double) { return rho(0); }, bc_ents_ptr));
        }
      } else {
        SETERRQ1(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
                 "There should be (3 acceleration + 1 density) attributes in "
                 "BODY_FORCE blockset, but is %d",
                 attr.size());
      }
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MDynamics::OPs() {
  MoFEMFunctionBegin;
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto dm = mField.getInterface<Simple>()->getDM();

  auto integration_rule_vol = [](int, int, int approx_order) {
    return 3 * approx_order;
  };
  auto integration_rule_boundary = [](int, int, int approx_order) {
    return 3 * approx_order;
  };

  auto push_methods = [&](auto method, string element_name, bool is_lhs,
                          bool is_rhs) {
    MoFEMFunctionBeginHot;
    boost::shared_ptr<FEMethod> null;
    if (isQuasiStatic) {
      if (is_lhs)
        CHKERR DMMoFEMTSSetIJacobian(dm, element_name.c_str(), method, method,
                                     method);
      if (is_rhs)
        CHKERR DMMoFEMTSSetIFunction(dm, element_name.c_str(), method, method,
                                     method);
    } else {
      if (is_lhs)
        CHKERR DMMoFEMTSSetI2Jacobian(dm, element_name.c_str(), method, method,
                                      method);
      if (is_rhs)
        CHKERR DMMoFEMTSSetI2Function(dm, element_name.c_str(), method, method,
                                      method);
    }
    MoFEMFunctionReturnHot(0);
  };

  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule_vol);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule_vol);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule_boundary);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule_boundary);


  // neumann bcs declaration here
  // CHKERR simple->reSetUp();

  auto get_history_param = [&](string prefix) {
    char load_hist_file[255] = "hist.in";
    PetscBool ctg_flag = PETSC_FALSE;
    string new_param_file = string("-") + prefix + string("_history");
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, new_param_file.c_str(),
                                 load_hist_file, 255, &ctg_flag);
    if (ctg_flag)
      return new_param_file;
    return string("-load_history");
  };

  CHKERR MetaNeumannForces::setMomentumFluxOperators(mField, neumann_forces,
                                                     PETSC_NULL, "U");
  CHKERR MetaNodalForces::setOperators(mField, nodal_forces, PETSC_NULL, "U");
  CHKERR MetaEdgeForces::setOperators(mField, edge_forces, PETSC_NULL, "U");

  auto set_neumann_methods = [&](auto &neumann_el, string hist_name) {
    MoFEMFunctionBeginHot;
    for (auto &&mit : neumann_el) {
      mit->second->methodsOp.push_back(
          new TimeForceScale(get_history_param(hist_name), false));
      CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS", mit->second->getLoopFe(),
                            false, false);
      CHKERR push_methods(&mit->second->getLoopFe(), mit->first.c_str(), false,
                          true);
      // CHKERR DMMoFEMTSSetIFunction(dm, mit->first.c_str(),
      //                              &mit->second->getLoopFe(), NULL, NULL);
    }
    MoFEMFunctionReturnHot(0);
  };

  CHKERR set_neumann_methods(neumann_forces, "force");
  CHKERR set_neumann_methods(nodal_forces, "force");
  CHKERR set_neumann_methods(edge_forces, "force");

  dirichletBcPtr = boost::make_shared<DirichletDisplacementBc>(mField, "U");
  dirichletBcPtr->methodsOp.push_back(
      new TimeForceScale(get_history_param("dirichlet"), false));
  CHKERR push_methods(dirichletBcPtr.get(), simple->getDomainFEName().c_str(),
                      true, true);

  auto load_pre_proc_methods = boost::make_shared<LoadPreprocMethods>(mField);

  Projection10NodeCoordsOnField ent_method_material(mField,
                                                    "MESH_NODE_POSITIONS");
  CHKERR mField.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);

  elasticElementPtr = boost::make_shared<NonlinearElasticElement>(mField, 2);
  elasticMaterialsPtr = boost::make_shared<ElasticMaterials>(mField);

  CHKERR elasticMaterialsPtr->setBlocks(elasticElementPtr->setOfBlocks);

  CHKERR addHOOpsVol("MESH_NODE_POSITIONS", elasticElementPtr->getLoopFeRhs(),
                     true, false, false, false);
  CHKERR addHOOpsVol("MESH_NODE_POSITIONS", elasticElementPtr->getLoopFeLhs(),
                     true, false, false, false);
  CHKERR addHOOpsVol("MESH_NODE_POSITIONS",
                     elasticElementPtr->getLoopFeEnergy(), true, false, false,
                     false);
  CHKERR elasticElementPtr->setOperators("U", "MESH_NODE_POSITIONS", false,
                                         true);

  CHKERR push_methods(&elasticElementPtr->getLoopFeRhs(),
                      simple->getDomainFEName().c_str(), false, true);
  CHKERR push_methods(&elasticElementPtr->getLoopFeLhs(),
                      simple->getDomainFEName().c_str(), true, false);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MDynamics::tsSolve() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  ISManager *is_manager = mField.getInterface<ISManager>();
  auto dm = simple->getDM();

  auto set_time_monitor = [&](auto dm, auto solver) {
    MoFEMFunctionBegin;
    monitorPtr = boost::make_shared<MDmonitor>(dm);
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitorPtr, null, null);
    MoFEMFunctionReturn(0);
  };

  auto D = smartCreateDMVector(dm);
  SmartPetscObj<TS> solver;

  if (isQuasiStatic) {
    solver = pipeline_mng->createTS();
    CHKERR TSSetFromOptions(solver);
    CHKERR TSSetSolution(solver, D);
  } else {
    solver = pipeline_mng->createTS2();
    CHKERR TSSetFromOptions(solver);
    CHKERR TSSetType(solver, TSALPHA2);
    auto DD = smartVectorDuplicate(D);
    CHKERR TS2SetSolution(solver, D, DD);
  }

  CHKERR set_time_monitor(dm, solver);
  CHKERR OPs();
  CHKERR postProcessSetup();
  
  CHKERR TSSetExactFinalTime(solver, TS_EXACTFINALTIME_MATCHSTEP);
  CHKERR TSSetUp(solver);
  CHKERR TSSolve(solver, NULL);

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MDynamics::postProcessSetup() {
  MoFEMFunctionBegin;
  postProc = boost::make_shared<PostProcEle>(mField);
  monitorPtr->postProc = postProc;

  CHKERR postProc->generateReferenceElementMesh();
  if (mField.check_field("MESH_NODE_POSITIONS"))
    CHKERR addHOOpsVol("MESH_NODE_POSITIONS", *postProc, true, false, false,
                       false);
  CHKERR postProc->addFieldValuesPostProc("U");
  CHKERR postProc->addFieldValuesGradientPostProc("U");

  auto sit = elasticElementPtr->setOfBlocks.begin();
  for (; sit != elasticElementPtr->setOfBlocks.end(); sit++) {
    postProc->getOpPtrVector().push_back(
        new PostProcStress(postProc->postProcMesh, postProc->mapGaussPts, "U",
                           sit->second, postProc->commonData, true));
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MDmonitor::postProcess() {
  MoFEMFunctionBegin;
  if (ts_step % saveEveryNthStep == 0) {
    CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
    // CHKERR DMoFEMLoopFiniteElements(dM, "ELASTIC", postProc);
    CHKERR postProc->writeFile(
        "out_vol_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
  }
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