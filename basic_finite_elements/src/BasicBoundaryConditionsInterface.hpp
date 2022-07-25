/** \file BasicBoundaryConditionsInterface.hpp
  \brief Header file for BasicBoundaryConditionsInterface element implementation
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

// TODO: This is still work in progress !!!

// FIX and ROTATE Boundary conditions
// include SurfacePressureComplexForLazy for following load


#ifndef __BASICBOUNDARYCONDIONSINTERFACE_HPP__
#define __BASICBOUNDARYCONDIONSINTERFACE_HPP__


/** \brief Set of functions declaring elements and setting operators
 * for basic boundary conditions interface
 */
struct BasicBoundaryConditionsInterface : public GenericElementInterface {

  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = FaceElementForcesAndSourcesCore;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;

  using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<
      PETSC>::LinearForm<GAUSS>::OpSource<1, 3>;
  using OpMass = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
      GAUSS>::OpMass<1, 3>;
  using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
      PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, 3, 1>;

  using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
      PETSC>::BiLinearForm<GAUSS>::OpMass<1, 3>;
  using OpBoundaryVec = FormsIntegrators<BoundaryEleOp>::Assembly<
      PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, 3, 0>;
  using OpBoundaryInternal = FormsIntegrators<BoundaryEleOp>::Assembly<
      PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, 3, 1>;

  MoFEM::Interface &mField;
  SmartPetscObj<DM> dM;

  double *snesLambdaLoadFactorPtr;

  struct LoadScale : public MethodForForceScaling {
    double *lAmbda;
    LoadScale(double *my_lambda) : lAmbda(my_lambda){};
    MoFEMErrorCode scaleNf(const FEMethod *fe, VectorDouble &nf) {
      MoFEMFunctionBegin;
      nf *= *lAmbda;
      MoFEMFunctionReturn(0);
    }
  };
  
  bool isDisplacementField;
  bool isQuasiStatic;
  bool isPartitioned;
  bool isLinear;

  BitRefLevel bIt;

  string positionField;
  string meshNodeField;

  boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
  boost::ptr_map<std::string, NodalForce> nodal_forces;
  boost::ptr_map<std::string, EdgeForce> edge_forces;

  boost::shared_ptr<FluidPressure> fluidPressureElementPtr;

  boost::shared_ptr<FaceElementForcesAndSourcesCore> springRhsPtr;
  boost::shared_ptr<FaceElementForcesAndSourcesCore> springLhsPtr;

  boost::shared_ptr<VolumeElementForcesAndSourcesCore> bodyForceRhsPtr;
  boost::shared_ptr<VolumeElementForcesAndSourcesCore> bodyForceLhsPtr;
  // boost::shared_ptr<FEMethod> dirichletBcPtr;

  boost::shared_ptr<DirichletDisplacementBc> dirichletBcPtr;
  boost::shared_ptr<KelvinVoigtDamper> damperElementPtr;

  const string domainProblemName;
  const string domainElementName;

  using BcDataTuple = std::tuple<VectorDouble, double,
                                 boost::shared_ptr<MethodForForceScaling>>;

  std::map<int, BcDataTuple> bodyForceMap;
  std::map<int, BcDataTuple> dispConstraintMap;

  // boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProc;
  // boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcSkin;

  BasicBoundaryConditionsInterface(
      MoFEM::Interface &m_field, string postion_field,
      string mesh_pos_field_name = "MESH_NODE_POSITIONS",
      string problem_name = "ELASTIC",
      string domain_element_name = "ELASTIC_FE",
      bool is_displacement_field = true, bool is_quasi_static = true,
      double *snes_load_factor = nullptr, bool is_partitioned = true)
      : mField(m_field), positionField(postion_field),
        meshNodeField(mesh_pos_field_name), domainProblemName(problem_name),
        domainElementName(domain_element_name),
        isDisplacementField(is_displacement_field),
        isQuasiStatic(is_quasi_static),
        snesLambdaLoadFactorPtr(snes_load_factor),
        isPartitioned(is_partitioned), isLinear(PETSC_FALSE) {}

  ~BasicBoundaryConditionsInterface() {}

  MoFEMErrorCode getCommandLineParameters() override {
    MoFEMFunctionBegin;

    PetscBool quasi_static = PETSC_FALSE;
    PetscBool is_linear = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "-is_quasi_static", &quasi_static,
                               PETSC_NULL);
    CHKERR PetscOptionsGetBool(PETSC_NULL, "-is_linear", &is_linear,
                               PETSC_NULL);

    isQuasiStatic = quasi_static;
    isLinear = is_linear;

    MoFEMFunctionReturn(0);
  };

  MoFEMErrorCode addElementFields() override { return 0;};

  MoFEMErrorCode createElements() override {
    MoFEMFunctionBeginHot;

    CHKERR MetaSpringBC::addSpringElements(mField, positionField,
                                           meshNodeField);
    CHKERR MetaNeumannForces::addNeumannBCElements(mField, positionField);
    CHKERR MetaNodalForces::addElement(mField, positionField);
    CHKERR MetaEdgeForces::addElement(mField, positionField);

    if (!isDisplacementField)
      dirichletBcPtr = boost::make_shared<DirichletSpatialRemoveDofsBc>(
          mField, positionField, domainProblemName, meshNodeField,
          "DISPLACEMENT", isPartitioned);
    else
      dirichletBcPtr = boost::make_shared<DirichletDisplacementRemoveDofsBc>(
          mField, positionField, domainProblemName, "DISPLACEMENT",
          isPartitioned);
    // CHKERR dynamic_cast<DirichletDisplacementRemoveDofsBc &>(
    //     *dirichletBcPtr).iNitialize();

    CHKERR mField.add_finite_element("DAMPER_FE", MF_ZERO);
    CHKERR mField.modify_finite_element_add_field_row("DAMPER_FE",
                                                      positionField);
    CHKERR mField.modify_finite_element_add_field_col("DAMPER_FE",
                                                      positionField);
    CHKERR mField.modify_finite_element_add_field_data("DAMPER_FE",
                                                       positionField);
    CHKERR mField.modify_finite_element_add_field_data("DAMPER_FE",
                                                       meshNodeField);

    // CHKERR mField.add_finite_element(domainElementName, "FLUID_PRESSURE_FE");

    fluidPressureElementPtr = boost::make_shared<FluidPressure>(mField);
    fluidPressureElementPtr->addNeumannFluidPressureBCElements(positionField);
    CHKERR addHOOpsFace3D(meshNodeField, fluidPressureElementPtr->getLoopFe(),
                          false, false);

    damperElementPtr = boost::make_shared<KelvinVoigtDamper>(mField);
    damperElementPtr->commonData.meshNodePositionName = meshNodeField;

    auto &common_data = damperElementPtr->commonData;

    common_data.spatialPositionName = positionField;
    common_data.spatialPositionNameDot = "DOT_" + positionField;
    damperElementPtr->setBlockDataMap(); FIXME:

    for (auto &[id, data] : damperElementPtr->blockMaterialDataMap) {
      data.lInear = isLinear;
      int cid = id;
      damperElementPtr->constitutiveEquationMap.insert(
          cid, new KelvinVoigtDamper::ConstitutiveEquation<adouble>(data, isDisplacementField));
      CHKERR mField.add_ents_to_finite_element_by_type(data.tEts, MBTET,
                                                       "DAMPER_FE");
    }

    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode setOperators() override {
    MoFEMFunctionBeginHot;
    CHKERR MetaNodalForces::setOperators(mField, nodal_forces, PETSC_NULL,
                                         positionField);
    CHKERR MetaEdgeForces::setOperators(mField, edge_forces, PETSC_NULL,
                                        positionField);

    CHKERR MetaNeumannForces::setMomentumFluxOperators(
        mField, neumann_forces, PETSC_NULL, positionField);
    springLhsPtr = boost::make_shared<FaceElementForcesAndSourcesCore>(mField);
    springRhsPtr = boost::make_shared<FaceElementForcesAndSourcesCore>(mField);
    bodyForceRhsPtr =
        boost::make_shared<VolumeElementForcesAndSourcesCore>(mField);
    bodyForceLhsPtr =
        boost::make_shared<VolumeElementForcesAndSourcesCore>(mField);

    CHKERR MetaSpringBC::setSpringOperators(mField, springLhsPtr, springRhsPtr,
                                            positionField,
                                            meshNodeField);

    
    fluidPressureElementPtr->setNeumannFluidPressureFiniteElementOperators(
        positionField, PETSC_NULL, true, true);

    // KelvinVoigtDamper::CommonData &common_data =
    // damperElementPtr->commonData;
    CHKERR damperElementPtr->setOperators(3);

    // auto dm = mField.getInterface<Simple>()->getDM();
    auto bc_mng = mField.getInterface<BcManager>();
    auto *pipeline_mng = mField.getInterface<PipelineManager>();

    // FIXME: this has to also work with removeDofsOnEntitiesNotDistributed !!! 
    // CHKERR bc_mng->removeBlockDOFsOnEntities(domainProblemName, "REMOVE_X",
    //                                          positionField, 0, 0);
    // CHKERR bc_mng->removeBlockDOFsOnEntities(domainProblemName, "REMOVE_Y",
    //                                          positionField, 1, 1);
    // CHKERR bc_mng->removeBlockDOFsOnEntities(domainProblemName, "REMOVE_Z",
    //                                          positionField, 2, 2);
    // CHKERR bc_mng->removeBlockDOFsOnEntities(domainProblemName, "REMOVE_ALL",
    //                                          positionField, 0, 3);

    CHKERR bc_mng->pushMarkDOFsOnEntities(domainProblemName, "FIX_X",
                                          positionField, 0, 0);
    CHKERR bc_mng->pushMarkDOFsOnEntities(domainProblemName, "FIX_Y",
                                          positionField, 1, 1);
    CHKERR bc_mng->pushMarkDOFsOnEntities(domainProblemName, "FIX_Z",
                                          positionField, 2, 2);
    CHKERR bc_mng->pushMarkDOFsOnEntities(domainProblemName, "FIX_ALL",
                                          positionField, 0, 3);
    CHKERR bc_mng->pushMarkDOFsOnEntities(domainProblemName, "ROTATE",
                                          positionField, 0, 3);

    mBoundaryMarker =
        bc_mng->getMergedBlocksMarker(vector<string>{"FIX_", "ROTATE"});

    auto get_id_block_param = [&](string base_name, int id) {
      char load_hist_file[255] = "hist.in";
      PetscBool ctg_flag = PETSC_FALSE;
      string param_name_with_id = "-" + base_name + "_" + to_string(id);
      CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL,
                                   param_name_with_id.c_str(), load_hist_file,
                                   255, &ctg_flag);
      if (ctg_flag)
        return param_name_with_id;
      
      param_name_with_id = "-" + base_name;
      CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL,
                                   param_name_with_id.c_str(), load_hist_file,
                                   255, &ctg_flag);
      if (ctg_flag) {
        MOFEM_LOG("WORLD", Sev::verbose)
            << "Setting one accelerogram for all blocks!";
        return param_name_with_id;
      }
      
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

          VectorDouble accel({attr[1], attr[2], attr[3]});
          double density = attr[0];
          bool inertia_flag =
              attr.size() > 4 ? bool(std::floor(attr[4])) : true;
          // if accelerogram is provided then change the acceleration,
          // otherwise use whatever is in that block TODO:
          boost::shared_ptr<MethodForForceScaling> method_for_scaling;
          auto param_name_for_scaling =
              get_id_block_param("accelerogram", id);

          if (!param_name_for_scaling.empty())
            method_for_scaling = boost::shared_ptr<MethodForForceScaling>(
                new TimeAccelerogram(param_name_for_scaling));

          bodyForceMap[id] =
              std::make_tuple(accel, density, method_for_scaling);
          auto &acc = std::get<0>(bodyForceMap.at(id));
          auto &rho0 = std::get<1>(bodyForceMap.at(id));
          auto &method = std::get<2>(bodyForceMap.at(id));

          auto get_scale_body = [&](double, double, double) {
            auto *pipeline_mng = mField.getInterface<PipelineManager>();
            FTensor::Index<'i', 3> i;
            auto acc_c = acc;
            auto fe_domain_rhs = bodyForceRhsPtr;
            if (method)
              CHKERR MethodForForceScaling::applyScale(fe_domain_rhs.get(),
                                                       method, acc_c);

            FTensor::Tensor1<double, 3> t_source(acc_c(0), acc_c(1), acc_c(2));
            t_source(i) *= rho0;
            return t_source;
          };

          // FIXME: this require correction for large strains, multiply by det_F
          auto get_rho = [&](double, double, double) {
            auto *pipeline_mng = mField.getInterface<PipelineManager>();
            auto &fe_domain_lhs = bodyForceLhsPtr;
            return rho0 * fe_domain_lhs->ts_aa;
          };

          auto &pipeline_rhs = bodyForceRhsPtr->getOpPtrVector();
          auto &pipeline_lhs = bodyForceLhsPtr->getOpPtrVector();

          CHKERR addHOOpsVol("MESH_NODE_POSITIONS", *bodyForceRhsPtr, true,
                             false, false, false);
          CHKERR addHOOpsVol("MESH_NODE_POSITIONS", *bodyForceLhsPtr, true,
                             false, false, false);

          // boundaryMarker.reset();
          //FIXME: fix for large strains
          pipeline_rhs.push_back(
              new OpSetBc(positionField, true, mBoundaryMarker));
          pipeline_rhs.push_back(
              new OpBodyForce(positionField, get_scale_body, bc_ents_ptr));
          if (!isQuasiStatic && inertia_flag) {
            pipeline_lhs.push_back(
                new OpSetBc(positionField, true, mBoundaryMarker));
            pipeline_lhs.push_back(
                new OpMass(positionField, positionField, get_rho, bc_ents_ptr));
            auto mat_acceleration = boost::make_shared<MatrixDouble>();
            pipeline_rhs.push_back(new OpCalculateVectorFieldValuesDotDot<3>(
                positionField, mat_acceleration));
            pipeline_rhs.push_back(new OpInertiaForce(
                positionField, mat_acceleration,
                [&](double, double, double) { return rho0; }, bc_ents_ptr));
            pipeline_lhs.push_back(new OpUnSetBc(positionField));
          }
          pipeline_rhs.push_back(new OpUnSetBc(positionField));

        } else {
          SETERRQ1(
              PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "There should be (1 density + 3 accelerations ) attributes in "
              "BODY_FORCE blockset, but is %d. Optionally, you can set 5th "
              "parameter to inertia flag.",
              attr.size());
        }
      }
    }

    auto integration_rule_vol = [](int, int, int approx_order) {
      return 2 * approx_order + 1;
    };
    auto integration_rule_boundary = [](int, int, int approx_order) {
      return 2 * approx_order + 1;
    };

    // CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule_vol);
    // CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule_vol);
    // CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(
    //     integration_rule_boundary);
    // CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(
    //     integration_rule_boundary);

    springLhsPtr->getRuleHook = integration_rule_boundary;
    springRhsPtr->getRuleHook = integration_rule_boundary;
    bodyForceLhsPtr->getRuleHook = integration_rule_vol;
    bodyForceRhsPtr->getRuleHook = integration_rule_vol;

    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) override {
    MoFEMFunctionBeginHot;
    this->dM = dm;
    auto simple = mField.getInterface<Simple>();
    vector<const char *> element_list{"FORCE_FE", "PRESSURE_FE",
                                      "FLUID_PRESSURE_FE",
                                      "SPRING",
                                      "DAMPER_FE"};
    for (auto &el : element_list) {
      CHKERR DMMoFEMAddElement(dM, el);
      simple->getOtherFiniteElements().push_back(el);
    }
    // if (!fluidPressureElementPtr->setOfFluids.empty())
    // FIXME:
    // CHKERR mField.modify_problem_add_finite_element(domainProblemName,
    //                                                 "FLUID_PRESSURE_FE");
    // CHKERR dynamic_cast<DirichletDisplacementRemoveDofsBc &>(
    //     *dirichletBcPtr).iNitialize();
    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode updateElementVariables() override { return 0; };
  MoFEMErrorCode postProcessElement(int step) override { return 0; };

  string getHistoryParam(string prefix) {
    char load_hist_file[255] = "hist.in";
    PetscBool ctg_flag = PETSC_FALSE;
    string new_param_file = string("-") + prefix + string("_history");
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, new_param_file.c_str(),
                                 load_hist_file, 255, &ctg_flag);
    if (ctg_flag)
      return new_param_file;
    return string("-load_history");
  };

  template <typename T>
  MoFEMErrorCode setupSolverFunction(const TSType type = IM) {
    CHKERR setupSolverImpl<T, true>(type);
    MoFEMFunctionReturnHot(0);
  }

  template <typename T>
  MoFEMErrorCode setupSolverJacobian(const TSType type = IM) {
    CHKERR setupSolverImpl<T, false>(type);
    MoFEMFunctionReturnHot(0);
  }

  template <typename T, bool RHS>
  MoFEMErrorCode setupSolverImpl(const TSType type = IM) {
    MoFEMFunctionBeginHot;
    // auto dm = dM;
    // boost::shared_ptr<FEMethod> null;
    
    auto set_solver_pipelines =
        [&](PetscErrorCode (*function)(DM, const char fe_name[], MoFEM::FEMethod *,
                                       MoFEM::BasicMethod *,
                                       MoFEM::BasicMethod *),
            PetscErrorCode (*jacobian)(DM, const char fe_name[], MoFEM::FEMethod *,
                                       MoFEM::BasicMethod *,
                                       MoFEM::BasicMethod *)) {

          MoFEMFunctionBeginHot;
          if (RHS) {
            CHKERR DMoFEMPreProcessFiniteElements(dM, dirichletBcPtr.get());

            // auto push_fmethods = [&](auto method, string element_name) {
            //   CHKERR function(dm, element_name.c_str(), method, method, method);
            // };

            auto set_neumann_methods = [&](auto &neumann_el, string hist_name) {
              MoFEMFunctionBeginHot;
              for (auto &&mit : neumann_el) {
                if constexpr (std::is_same_v<T, SNES>)
                  mit->second->methodsOp.push_back(
                      new LoadScale(snesLambdaLoadFactorPtr));
                if constexpr (std::is_same_v<T, TS>)
                  mit->second->methodsOp.push_back(
                      new TimeForceScale(getHistoryParam(hist_name), false));
                string element_name = mit->first;
                CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS",
                                      mit->second->getLoopFe(), false, false);
                // CHKERR push_fmethods(&mit->second->getLoopFe(), element_name);
                CHKERR function(dM, element_name.c_str(),
                                &mit->second->getLoopFe(), NULL, NULL);
              }
              MoFEMFunctionReturnHot(0);
            };

            CHKERR set_neumann_methods(neumann_forces, "force");
            CHKERR set_neumann_methods(nodal_forces, "force");
            CHKERR set_neumann_methods(edge_forces, "force");

            if (std::is_same_v<T, TS>)
              dirichletBcPtr->methodsOp.push_back(
                  new TimeForceScale(getHistoryParam("dirichlet"), false));

            CHKERR function(dM, domainElementName.c_str(), dirichletBcPtr.get(),
                            dirichletBcPtr.get(), dirichletBcPtr.get());
            CHKERR function(dM, domainElementName.c_str(),
                            bodyForceRhsPtr.get(), NULL, NULL);
            CHKERR function(dM, "SPRING", springRhsPtr.get(), NULL, NULL);
            CHKERR function(dM, "DAMPER_FE", &damperElementPtr->feRhs, NULL,
                            NULL);
            CHKERR function(dM, "FLUID_PRESSURE_FE",
                            &fluidPressureElementPtr->getLoopFe(), NULL, NULL);
          } else {

            CHKERR jacobian(dM, domainElementName.c_str(), dirichletBcPtr.get(),
                            dirichletBcPtr.get(), dirichletBcPtr.get());
            CHKERR jacobian(dM, domainElementName.c_str(),
                            bodyForceLhsPtr.get(), NULL, NULL);
            CHKERR jacobian(dM, "SPRING", springLhsPtr.get(), NULL, NULL);

            CHKERR jacobian(dM, "DAMPER_FE", &damperElementPtr->feLhs, NULL,
                            NULL);
          }

          MoFEMFunctionReturnHot(0);
        };

    if constexpr (std::is_same_v<T, SNES>) {

      if (!snesLambdaLoadFactorPtr)
        SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
                "SNES lambda factor pointer not set in the module constructor");

        CHKERR set_solver_pipelines(&DMMoFEMSNESSetFunction,
                                    &DMMoFEMSNESSetJacobian);

    } else if constexpr (std::is_same_v<T, TS>) {

      switch (type) {
      case IM:
        CHKERR set_solver_pipelines(&DMMoFEMTSSetIFunction,
                                    &DMMoFEMTSSetIJacobian);
        break;
      case IM2:
        CHKERR set_solver_pipelines(&DMMoFEMTSSetI2Function,
                                    &DMMoFEMTSSetI2Jacobian);
        break;
      case EX:
        CHKERR set_solver_pipelines(&DMMoFEMTSSetRHSFunction,
                                    &DMMoFEMTSSetRHSJacobian);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
                "This TS is not yet implemented for basic BCs");
        break;
      }

    } else
      static_assert(!std::is_same_v<T, KSP>,
                    "this solver has not been implemented for basic BCs yet");


    MoFEMFunctionReturnHot(0);
  }

  // template <> MoFEMErrorCode setupSolverFunction<SNES>(const TSType type);
  // template <> MoFEMErrorCode setupSolverJacobian<SNES>(const TSType type);
  // template <> MoFEMErrorCode setupSolverFunction<TS>(const TSType type);
  // template <> MoFEMErrorCode setupSolverJacobian<TS>(const TSType type);


  MoFEMErrorCode setupSolverFunctionSNES() override {
    MoFEMFunctionBegin;
    CHKERR this->setupSolverFunction<SNES>();
    MoFEMFunctionReturn(0);
  }
  MoFEMErrorCode setupSolverJacobianSNES() override {
    MoFEMFunctionBegin;
    CHKERR this->setupSolverJacobian<SNES>();
    MoFEMFunctionReturn(0);
  }
  MoFEMErrorCode
  setupSolverFunctionTS(const TSType type) override {
    MoFEMFunctionBegin;
    CHKERR this->setupSolverFunction<TS>(type);
    MoFEMFunctionReturn(0);
  }
  MoFEMErrorCode
  setupSolverJacobianTS(const TSType type) override {
    MoFEMFunctionBegin;
    CHKERR this->setupSolverJacobian<TS>(type);
    MoFEMFunctionReturn(0);
  }

};

#endif //__BASICBOUNDARYCONDIONSINTERFACE_HPP__