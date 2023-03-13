/** \file NonlinearElasticElementInterface.hpp
* \example NonlinearElasticElementInterface.hpp

  \brief Header file for NonlinearElasticElementInterface element implementation
*/



#ifndef __NONLINEARELEMENTINTERFACE_HPP__
#define __NONLINEARELEMENTINTERFACE_HPP__

/** \brief Set of functions declaring elements and setting operators
 * for generic element interface
 */
struct NonlinearElasticElementInterface : public GenericElementInterface {

  MoFEM::Interface &mField;
  SmartPetscObj<DM> dM;
  PetscBool isQuasiStatic;
  
  PetscInt oRder;
  bool isDisplacementField;
  PetscBool postProcSkin;
  PetscBool postProcVolume;
  PetscBool calcEnergy;
  Range skinEnts;
  BitRefLevel bIt;
  boost::shared_ptr<NonlinearElasticElement> elasticElementPtr;
  boost::shared_ptr<ElasticMaterials> elasticMaterialsPtr;
  boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProcMeshPtr;
  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcSkinPtr;

  string positionField;
  string meshNodeField;

  NonlinearElasticElementInterface(MoFEM::Interface &m_field,
                                   string postion_field,
                                   string mesh_posi_field_name = "MESH_NODE_POSITIONS",
                                   bool is_displacement_field = true,
                                   PetscBool is_quasi_static = PETSC_TRUE)
      : mField(m_field), positionField(postion_field),
        meshNodeField(mesh_posi_field_name),
        isDisplacementField(is_displacement_field),
        isQuasiStatic(is_quasi_static) {
    oRder = 1;
  }

  ~NonlinearElasticElementInterface() {}

  MoFEMErrorCode getCommandLineParameters() {
    MoFEMFunctionBegin;
    isQuasiStatic = PETSC_FALSE;
    oRder = 2;
    postProcSkin = PETSC_TRUE;
    postProcVolume = PETSC_FALSE;
    calcEnergy = PETSC_TRUE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "-is_quasi_static", &isQuasiStatic,
                               PETSC_NULL);
    CHKERR PetscOptionsGetBool(PETSC_NULL, "-post_proc_skin", &postProcSkin,
                               PETSC_NULL);
    CHKERR PetscOptionsGetBool(PETSC_NULL, "-post_proc_volume", &postProcVolume,
                               PETSC_NULL);
    CHKERR PetscOptionsGetBool(PETSC_NULL, "-calc_energy", &calcEnergy,
                               PETSC_NULL);
    CHKERR PetscOptionsGetInt(PETSC_NULL, "-order", &oRder, PETSC_NULL);

    MoFEMFunctionReturn(0);
  };

  MoFEMErrorCode addElementFields() {
    MoFEMFunctionBeginHot;
    auto simple = mField.getInterface<Simple>();
    if (!mField.check_field(positionField)) {
      CHKERR simple->addDomainField(positionField, H1, AINSWORTH_LEGENDRE_BASE,
                                    3);
      CHKERR simple->addBoundaryField(positionField, H1,
                                      AINSWORTH_LEGENDRE_BASE, 3);
      CHKERR simple->setFieldOrder(positionField, oRder);
    }
    if (!mField.check_field(meshNodeField)) {
      CHKERR simple->addDataField(meshNodeField, H1, AINSWORTH_LEGENDRE_BASE,
                                  3);
      CHKERR simple->setFieldOrder(meshNodeField, 2);
    }

    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode getEntsOnMeshSkin() {
    MoFEMFunctionBeginHot;

    Range body_ents;
    auto meshset = mField.getInterface<Simple>()->getMeshSet();
    CHKERR mField.get_moab().get_entities_by_type(meshset, MBTET, body_ents);
    // cout << "body_ents for skin: " << body_ents.size() << endl;
    Skinner skin(&mField.get_moab());
    Range skin_ents;
    CHKERR skin.find_skin(0, body_ents, false, skin_ents);
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    if (pcomm == NULL) {
      pcomm = new ParallelComm(&mField.get_moab(), mField.get_comm());
    }
    CHKERR pcomm->filter_pstatus(skin_ents,
                                 PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                 PSTATUS_NOT, -1, &skinEnts);

    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode createElements() {
    MoFEMFunctionBeginHot;

    elasticElementPtr = boost::make_shared<NonlinearElasticElement>(mField, 2);
    elasticMaterialsPtr = boost::make_shared<ElasticMaterials>(mField, "HOOKE");
    CHKERR elasticMaterialsPtr->setBlocks(elasticElementPtr->setOfBlocks);

    CHKERR addHOOpsVol(meshNodeField, elasticElementPtr->getLoopFeRhs(), true,
                       false, false, false);
    CHKERR addHOOpsVol(meshNodeField, elasticElementPtr->getLoopFeLhs(), true,
                       false, false, false);
    CHKERR addHOOpsVol(meshNodeField, elasticElementPtr->getLoopFeEnergy(),
                       true, false, false, false);
    CHKERR elasticElementPtr->addElement("ELASTIC", positionField,
                                         meshNodeField, false);

    if (postProcSkin) {
      CHKERR getEntsOnMeshSkin();

      CHKERR mField.add_finite_element("POST_PROC_SKIN");
      CHKERR mField.modify_finite_element_add_field_row("POST_PROC_SKIN",
                                                        positionField);
      CHKERR mField.modify_finite_element_add_field_col("POST_PROC_SKIN",
                                                        positionField);
      CHKERR mField.modify_finite_element_add_field_data("POST_PROC_SKIN",
                                                         positionField);
      CHKERR mField.modify_finite_element_add_field_data("POST_PROC_SKIN",
                                                         meshNodeField);
      CHKERR mField.add_ents_to_finite_element_by_dim(skinEnts, 2,
                                                      "POST_PROC_SKIN");
    }

    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode setOperators() {
    MoFEMFunctionBegin;
    auto &pipeline_rhs = elasticElementPtr->feRhs.getOpPtrVector();
    auto &pipeline_lhs = elasticElementPtr->feLhs.getOpPtrVector();

    pipeline_rhs.push_back(new OpSetBc(positionField, true, mBoundaryMarker));
    pipeline_lhs.push_back(new OpSetBc(positionField, true, mBoundaryMarker));

    CHKERR elasticElementPtr->setOperators(positionField, meshNodeField, false,
                                           isDisplacementField);
                                           
    pipeline_rhs.push_back(new OpUnSetBc(positionField));
    pipeline_lhs.push_back(new OpUnSetBc(positionField));
    MoFEMFunctionReturn(0);
  }

  BitRefLevel getBitRefLevel() { return bIt; };
  MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) {
    MoFEMFunctionBeginHot;
    this->dM = dm;
    CHKERR DMMoFEMAddElement(dM, "ELASTIC");
    if (postProcSkin)
      CHKERR DMMoFEMAddElement(dM, "POST_PROC_SKIN");
    mField.getInterface<Simple>()->getOtherFiniteElements().push_back(
        "ELASTIC");
    mField.getInterface<Simple>()->getOtherFiniteElements().push_back(
        "POST_PROC_SKIN");
           
    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode setupSolverJacobianSNES() {
    MoFEMFunctionBegin;

    CHKERR DMMoFEMSNESSetJacobian(
        dM, "ELASTIC", &elasticElementPtr->getLoopFeLhs(), NULL, NULL);

    MoFEMFunctionReturn(0);
  };
  MoFEMErrorCode setupSolverFunctionSNES() {
    MoFEMFunctionBegin;
    CHKERR DMMoFEMSNESSetFunction(dM, "ELASTIC",
                                  &elasticElementPtr->getLoopFeRhs(),
                                  PETSC_NULL, PETSC_NULL);
    MoFEMFunctionReturn(0);
  };

  MoFEMErrorCode setupSolverJacobianTS(const TSType type) {
    MoFEMFunctionBegin;
    auto &method = elasticElementPtr->getLoopFeLhs();
    switch (type) {
    case IM:
      CHKERR DMMoFEMTSSetIJacobian(dM, "ELASTIC", &method, &method, &method);
      break;
    case IM2:
      CHKERR DMMoFEMTSSetI2Jacobian(dM, "ELASTIC", &method, &method, &method);
      break;
    case EX:
      CHKERR DMMoFEMTSSetRHSJacobian(dM, "ELASTIC", &method, &method, &method);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "This TS is not yet implemented");
      break;
    }
    MoFEMFunctionReturn(0);
  };

  MoFEMErrorCode setupSolverFunctionTS(const TSType type) {
    MoFEMFunctionBegin;
    auto &method = elasticElementPtr->getLoopFeRhs();
    switch (type) {
    case IM:
      CHKERR DMMoFEMTSSetIFunction(dM, "ELASTIC", &method, &method, &method);
      break;
    case IM2:
      CHKERR DMMoFEMTSSetI2Function(dM, "ELASTIC", &method, &method, &method);
      break;
    case EX:
      CHKERR DMMoFEMTSSetRHSFunction(dM, "ELASTIC", &method, &method, &method);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
      break;
    }

    MoFEMFunctionReturn(0);
  };

  MoFEMErrorCode updateElementVariables() { return 0; };
  MoFEMErrorCode postProcessElement(int step) {
    MoFEMFunctionBegin;
    
    if (elasticElementPtr->setOfBlocks.empty())
      MoFEMFunctionReturnHot(0);

    if (postProcVolume && !postProcMeshPtr) {
      postProcMeshPtr = boost::make_shared<PostProcVolumeOnRefinedMesh>(mField);

      if (mField.check_field("MESH_NODE_POSITIONS"))
        CHKERR addHOOpsVol("MESH_NODE_POSITIONS", *postProcMeshPtr, true, false,
                           false, false);
      CHKERR postProcMeshPtr->generateReferenceElementMesh();
      CHKERR postProcMeshPtr->addFieldValuesPostProc(positionField);
      CHKERR postProcMeshPtr->addFieldValuesPostProc(meshNodeField);
      CHKERR postProcMeshPtr->addFieldValuesGradientPostProc(positionField);

      for (auto &sit : elasticElementPtr->setOfBlocks) {
        postProcMeshPtr->getOpPtrVector().push_back(new PostProcStress(
            postProcMeshPtr->postProcMesh, postProcMeshPtr->mapGaussPts,
            positionField, sit.second, postProcMeshPtr->commonData,
            isDisplacementField));
      }
    }

    if (postProcSkin && !postProcSkinPtr) {
      postProcSkinPtr = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);

      if (mField.check_field("MESH_NODE_POSITIONS"))
        CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS", *postProcSkinPtr, false,
                              false);

      CHKERR postProcSkinPtr->generateReferenceElementMesh();
      CHKERR postProcSkinPtr->addFieldValuesPostProc(positionField);
      CHKERR postProcSkinPtr->addFieldValuesPostProc(meshNodeField);
      CHKERR postProcSkinPtr->addFieldValuesGradientPostProcOnSkin(
          positionField, "ELASTIC");
    }

    if (calcEnergy) {
      elasticElementPtr->getLoopFeEnergy().snes_ctx = SnesMethod::CTX_SNESNONE;
      elasticElementPtr->getLoopFeEnergy().eNergy = 0;
      // MOFEM_LOG("WORLD", Sev::inform) << "Loop energy\n";
      CHKERR DMoFEMLoopFiniteElements(dM, "ELASTIC",
                                      &elasticElementPtr->getLoopFeEnergy());

      auto E = elasticElementPtr->getLoopFeEnergy().eNergy;
      // Print elastic energy
      MOFEM_LOG_C("WORLD", Sev::inform, "%d Time %3.2e Elastic energy %3.2e",
                  step, elasticElementPtr->getLoopFeRhs().ts_t, E);
    }

    if (postProcVolume) {
      CHKERR DMoFEMLoopFiniteElements(dM, "ELASTIC", postProcMeshPtr);
      auto out_name = "out_vol_" + to_string(step) + ".h5m";
      CHKERR postProcMeshPtr->writeFile(out_name);
    }

    if (postProcSkin) {
      CHKERR DMoFEMLoopFiniteElements(dM, "POST_PROC_SKIN", postProcSkinPtr);
      auto out_name = "out_skin_" + to_string(step) + ".h5m";
      CHKERR postProcSkinPtr->writeFile(out_name);
    }

    MoFEMFunctionReturn(0);
  };
};

#endif //__NONLINEARELEMENTINTERFACE_HPP__