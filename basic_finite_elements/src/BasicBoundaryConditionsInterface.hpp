/** \file BasicBoundaryConditionsInterface.hpp
  \brief Header file for BasicBoundaryConditionsInterface element implementation
*/



// TODO: This is still work in progress

#ifndef __BASICBOUNDARYCONDIONSINTERFACE_HPP__
#define __BASICBOUNDARYCONDIONSINTERFACE_HPP__

/** \brief Set of functions declaring elements and setting operators
 * for basic boundary conditions interface
 */
struct BasicBoundaryConditionsInterface : public GenericElementInterface {

  MoFEM::Interface &mField;
  SmartPetscObj<DM> dM;
  PetscInt oRder;

  double *snesLambdaLoadFactor;

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

  BitRefLevel bIt;

  string positionField;
  string meshNodeField;

  boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
  boost::ptr_map<std::string, NodalForce> nodal_forces;
  boost::ptr_map<std::string, EdgeForce> edge_forces;

  boost::shared_ptr<FaceElementForcesAndSourcesCore> springRhsPtr;
  boost::shared_ptr<FaceElementForcesAndSourcesCore> springLhsPtr;
  boost::shared_ptr<FEMethod> dirichletBcPtr;

  const string elasticProblemName;
  using BcDataTuple = std::tuple<VectorDouble, VectorDouble,
                                 boost::shared_ptr<MethodForForceScaling>>;

  std::map<int, BcDataTuple> bodyForceMap;
  std::map<int, BcDataTuple> dispConstraintMap;

  boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProc;
  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcSkin;

  BasicBoundaryConditionsInterface(
      MoFEM::Interface &m_field, string postion_field,
      string mesh_pos_field_name = "MESH_NODE_POSITIONS",
      string problem_name = "ELASTIC", bool is_displacement_field = true,
      bool is_quasi_static = true, double *snes_load_factor = nullptr,
      bool is_partitioned = true)
      : mField(m_field), positionField(postion_field),
        meshNodeField(mesh_pos_field_name), elasticProblemName(problem_name),
        isDisplacementField(is_displacement_field),
        isQuasiStatic(is_quasi_static), snesLambdaLoadFactor(snes_load_factor),
        isPartitioned(is_partitioned) {
    oRder = 1;
  }

  ~BasicBoundaryConditionsInterface() { delete snesLambdaLoadFactor; }

  MoFEMErrorCode getCommandLineParameters() { return 0; };

  MoFEMErrorCode addElementFields() {
    MoFEMFunctionBeginHot;
    // Add spring boundary condition applied on surfaces.
    CHKERR MetaSpringBC::addSpringElements(mField, positionField,
                                           meshNodeField);
    CHKERR MetaNeumannForces::addNeumannBCElements(mField, positionField);
    CHKERR MetaNodalForces::addElement(mField, positionField);
    CHKERR MetaEdgeForces::addElement(mField, positionField);

    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode createElements() {
    MoFEMFunctionBeginHot;
    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode setOperators() {
    MoFEMFunctionBeginHot;
    CHKERR MetaNodalForces::setOperators(mField, nodal_forces, PETSC_NULL,
                                         positionField);
    CHKERR MetaEdgeForces::setOperators(mField, edge_forces, PETSC_NULL,
                                        positionField);

    CHKERR MetaNeumannForces::setMomentumFluxOperators(
        mField, neumann_forces, PETSC_NULL, positionField);
    springLhsPtr = boost::make_shared<FaceElementForcesAndSourcesCore>(mField);
    springRhsPtr = boost::make_shared<FaceElementForcesAndSourcesCore>(mField);

    CHKERR MetaSpringBC::setSpringOperators(mField, springLhsPtr, springRhsPtr,
                                            positionField,
                                            "MESH_NODE_POSITIONS");

    MoFEMFunctionReturnHot(0);
  };

  BitRefLevel getBitRefLevel() { return bIt; };
  MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) {
    MoFEMFunctionBeginHot;
    this->dM = dm;
    CHKERR DMMoFEMAddElement(dM, "FORCE_FE");
    CHKERR DMMoFEMAddElement(dM, "PRESSURE_FE");
    CHKERR DMMoFEMAddElement(dM, "SPRING");

    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode setupSolverJacobianSNES() {
    MoFEMFunctionBegin;
    // DMMoFEMSNESSetJacobian
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
    MoFEMFunctionReturn(0);
  };

  MoFEMErrorCode setupSolverFunctionSNES() {
    MoFEMFunctionBegin;
    // DMMoFEMSNESSetFunction
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
    if (!snesLambdaLoadFactor)
      MOFEM_LOG("WORLD", Sev::error)
          << "SNES Lambda factor not specified for this type of solver. (hint: "
             "check constructor of the interface";

    MoFEMFunctionReturn(0);
  };

  MoFEMErrorCode setupSolverJacobianTS(const TSType type) {
    MoFEMFunctionBegin;

    switch (type) {
    case IM:
      break;
    case IM2:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
      break;
    }
    MoFEMFunctionReturn(0);
  };

  MoFEMErrorCode setupSolverFunctionTS(const TSType type) {
    MoFEMFunctionBegin;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
    switch (type) {
    case IM:
      break;
    case IM2:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
      break;
    }
    MoFEMFunctionReturn(0);
  };

  MoFEMErrorCode updateElementVariables() { return 0; };
  MoFEMErrorCode postProcessElement(int step) { return 0; };
};

#endif //__BASICBOUNDARYCONDIONSINTERFACE_HPP__