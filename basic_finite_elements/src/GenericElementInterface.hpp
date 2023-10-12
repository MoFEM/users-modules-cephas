/** \file GenericElementInterface.hpp
  \brief Header file for GenericElementInterface element implementation
*/



#ifndef __GENERICELEMENTINTERFACE_HPP__
#define __GENERICELEMENTINTERFACE_HPP__

/** \brief Set of functions declaring elements and setting operators
 * for generic element interface
 */
struct GenericElementInterface {

  // typedef const char *TSType;
  enum TSType { EX, IM, IM2, IMEX, DEFAULT };
  using BcMarkerPtr = boost::shared_ptr<std::vector<char unsigned>>;


  BcMarkerPtr mBoundaryMarker;
  int atomTest;
  int restartRunStep;

  boost::shared_ptr<FEMethod> monitorPtr;

  // TODO: members to add
  // add global/common boundary marker
  // add global/common post processing
  // add global/common post processing on skin

  GenericElementInterface() {}
  virtual ~GenericElementInterface() {}

  virtual MoFEMErrorCode setGlobalBoundaryMarker(BcMarkerPtr mark) {
    MoFEMFunctionBeginHot;
    mBoundaryMarker = mark;
    MoFEMFunctionReturnHot(0);
  }

  virtual BcMarkerPtr getGlobalBoundaryMarker() { return mBoundaryMarker; };

  virtual MoFEMErrorCode setMonitorPtr(boost::shared_ptr<FEMethod> monitor_ptr) {
    MoFEMFunctionBeginHot;
    monitorPtr = monitor_ptr;
    MoFEMFunctionReturnHot(0);
    return 0;
  }

  // get optional command line parameters
  virtual MoFEMErrorCode getCommandLineParameters() { return 0; };
  virtual MoFEMErrorCode addElementFields() = 0;
  virtual MoFEMErrorCode createElements() = 0;
  // create and add finite elements to the DM
  virtual MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) = 0;
  
  virtual BitRefLevel getBitRefLevel() { return BitRefLevel().set(); };
  virtual BitRefLevel getBitRefLevelMask() { return BitRefLevel().set(); };
  virtual MoFEMErrorCode setOperators() = 0;
  // setup operators and boundary conditions

  // virtual MoFEMErrorCode setupSolverKSP() = 0;
  // virtual MoFEMErrorCode setupSolverTAO() = 0;
  virtual MoFEMErrorCode setupSolverJacobianSNES() {
    MoFEMFunctionBeginHot;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "SNES not implemented");
    MoFEMFunctionReturnHot(0);
  };

  virtual MoFEMErrorCode setupSolverFunctionSNES() {
    MoFEMFunctionBeginHot;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "SNES not implemented");
    MoFEMFunctionReturnHot(0);
  };

  virtual MoFEMErrorCode setupSolverJacobianTS(const TSType type) = 0;
  virtual MoFEMErrorCode setupSolverFunctionTS(const TSType type) = 0;

  // function executed in the monitor or at each SNES iteration
  virtual MoFEMErrorCode updateElementVariables() { return 0; };
  // postprocessing executed in the monitor or at each SNES iteration
  virtual MoFEMErrorCode postProcessElement(int step) = 0;

  virtual MoFEMErrorCode testOperators() { return 0; };
};

#endif //__GENERICELEMENTINTERFACE_HPP__