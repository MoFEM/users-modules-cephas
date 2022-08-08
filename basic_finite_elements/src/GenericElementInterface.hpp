/** \file GenericElementInterface.hpp
  \brief Header file for GenericElementInterface element implementation
*/



#ifndef __GENERICELEMENTINTERFACE_HPP__
#define __GENERICELEMENTINTERFACE_HPP__

/** \brief Set of functions declaring elements and setting operators
 * for generic element interface
 */
struct GenericElementInterface {

  enum TSType { EX, IM, IM2, IMEX, DEFAULT };

  // GenericElementInterface() = delete;
  GenericElementInterface() {}
  virtual ~GenericElementInterface() {}

  virtual MoFEMErrorCode getCommandLineParameters() { return 0; };
  virtual MoFEMErrorCode addElementFields() = 0;
  virtual MoFEMErrorCode createElements() = 0;
  virtual BitRefLevel getBitRefLevel() { return BitRefLevel().set(); };
  virtual BitRefLevel getBitRefLevelMask() { return BitRefLevel().set(); };
  virtual MoFEMErrorCode setOperators() = 0;
  virtual MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) = 0;

  // virtual MoFEMErrorCode setupSolverKSP() = 0;
  // virtual MoFEMErrorCode setupSolverTAO() = 0;

  virtual MoFEMErrorCode setupSolverJacobianSNES() {
    MoFEMFunctionBeginHot;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
    MoFEMFunctionReturnHot(0);
  };

  virtual MoFEMErrorCode setupSolverFunctionSNES() {
    MoFEMFunctionBeginHot;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
    MoFEMFunctionReturnHot(0);
  };

  virtual MoFEMErrorCode setupSolverJacobianTS(const TSType type) = 0;
  virtual MoFEMErrorCode setupSolverFunctionTS(const TSType type) = 0;

  virtual MoFEMErrorCode updateElementVariables() { return 0; };
  virtual MoFEMErrorCode postProcessElement(int step) = 0;
};

#endif //__GENERICELEMENTINTERFACE_HPP__