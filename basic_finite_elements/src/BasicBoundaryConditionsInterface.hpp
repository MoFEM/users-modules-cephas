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

#ifndef __GENERICELEMENTINTERFACE_HPP__
#define __GENERICELEMENTINTERFACE_HPP__

/** \brief Set of functions declaring elements and setting operators
 * for generic element interface
 */
struct BasicBoundaryConditionsInterface {

  MoFEM::Interface &mField;
  SmartPetscObj<DM> dM;
  PetscInt oRder;

  bool isDisplacementField;
  BitRefLevel bIt;

  string positionField;
  string meshNodeField;

  boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
  boost::ptr_map<std::string, NodalForce> nodal_forces;
  boost::ptr_map<std::string, EdgeForce> edge_forces;

  string elasticProblemName;
  using BcDataTuple = std::tuple<VectorDouble, VectorDouble,
                                 boost::shared_ptr<MethodForForceScaling>>;

  BasicBoundaryConditionsInterface(MoFEM::Interface &m_field,
                                   string postion_field,
                                   string mesh_posi_field_name = "MESH_NODE_POSITIONS",
                                   bool is_displacement_field = true,
                                   bool is_quasi_static = true)
      : mField(m_field), positionField(postion_field),
        meshNodeField(mesh_posi_field_name),
        isDisplacementField(is_displacement_field),
        isQuasiStatic(is_quasi_static) {
    oRder = 1;
  }

  BasicBoundaryConditionsInterface() {}

  ~BasicBoundaryConditionsInterface() {}

  MoFEMErrorCode getCommandLineParameters() { return 0; };

  MoFEMErrorCode addElementFields() {
    MoFEMFunctionBeginHot;
    auto simple = mField.getInterface<Simple>();

    CHKERR MetaNeumannForces::addNeumannBCElements(mField, positionField);
    CHKERR MetaNodalForces::addElement(mField, positionField);
    CHKERR MetaEdgeForces::addElement(mField, positionField);

    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode createElements() {
    MoFEMFunctionBeginHot;

    CHKERR MetaNeumannForces::setMomentumFluxOperators(mField, neumann_forces,
                                                       PETSC_NULL, "U");
    CHKERR MetaNodalForces::setOperators(mField, nodal_forces, PETSC_NULL, "U");
    CHKERR MetaEdgeForces::setOperators(mField, edge_forces, PETSC_NULL, "U");
    
    MoFEMFunctionReturnHot(0);
  };

  BitRefLevel &getElementBitRefLevel() { return bIt; };
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


    MoFEMFunctionReturn(0);
  };
  MoFEMErrorCode setupSolverFunctionSNES() {
    MoFEMFunctionBegin;
    
    
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
  MoFEMErrorCode postProcessElement(int step) {
    MoFEMFunctionBegin;

    MoFEMFunctionReturn(0);
  };
};

#endif //__GENERICELEMENTINTERFACE_HPP__