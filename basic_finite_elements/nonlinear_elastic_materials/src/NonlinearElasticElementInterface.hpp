/** \file NonlinearElasticElementInterface.hpp
  \brief Header file for NonlinearElasticElementInterface element implementation
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
  BitRefLevel bIt;
  boost::shared_ptr<NonlinearElasticElement> elasticElementPtr;
  boost::shared_ptr<ElasticMaterials> elasticMaterialsPtr;
  boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProcMeshPtr;

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
    CHKERR PetscOptionsGetBool(PETSC_NULL, "-is_quasi_static", &isQuasiStatic,
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


    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode setOperators() {
    MoFEMFunctionBegin;
    CHKERR elasticElementPtr->setOperators(positionField, meshNodeField, false,
                                           isDisplacementField);
    MoFEMFunctionReturn(0);
  }
  
  BitRefLevel &getElementBitRefLevel() { return bIt; };
  MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) {
    MoFEMFunctionBeginHot;
    this->dM = dm;
    CHKERR DMMoFEMAddElement(dM, "ELASTIC");

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

    switch (type) {
    case IM:
      CHKERR DMMoFEMTSSetIJacobian(
          dM, "ELASTIC", &elasticElementPtr->getLoopFeLhs(), NULL, NULL);
      break;
    case IM2:
      CHKERR DMMoFEMTSSetI2Jacobian(
          dM, "ELASTIC", &elasticElementPtr->getLoopFeLhs(), NULL, NULL);
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
      CHKERR DMMoFEMTSSetIFunction(
          dM, "ELASTIC", &elasticElementPtr->getLoopFeLhs(), NULL, NULL);
      break;
    case IM2:
      CHKERR DMMoFEMTSSetI2Function(
          dM, "ELASTIC", &elasticElementPtr->getLoopFeLhs(), NULL, NULL);
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
    if (!postProcMeshPtr) {
      postProcMeshPtr = boost::make_shared<PostProcVolumeOnRefinedMesh>(mField);
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

    elasticElementPtr->getLoopFeEnergy().snes_ctx = SnesMethod::CTX_SNESNONE;
    elasticElementPtr->getLoopFeEnergy().eNergy = 0;
    MOFEM_LOG("WORLD", Sev::inform) << "Loop energy\n";
    CHKERR DMoFEMLoopFiniteElements(dM, "ELASTIC",
                                    &elasticElementPtr->getLoopFeEnergy());
    // Print elastic energy
    MOFEM_LOG_C("WORLD", Sev::inform, "Elastic energy %6.4e\n",
                elasticElementPtr->getLoopFeEnergy().eNergy);

    CHKERR DMoFEMLoopFiniteElements(dM, "ELASTIC", postProcMeshPtr);
    auto out_name = "out_" + to_string(step) + ".h5m";
    MOFEM_LOG_C("WORLD", Sev::inform, "out file %s\n", out_name.c_str());
    CHKERR postProcMeshPtr->writeFile(out_name);

    MoFEMFunctionReturn(0);
  };
};

#endif //__NONLINEARELEMENTINTERFACE_HPP__