/** \file GenericElementInterface.hpp
  \brief Header file for GenericElementInterface element implementation
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
struct GenericElementInterface {

  enum TSType { EX, IM, IM2, IMEX, DEFAULT };

  // MoFEM::Interface &mField;
  // GenericElementInterface(MoFEM::Interface &m_field) : mField(m_field) {}
  // GenericElementInterface() = delete;
  GenericElementInterface() {}
  virtual ~GenericElementInterface() {}

  virtual MoFEMErrorCode getCommandLineParameters() = 0;
  virtual MoFEMErrorCode addElementFields() = 0;
  virtual MoFEMErrorCode createElements() = 0;
  virtual BitRefLevel &getElementBitRefLevel() = 0;
  virtual MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) = 0;

  // virtual MoFEMErrorCode setupSolverKSP() = 0;
  //   virtual MoFEMErrorCode setupSolverTAO() = 0;
  virtual MoFEMErrorCode setupSolverJacobianSNES() = 0;
  virtual MoFEMErrorCode setupSolverFunctionSNES() = 0;

  virtual MoFEMErrorCode setupSolverJacobianTS(const TSType type) = 0;
  virtual MoFEMErrorCode setupSolverFunctionTS(const TSType type) = 0;

  virtual MoFEMErrorCode updateElementVariables() = 0;
  virtual MoFEMErrorCode postProcessElement(int step) = 0;
};

#endif //__GENERICELEMENTINTERFACE_HPP__