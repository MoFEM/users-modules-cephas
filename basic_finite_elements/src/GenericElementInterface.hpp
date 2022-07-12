/** \file GenericElementInterface.hpp
  \brief Header file for GenericElementInterface element implementation
*/

/* MIT License
 *
 * Copyright (c) 2022
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
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