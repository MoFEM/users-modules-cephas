/* \fiele MethodForForceScaling.hpp

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

#ifndef __METHOD_FOR_FORCE_SCALING_HPP__
#define __METHOD_FOR_FORCE_SCALING_HPP__

/// Class used to scale loads, f.e. in arc-length control
struct MethodForForceScaling {

  virtual MoFEMErrorCode scaleNf(const FEMethod *fe, VectorDouble &Nf) = 0;
  virtual MoFEMErrorCode getForceScale(const double ts_t, double &scale) {
    MoFEMFunctionBeginHot;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "not implemented");
    MoFEMFunctionReturnHot(0);
  }

  static MoFEMErrorCode
  applyScale(const FEMethod *fe,
             boost::ptr_vector<MethodForForceScaling> &methods_op,
             VectorDouble &nf) {
    MoFEMFunctionBegin;
    for (auto vit = methods_op.begin(); vit != methods_op.end(); vit++)
      CHKERR vit->scaleNf(fe, nf);
    MoFEMFunctionReturn(0);
  }

  static MoFEMErrorCode
  applyScale(const FEMethod *fe,
             boost::shared_ptr<MethodForForceScaling> method_op,
             VectorDouble &nf) {
    MoFEMFunctionBegin;
    CHKERR method_op->scaleNf(fe, nf);
    MoFEMFunctionReturn(0);
  }

  virtual ~MethodForForceScaling() {}
};

#endif //__METHOD_FOR_FORCE_SCALING_HPP__
