/* \fiele MethodForForceScaling.hpp

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
