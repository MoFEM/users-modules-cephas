/* \fiele MethodForForceScaling.hpp

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
