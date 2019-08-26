#ifndef __RDOPERATORS_HPP__
#define __RDOPERATORS_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

namespace ReactionDiffusion {

using FaceEle = MoFEM::FaceElementForcesAndSourcesCoreSwitch<
    FaceElementForcesAndSourcesCore::NO_HO_GEOMETRY |
    FaceElementForcesAndSourcesCore::NO_CONTRAVARIANT_TRANSFORM_HDIV |
    FaceElementForcesAndSourcesCore::NO_COVARIANT_TRANSFORM_HCURL>;

using BoundaryEle = MoFEM::EdgeElementForcesAndSourcesCoreSwitch<
    EdgeElementForcesAndSourcesCore::NO_HO_GEOMETRY |
    EdgeElementForcesAndSourcesCore::NO_COVARIANT_TRANSFORM_HCURL>;

using OpFaceEle = FaceEle::UserDataOperator;
using OpBoundaryEle = BoundaryEle::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;


struct NaturalBoundaryValues {
  NaturalBoundaryValues(EntityHandle fe_entity, VectorDouble &gauss_point)
  : fE_entity(fe_entity)
  , gPoint(gauss_point)
  {}

  MoFEMErrorCode set_value(double &out_value) { 
    MoFEMFunctionBegin;
    out_value = 0;
    MoFEMFunctionReturn(0);
  }
  EntityHandle fE_entity;
  VectorDouble gPoint;
};

struct EssentialBoundaryValues {
  EssentialBoundaryValues(EntityHandle fe_entity, VectorDouble &gauss_point)
      : fE_entity(fe_entity), gPoint(gauss_point) {}

  MoFEMErrorCode set_value(double &out_value) {
    MoFEMFunctionBegin;
    out_value = 0;
    MoFEMFunctionReturn(0);
  }
  EntityHandle fE_entity;
  VectorDouble gPoint;
}; 

struct InitialValues {
  InitialValues(VectorDouble &gauss_point)
  : gPoint(gauss_point)
  {}
  MoFEMErrorCode set_value(double &out_value) {
    MoFEMFunctionBegin;
    out_value = 0.5;
    MoFEMFunctionReturn(0);
  }

  VectorDouble gPoint;
};

}; // namespace ReactionDiffusion

#endif //__RDOPERATORS_HPP__