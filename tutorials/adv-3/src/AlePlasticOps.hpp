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

/** \file AlePlastic.hpp
 * \example AlePlastic.hpp
 */

namespace AlePlastic {

//! [Common data]
struct CommonData : public PlasticOps::CommonData {

  // MatrixDouble tempFluxVal;
  // VectorDouble templDivFlux;
  // VectorDouble tempValDot;

  // inline auto getTempFluxValPtr() {
  //   return boost::shared_ptr<MatrixDouble>(shared_from_this(), &tempFluxVal);
  // }
  // inline auto getTempValDotPtr() {
  //   return boost::shared_ptr<VectorDouble>(shared_from_this(), &tempValDot);
  // }
  // inline auto getTempDivFluxPtr() {
  //   return boost::shared_ptr<VectorDouble>(shared_from_this(), &templDivFlux);
  // }

  //for rotation
  boost::shared_ptr<VectorDouble> plasticTauJumpPtr;
  boost::shared_ptr<MatrixDouble> plasticStrainJumpPtr;
  boost::shared_ptr<VectorDouble> plastic_N_TauJumpPtr;
  boost::shared_ptr<MatrixDouble> plastic_N_StrainJumpPtr;
  boost::shared_ptr<MatrixDouble> guidingVelocityPtr;

  // data for skeleton computation
  map<int, MatrixDouble> plastic_N_StrainSideMap;
  map<int, VectorDouble> plastic_N_TauSideMap;
  map<int, MatrixDouble> plasticStrainSideMap;
  map<int, VectorDouble> plasticTauSideMap;
  map<int, MatrixDouble> velocityVecSideMap;
};
//! [Common data]


} // namespace AlePlastic