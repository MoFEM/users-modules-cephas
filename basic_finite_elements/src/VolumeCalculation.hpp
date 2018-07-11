/** \file VolumeCalculation.hpp

 * \brief Operator can be used with any volume element to calculate sum of
 * volumes of all volumes in the set

 */

/*
 * This file is part of MoFEM.
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

#ifndef __VOLUME_CALCULATION_HPP__
#define __VOLUME_CALCULATION_HPP__

struct VolumeCalculation: public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

  Vec volumeVec;

  VolumeCalculation(const std::string &field_name,Vec volume_vec):
  MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(field_name,UserDataOperator::OPROW),
  volumeVec(volume_vec) {
  }

  

  MoFEMErrorCode doWork(
    int row_side,EntityType row_type,DataForcesAndSourcesCore::EntData &row_data
  ) {
    MoFEMFunctionBeginHot;

    //do it only once, no need to repeat this for edges,faces or tets
    if(row_type != MBVERTEX) MoFEMFunctionReturnHot(0);

    int nb_gauss_pts = row_data.getN().size1();
    for(int gg = 0;gg<nb_gauss_pts;gg++) {

      double vol = getVolume()*getGaussPts()(3,gg);
      if(getHoGaussPtsDetJac().size()>0) {
        vol *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
      }

      ierr = VecSetValue(volumeVec,0,vol,ADD_VALUES); CHKERRG(ierr);

    }

    MoFEMFunctionReturnHot(0);
  }

};

#endif //__VOLUME_CALCULATION_HPP__
