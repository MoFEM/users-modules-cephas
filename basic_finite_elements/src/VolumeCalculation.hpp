/** \file VolumeCalculation.hpp

 * \brief Operator can be used with any volume element to calculate sum of
 * volumes of all volumes in the set

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

#ifndef __VOLUME_CALCULATION_HPP__
#define __VOLUME_CALCULATION_HPP__

/**
 * @brief Calculate volume 
 * 
 */
struct VolumeCalculation
    : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

  Vec volumeVec;

  VolumeCalculation(const std::string &field_name, Vec volume_vec)
      : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
            field_name, UserDataOperator::OPROW),
        volumeVec(volume_vec) {}

  MoFEMErrorCode doWork(int row_side, EntityType row_type,
                        EntitiesFieldData::EntData &row_data) {
    MoFEMFunctionBegin;

    // do it only once, no need to repeat this for edges,faces or tets
    if (row_type != MBVERTEX)
      MoFEMFunctionReturnHot(0);

    int nb_gauss_pts = row_data.getN().size1();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double vol = getVolume() * getGaussPts()(3, gg);

      CHKERR VecSetValue(volumeVec, 0, vol, ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }
};

#endif //__VOLUME_CALCULATION_HPP__
