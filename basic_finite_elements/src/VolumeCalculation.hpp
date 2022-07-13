/** \file VolumeCalculation.hpp

 * \brief Operator can be used with any volume element to calculate sum of
 * volumes of all volumes in the set

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
