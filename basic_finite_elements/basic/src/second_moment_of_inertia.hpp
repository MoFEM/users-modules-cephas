using Element = MoFEM::VolumeElementForcesAndSourcesCoreBase;
using OpElement = Element::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

struct OpZero : public OpElement {
  OpZero(boost::shared_ptr<CommonData> &common_data_ptr)
      : OpElement("rho", OPROW), commonDataPtr(common_data_ptr) {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpFirst : public OpElement {
  OpFirst(boost::shared_ptr<CommonData> &common_data_ptr)
      : OpElement("rho", OPROW), commonDataPtr(common_data_ptr) {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpSecond : public OpElement {
  OpSecond(boost::shared_ptr<CommonData> &common_data_ptr)
      : OpElement("rho", OPROW), commonDataPtr(common_data_ptr) {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

MoFEMErrorCode OpZero::doWork(int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  if (type == MBVERTEX) {
    const int nb_integration_pts = getGaussPts().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_rho =
        getFTensor0FromVec(*(commonDataPtr->rhoAtIntegrationPts));
    FTensor::Index<'i', 3> i;
    const double volume = getVolume();
    double element_local_value = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      element_local_value += t_w * t_rho * volume;
      ++t_w;
      ++t_rho;
    }
    const int index = CommonData::ZERO;
    CHKERR VecSetValue(commonDataPtr->petscVec, index, element_local_value,
                       ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode OpFirst::doWork(int side, EntityType type,
                                                  EntData &data) {
  MoFEMFunctionBegin;
  if (type == MBVERTEX) {
    const int nb_integration_pts = getGaussPts().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_rho =
        getFTensor0FromVec(*(commonDataPtr->rhoAtIntegrationPts));
    auto t_coords = getFTensor1CoordsAtGaussPts();
    const double volume = getVolume();

    VectorDouble3 element_local_value(3);
    FTensor::Index<'i', 3> i;
    auto t_s = FTensor::Tensor1<FTensor::PackPtr<double *, 0>, 3>(
        &element_local_value[0], &element_local_value[1],
        &element_local_value[2]);

    t_s(i) = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      t_s(i) += t_w * t_rho * volume * t_coords(i);
      ++t_w;
      ++t_rho;
      ++t_coords;
    }
    constexpr std::array<int, 3> indices = {
        CommonData::FIRST_X, CommonData::FIRST_Y, CommonData::FIRST_Z};
    CHKERR VecSetValues(commonDataPtr->petscVec, 3, indices.data(),
                          &element_local_value[0], ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode OpSecond::doWork(int side, EntityType type,
                                                   EntData &data) {
  MoFEMFunctionBegin;
  if (type == MBVERTEX) {
    const int nb_integration_pts = getGaussPts().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_rho =
        getFTensor0FromVec(*(commonDataPtr->rhoAtIntegrationPts));
    auto t_coords = getFTensor1CoordsAtGaussPts();
    const double volume = getVolume();

    VectorDouble6 element_local_value(6);
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Tensor2_symmetric<FTensor::PackPtr<double *, 0>, 3> t_I(
        &element_local_value[0], &element_local_value[1],
        &element_local_value[2], &element_local_value[3],
        &element_local_value[4], &element_local_value[5]);

    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      t_I(i, j) += (t_w * t_rho * volume) * (t_coords(i) ^ t_coords(j));
      ++t_w;
      ++t_rho;
      ++t_coords;
    }

    constexpr std::array<int, 6> indices = {
        CommonData::SECOND_XX, CommonData::SECOND_XY, CommonData::SECOND_XZ,
        CommonData::SECOND_YY, CommonData::SECOND_YZ, CommonData::SECOND_ZZ};
    CHKERR VecSetValues(commonDataPtr->petscVec, 6, indices.data(),
                        &element_local_value[0], ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}
