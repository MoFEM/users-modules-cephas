/**
 * @file CalculateTraction.hpp
 * @brief Calculate traction for linear problem
 * @date 2023-03-29
 *
 * @copyright Copyright (c) 2023
 *
 */

namespace ElasticExample {

struct OpCalculateTraction : public BoundaryEleOp {
  OpCalculateTraction(boost::shared_ptr<MatrixDouble> stress_ptr,
                      boost::shared_ptr<MatrixDouble> traction_ptr)
      : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE), stressPtr(stress_ptr),
        tractionPtr(traction_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    auto nb_int_pts = getGaussPts().size2();
    tractionPtr->resize(SPACE_DIM, nb_int_pts, false);
    auto t_normal = getFTensor1NormalsAtGaussPts();
    auto t_stress = getFTensor2SymmetricFromMat<SPACE_DIM>(*stressPtr);
    auto t_traction = getFTensor1FromMat<SPACE_DIM>(*tractionPtr);
    for (auto gg = 0; gg != nb_int_pts; ++gg) {
      const auto l = std::sqrt(t_normal(j) * t_normal(j));
      t_traction(i) = (t_stress(i, j) * t_normal(j)) / l;
      ++t_normal;
      ++t_stress;
      ++t_traction;
    }
    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> stressPtr;
  boost::shared_ptr<MatrixDouble> tractionPtr;
};

} // namespace ElasticExample