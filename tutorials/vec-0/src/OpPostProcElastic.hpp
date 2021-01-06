/**
 * \file OpPostProcElastic.hpp
 * \example OpPostProcElastic.hpp
 *
 * Postprocessing
 *
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

namespace Tutorial {

//! [Class definition]
template <int DIM> struct OpPostProcElastic : public DomainEleOp {
  OpPostProcElastic(const std::string field_name,
                    moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    boost::shared_ptr<MatrixDouble> m_strain_ptr,
                    boost::shared_ptr<MatrixDouble> m_stress_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<MatrixDouble> mStrainPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;
};
//! [Class definition]

//! [Postprocessing constructor]
template <int DIM>
OpPostProcElastic<DIM>::OpPostProcElastic(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<MatrixDouble> m_strain_ptr,
    boost::shared_ptr<MatrixDouble> m_stress_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), mStrainPtr(m_strain_ptr),
      mStressPtr(m_stress_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}
//! [Postprocessing constructor]

//! [Postprocessing]
template <int DIM>
MoFEMErrorCode OpPostProcElastic<DIM>::doWork(int side, EntityType type,
                                              EntData &data) {
  MoFEMFunctionBegin;

  auto get_tag = [&](const std::string name) {
    std::array<double, 9> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), 9, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3, 3);

  auto set_matrix_symm = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != DIM; ++r)
      for (size_t c = 0; c != DIM; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_plain_stress_strain = [&](auto &mat, auto &t) -> MatrixDouble3by3 & {
    mat(2, 2) = -poisson_ratio * (t(0, 0) + t(1, 1));
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_strain = get_tag("STRAIN");
  auto th_stress = get_tag("STRESS");

  size_t nb_gauss_pts = data.getN().size1();
  auto t_strain = getFTensor2SymmetricFromMat<DIM>(*(mStrainPtr));
  auto t_stress = getFTensor2SymmetricFromMat<DIM>(*(mStressPtr));

  switch (DIM) {
  case 2:
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      CHKERR set_tag(
          th_strain, gg,
          set_plain_stress_strain(set_matrix_symm(t_strain), t_stress));
      CHKERR set_tag(th_stress, gg, set_matrix_symm(t_stress));
      ++t_strain;
      ++t_stress;
    }
    break;
  case 3:
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      CHKERR set_tag(th_strain, gg, set_matrix_symm(t_strain));
      CHKERR set_tag(th_stress, gg, set_matrix_symm(t_stress));
      ++t_strain;
      ++t_stress;
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Not implemeneted dimension");
  }

  MoFEMFunctionReturn(0);
}
//! [Postprocessing]

} // namespace Tutorial