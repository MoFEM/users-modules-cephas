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

namespace OpContactTools {

//! [Common data]
struct CommonData {
  boost::shared_ptr<MatrixDouble> mDPtr;
  boost::shared_ptr<MatrixDouble> mGradPtr;
  boost::shared_ptr<MatrixDouble> mStrainPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;

  boost::shared_ptr<MatrixDouble> contactStressPtr;
  boost::shared_ptr<MatrixDouble> contactStressDivergencePtr;
  boost::shared_ptr<MatrixDouble> contactTractionPtr;
  boost::shared_ptr<MatrixDouble> contactDispPtr;

  boost::shared_ptr<MatrixDouble> curlContactStressPtr;
};
//! [Common data]

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;

struct OpInternalBoundaryContactRhs : public BoundaryEleOp {
  OpInternalBoundaryContactRhs(const std::string field_name,
                               boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainBoundaryRhs : public BoundaryEleOp {
  OpConstrainBoundaryRhs(const std::string field_name,
                         boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainBoundaryLhs_dU : public BoundaryEleOp {
  OpConstrainBoundaryLhs_dU(const std::string row_field_name,
                            const std::string col_field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpConstrainBoundaryLhs_dTraction : public BoundaryEleOp {
  OpConstrainBoundaryLhs_dTraction(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

template <typename T1, typename T2>
inline FTensor::Tensor1<double, SPACE_DIM>
normal(FTensor::Tensor1<T1, 3> &t_coords,
       FTensor::Tensor1<T2, SPACE_DIM> &t_disp) {
  FTensor::Tensor1<double, SPACE_DIM> t_normal;
  t_normal(i) = 0;
  t_normal(1) = 1.;
  return t_normal;
}

template <typename T>
inline double gap0(FTensor::Tensor1<T, 3> &t_coords,
                   FTensor::Tensor1<double, SPACE_DIM> &t_normal) {
  return (-0.5 - t_coords(1)) * t_normal(1);
}

template <typename T>
inline double gap(FTensor::Tensor1<T, SPACE_DIM> &t_disp,
                  FTensor::Tensor1<double, SPACE_DIM> &t_normal) {
  return t_disp(i) * t_normal(i);
}

template <typename T>
inline double normal_traction(FTensor::Tensor1<T, SPACE_DIM> &t_traction,
                              FTensor::Tensor1<double, SPACE_DIM> &t_normal) {
  return -t_traction(i) * t_normal(i);
}

inline double sign(double x) {
  if (x == 0)
    return 0;
  else if (x > 0)
    return 1;
  else
    return -1;
};

inline double w(const double g, const double t) { return g - cn * t; }

inline double constrian(double &&g0, double &&g, double &&t) {
  return (w(g - g0, t) + std::abs(w(g - g0, t))) / 2 + g0;
};

inline double diff_constrains_dtraction(double &&g0, double &&g, double &&t) {
  return -cn * (1 + sign(w(g - g0, t))) / 2;
}

inline double diff_constrains_dgap(double &&g0, double &&g, double &&t) {
  return (1 + sign(w(g - g0, t))) / 2;
}

OpInternalBoundaryContactRhs::OpInternalBoundaryContactRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpInternalBoundaryContactRhs::doWork(int side, EntityType type,
                                                    EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_traction =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));

    size_t nb_base_functions = data.getN().size2();
    auto t_base = data.getFTensor0N();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      auto t_nf = getFTensor1FromPtr<SPACE_DIM>(nf.data());

      const double alpha = t_w * getMeasure();

      size_t bb = 0;
      for (; bb != nb_dofs / SPACE_DIM; ++bb) {
        t_nf(i) -= alpha * t_base * t_traction(i);
        ++t_nf;
        ++t_base;
      }
      for (; bb != nb_base_functions; ++bb)
        ++t_base;

      ++t_traction;
      ++t_w;
    }

    // CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}


OpConstrainBoundaryRhs::OpConstrainBoundaryRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpConstrainBoundaryRhs::doWork(int side, EntityType type,
                                              EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_normal = getFTensor1Normal();
    t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

    auto t_w = getFTensor0IntegrationWeight();
    auto t_disp =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
    auto t_traction =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));
    auto t_coords = getFTensor1CoordsAtGaussPts();

    size_t nb_base_functions = data.getN().size2() / 3;
    auto t_base = data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      auto t_nf = getFTensor1FromPtr<SPACE_DIM>(nf.data());

      const double alpha = t_w * getMeasure();

      auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
      t_P(i, j) = t_contact_normal(i) * t_contact_normal(j);
      // Temporary solution to test if sliding boundary conditions works
      // t_P(i, j) = t_normal(i) * t_normal(j);

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
      t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

      FTensor::Tensor1<double, SPACE_DIM> t_rhs_constrains;

      t_rhs_constrains(i) =
          t_contact_normal(i) *
          constrian(gap0(t_coords, t_contact_normal),
                    gap(t_disp, t_contact_normal),
                    normal_traction(t_traction, t_contact_normal));

      FTensor::Tensor1<double, SPACE_DIM> t_rhs_tangent_disp,
          t_rhs_tangent_traction;
      t_rhs_tangent_disp(i) = t_Q(i, j) * t_disp(j);
      t_rhs_tangent_traction(i) = cn * t_Q(i, j) * t_traction(j);

      size_t bb = 0;
      for (; bb != nb_dofs / SPACE_DIM; ++bb) {
        const double beta = alpha * (t_base(i) * t_normal(i));

        t_nf(i) -= beta * t_rhs_constrains(i);
        t_nf(i) -= beta * t_rhs_tangent_disp(i);
        t_nf(i) += beta * t_rhs_tangent_traction(i);

        ++t_nf;
        ++t_base;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_disp;
      ++t_traction;
      ++t_coords;
      ++t_w;
    }

    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpConstrainBoundaryLhs_dU::OpConstrainBoundaryLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}
MoFEMErrorCode OpConstrainBoundaryLhs_dU::doWork(int row_side, int col_side,
                                                 EntityType row_type,
                                                 EntityType col_type,
                                                 EntData &row_data,
                                                 EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_normal = getFTensor1Normal();
    t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

    auto t_disp =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
    auto t_traction =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = t_w * getMeasure();

      auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
      t_P(i, j) = t_contact_normal(i) * t_contact_normal(j);
      // Temporary solution to test if sliding boundary conditions works
      // t_P(i, j) = t_normal(i) * t_normal(j);

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
      t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

      auto diff_constrain = diff_constrains_dgap(
          gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
          normal_traction(t_traction, t_contact_normal));

      size_t rr = 0;
      for (; rr != row_nb_dofs / SPACE_DIM; ++rr) {

        auto t_mat = getFTensor2FromArray<SPACE_DIM, SPACE_DIM, SPACE_DIM>(
            locMat, SPACE_DIM * rr);

        const double row_base = t_row_base(i) * t_normal(i);

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / SPACE_DIM; ++cc) {
          const double beta = alpha * row_base * t_col_base;

          t_mat(i, j) -= (beta * diff_constrain) * t_P(i, j);
          t_mat(i, j) -= beta * t_Q(i, j);

          ++t_col_base;
          ++t_mat;
        }

        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_disp;
      ++t_traction;
      ++t_coords;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpConstrainBoundaryLhs_dTraction::OpConstrainBoundaryLhs_dTraction(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}
MoFEMErrorCode OpConstrainBoundaryLhs_dTraction::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_normal = getFTensor1Normal();
    t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

    auto t_disp =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
    auto t_traction =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = t_w * getMeasure();

      auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
      t_P(i, j) = t_contact_normal(i) * t_contact_normal(j);
      // Temporary solution to test if sliding boundary conditions works
      // t_P(i, j) = t_normal(i) * t_normal(j);

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
      t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

      const double diff_traction = diff_constrains_dtraction(
          gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
          normal_traction(t_traction, t_contact_normal));

      size_t rr = 0;
      for (; rr != row_nb_dofs / SPACE_DIM; ++rr) {

        auto t_mat = getFTensor2FromArray<SPACE_DIM, SPACE_DIM, SPACE_DIM>(
            locMat, SPACE_DIM * rr);
        const double row_base = t_row_base(i) * t_normal(i);

        auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / SPACE_DIM; ++cc) {
          const double col_base = t_col_base(i) * t_normal(i);
          const double beta = alpha * row_base * col_base;

          t_mat(i, j) += (beta * diff_traction) * t_P(i, j);
          t_mat(i, j) += beta * cn * t_Q(i, j);

          ++t_col_base;
          ++t_mat;
        }

        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_disp;
      ++t_traction;
      ++t_coords;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

struct OpPostProcVertex : public BoundaryEleOp {
  OpPostProcVertex(MoFEM::Interface &m_field, const std::string field_name,
                  boost::shared_ptr<CommonData> common_data_ptr,
                  moab::Interface *moab_vertex);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  MoFEM::Interface &mField;
  moab::Interface *moabVertex;
  boost::shared_ptr<CommonData> commonDataPtr;
  ParallelComm *pComm;
};

OpPostProcVertex::OpPostProcVertex(
    MoFEM::Interface &m_field, const std::string field_name,
    boost::shared_ptr<CommonData> common_data_ptr, moab::Interface *moab_vertex)
    : mField(m_field), BoundaryEleOp(field_name, BoundaryEleOp::OPROW),
      commonDataPtr(common_data_ptr), moabVertex(moab_vertex) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  doEntities[boundary_ent] = true;
  pComm = ParallelComm::get_pcomm(moabVertex, MYPCOMM_INDEX);
  if (pComm == NULL)
    pComm = new ParallelComm(moabVertex, PETSC_COMM_WORLD);
}

MoFEMErrorCode OpPostProcVertex::doWork(int side, EntityType type,
                                       EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();
  std::array<double, 9> def;
  std::fill(def.begin(), def.end(), 0);

  auto get_tag = [&](const std::string name, size_t size) {
    Tag th;
    CHKERR moabVertex->tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def.data());
    return th;
  };

  auto t_coords = getFTensor1CoordsAtGaussPts();
  auto t_disp = getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));

  auto th_gap = get_tag("GAP", 1);
  auto th_cons = get_tag("CONSTRAINT", 1);
  auto th_traction = get_tag("TRACTION", 3);
  auto th_normal = get_tag("NORMAL", 3);
  auto th_cont_normal = get_tag("CONTACT_NORMAL", 3);
  auto th_disp = get_tag("DISPLACEMENT", 3);

  EntityHandle ent = getFEEntityHandle();

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    double coords[] = {0, 0, 0};
    EntityHandle new_vertex;
    for (int dd = 0; dd != 3; dd++) {
      coords[dd] = getCoordsAtGaussPts()(gg, dd);
    }
    CHKERR moabVertex->create_vertex(&coords[0], new_vertex);
    FTensor::Tensor1<double, 3> trac(t_traction(0), t_traction(1), 0.);
    FTensor::Tensor1<double, 3> disp(t_disp(0), t_disp(1), 0.);

    auto t_contact_normal = normal(t_coords, t_disp);
    const double g0 = gap0(t_coords, t_contact_normal);
    const double g = gap(t_disp, t_contact_normal);
    const double gap_tot = g - g0;
    const double constra = constrian(
        gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
        normal_traction(t_traction, t_contact_normal));
    CHKERR moabVertex->tag_set_data(th_gap, &new_vertex, 1, &gap_tot);
    CHKERR moabVertex->tag_set_data(th_cons, &new_vertex, 1, &constra);

    FTensor::Tensor1<double, 3> norm(t_contact_normal(0), t_contact_normal(1),
                                     0.);

    if (SPACE_DIM == 3) {
      trac(2) = t_traction(2);
      norm(2) = t_contact_normal(2);
      disp(2) = t_disp(2);
    }

    CHKERR moabVertex->tag_set_data(th_traction, &new_vertex, 1, &trac(0));
    CHKERR moabVertex->tag_set_data(th_cont_normal, &new_vertex, 1, &norm(0));
    CHKERR moabVertex->tag_set_data(th_disp, &new_vertex, 1, &disp(0));
    auto t_normal = getFTensor1Normal();
    CHKERR moabVertex->tag_set_data(th_normal, &new_vertex, 1, &t_normal(0));

    auto set_part = [&](const auto vert) {
      MoFEMFunctionBegin;
      const int rank = mField.get_comm_rank();
      CHKERR moabVertex->tag_set_data(pComm->part_tag(), &vert, 1, &rank);
      MoFEMFunctionReturn(0);
    };

    CHKERR set_part(new_vertex);

    ++t_traction;
    ++t_coords;
    ++t_disp;
  }

  MoFEMFunctionReturn(0);
}
template <int DIM> struct OpPostProcContact : public DomainEleOp {
  OpPostProcContact(const std::string field_name,
                    moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<CommonData> commonDataPtr;
};

//! [Operators definitions]
template <int DIM>
OpPostProcContact<DIM>::OpPostProcContact(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[boundary_ent], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
template <int DIM>
MoFEMErrorCode OpPostProcContact<DIM>::doWork(int side, EntityType type,
                                              EntData &data) {
  MoFEMFunctionBegin;

  auto get_tag_mat = [&](const std::string name) {
    std::array<double, 9> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), 9, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  auto get_tag_vec = [&](const std::string name) {
    std::array<double, 3> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), 3, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3, 3);

  auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != DIM; ++r)
      for (size_t c = 0; c != DIM; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_vector = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != DIM; ++r)
      mat(0, r) = t(r);
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_stress = get_tag_mat("SIGMA");
  auto th_div = get_tag_vec("DIV_SIGMA");

  size_t nb_gauss_pts = getGaussPts().size2();
  auto t_stress =
      getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->contactStressPtr));
  auto t_div =
      getFTensor1FromMat<DIM>(*(commonDataPtr->contactStressDivergencePtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    CHKERR set_tag(th_stress, gg, set_matrix(t_stress));
    CHKERR set_tag(th_div, gg, set_vector(t_div));
    ++t_stress;
    ++t_div;
  }

  MoFEMFunctionReturn(0);
}
//! [Postprocessing]

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm, boost::shared_ptr<CommonData> common_data_ptr,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uz_scatter)
      : dM(dm), commonDataPtr(common_data_ptr), uXScatter(ux_scatter),
        uYScatter(uy_scatter), uZScatter(uz_scatter),
        moabVertex(mbVertexPostproc) {
    MoFEM::Interface *m_field_ptr;
    CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);
    vertexPostProc = boost::make_shared<BoundaryEle>(*m_field_ptr);

    if (SPACE_DIM == 2)
      vertexPostProc->getOpPtrVector().push_back(
          new OpSetContrariantPiolaTransformOnEdge());
    vertexPostProc->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>(
            "U", commonDataPtr->contactDispPtr));
    vertexPostProc->getOpPtrVector().push_back(
        new OpCalculateHVecTensorTrace<SPACE_DIM, BoundaryEleOp>(
            "SIGMA", commonDataPtr->contactTractionPtr));
    vertexPostProc->getOpPtrVector().push_back(
        new OpPostProcVertex(*m_field_ptr, "U", commonDataPtr, &moabVertex));
    vertexPostProc->setRuleHook = [](int a, int b, int c) {
      return 2 * order + 1;
    };

    MatrixDouble inv_jac, jac;
    postProcFe = boost::make_shared<PostProcEle>(*m_field_ptr);
    postProcFe->generateReferenceElementMesh();
    if (SPACE_DIM == 2) {
      postProcFe->getOpPtrVector().push_back(new OpCalculateJacForFace(jac));
      postProcFe->getOpPtrVector().push_back(
          new OpCalculateInvJacForFace(inv_jac));
      postProcFe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(inv_jac));
      postProcFe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
      postProcFe->getOpPtrVector().push_back(
          new OpSetContravariantPiolaTransformFace(jac));
      postProcFe->getOpPtrVector().push_back(new OpSetInvJacHcurlFace(inv_jac));
    }

    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
            "U", commonDataPtr->mGradPtr));
    postProcFe->getOpPtrVector().push_back(new OpSymmetrizeTensor<SPACE_DIM>(
        "U", commonDataPtr->mGradPtr, commonDataPtr->mStrainPtr));
    postProcFe->getOpPtrVector().push_back(
        new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
            "U", commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr,
            commonDataPtr->mDPtr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateHVecTensorDivergence<SPACE_DIM, SPACE_DIM>(
            "SIGMA", commonDataPtr->contactStressDivergencePtr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateHVecTensorField<SPACE_DIM, SPACE_DIM>(
            "SIGMA", commonDataPtr->contactStressPtr));

    postProcFe->getOpPtrVector().push_back(
        new Tutorial::OpPostProcElastic<SPACE_DIM>(
            "U", postProcFe->postProcMesh, postProcFe->mapGaussPts,
            commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr));

    postProcFe->getOpPtrVector().push_back(new OpPostProcContact<SPACE_DIM>(
        "SIGMA", postProcFe->postProcMesh, postProcFe->mapGaussPts,
        commonDataPtr));
    postProcFe->addFieldValuesPostProc("U", "DISPLACEMENT");
  }

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    auto post_proc_volume = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe);
      CHKERR postProcFe->writeFile(
          "out_contact_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      MoFEMFunctionReturn(0);
    };

    auto post_proc_boundary = [&] {
      MoFEMFunctionBegin;
      std::ostringstream ostrm;
      ostrm << "out_debug_" << ts_step << ".h5m";
      CHKERR DMoFEMLoopFiniteElements(dM, "bFE", vertexPostProc);
      CHKERR moabVertex.write_file(ostrm.str().c_str(), "MOAB",
                                   "PARALLEL=WRITE_PART");
      moabVertex.delete_mesh();
      MoFEMFunctionReturn(0);
    };

    auto print_max_min = [&](auto &tuple, const std::string msg) {
      MoFEMFunctionBegin;
      CHKERR VecScatterBegin(std::get<1>(tuple), ts_u, std::get<0>(tuple),
                             INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecScatterEnd(std::get<1>(tuple), ts_u, std::get<0>(tuple),
                           INSERT_VALUES, SCATTER_FORWARD);
      double max, min;
      CHKERR VecMax(std::get<0>(tuple), PETSC_NULL, &max);
      CHKERR VecMin(std::get<0>(tuple), PETSC_NULL, &min);
      MOFEM_LOG_C("EXAMPLE", Sev::inform, "%s time %3.4e min %3.4e max %3.4e",
                  msg.c_str(), ts_t, min, max);
      MoFEMFunctionReturn(0);
    };

    CHKERR post_proc_volume();
    CHKERR post_proc_boundary();
    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");
    if (SPACE_DIM == 3)
      CHKERR print_max_min(uZScatter, "Uz");

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<CommonData> commonDataPtr;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<BoundaryEle> vertexPostProc;
  moab::Core mbVertexPostproc;
  moab::Interface &moabVertex;

};




}; // namespace OpContactTools
