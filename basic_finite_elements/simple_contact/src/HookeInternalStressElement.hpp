/**
 * \file HookeInternalStressElement.hpp
 * \example HookeInternalStressElement.hpp
 *
 * \brief Operators and data structures for calculating internal stress
 *
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

#ifndef __HOOKE_INTERNAL_STRESS_ELEMENT_HPP__
#define __HOOKE_INTERNAL_STRESS_ELEMENT_HPP__

#include <BasicFiniteElements.hpp>

struct HookeInternalStressElement : public HookeElement {

  struct DataAtIntegrationPts : public HookeElement::DataAtIntegrationPts {

    boost::shared_ptr<MatrixDouble> internalStressMat;
    boost::shared_ptr<MatrixDouble> actualStressMat;
    boost::shared_ptr<MatrixDouble> deviatoricStressMat;
    boost::shared_ptr<MatrixDouble> hydrostaticStressMat;

    boost::shared_ptr<MatrixDouble> spatPosMat;
    boost::shared_ptr<MatrixDouble> meshNodePosMat;

    DataAtIntegrationPts() {

      internalStressMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      actualStressMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      deviatoricStressMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      hydrostaticStressMat =
          boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      spatPosMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      meshNodePosMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    }
  };

  struct OpGetInternalStress : public VolUserDataOperator {

    OpGetInternalStress(const std::string row_field,
                        const std::string col_field,
                        boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
                        moab::Interface &input_mesh, char *stress_tag_name)
        : VolUserDataOperator(row_field, col_field, OPROW, true),
          dataAtPts(data_at_pts), inputMesh(input_mesh),
          stressTagName(stress_tag_name) {
      std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
    };

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  protected:
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
    moab::Interface &inputMesh;
    char *stressTagName;
  };

  struct OpInternalStrain_dx : public OpAssemble {

    OpInternalStrain_dx(const std::string row_field,
                        boost::shared_ptr<DataAtIntegrationPts> data_at_pts)
        : OpAssemble(row_field, row_field, data_at_pts, OPROW, true),
          dataAtPts(data_at_pts){};

    MoFEMErrorCode iNtegrate(EntData &row_data);

  protected:
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
  };

  template <int S = 0>
  struct OpGetAnalyticalInternalStress : public VolUserDataOperator {

    typedef boost::function<

        FTensor::Tensor2_symmetric<double, 3>(

            FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> &t_coords

            )

        >
        StrainFunction;

    OpGetAnalyticalInternalStress(
        const std::string row_field, const std::string col_field,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
        StrainFunction strain_fun)
        : VolUserDataOperator(row_field, col_field, OPROW, true),
          dataAtPts(data_at_pts), strainFun(strain_fun) {
      std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
    }

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  protected:
    StrainFunction strainFun;
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
  };

  struct OpSaveStress : public VolUserDataOperator {
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
    map<int, BlockData>
        &blockSetsPtr; // FIXME: (works only with the first block)
    moab::Interface &outputMesh;
    bool isALE;
    bool isFieldDisp;
    double scaleFactor;
    bool saveMean;

    OpSaveStress(const string row_field, const string col_field,
                 boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
                 map<int, BlockData> &block_sets_ptr,
                 moab::Interface &output_mesh, double scale_factor,
                 bool save_mean = false, bool is_ale = false,
                 bool is_field_disp = true)
        : VolUserDataOperator(row_field, col_field, OPROW, true),
          dataAtPts(data_at_pts), blockSetsPtr(block_sets_ptr),
          outputMesh(output_mesh), scaleFactor(scale_factor), isALE(is_ale),
          isFieldDisp(is_field_disp), saveMean(save_mean){};

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);
  };

  template <class ELEMENT>
  struct OpPostProcHookeElement : public ELEMENT::UserDataOperator {
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
    map<int, BlockData>
        &blockSetsPtr; // FIXME: (works only with the first block)
    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;
    bool isALE;
    bool isFieldDisp;

    OpPostProcHookeElement(const string row_field,
                           boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
                           map<int, BlockData> &block_sets_ptr,
                           moab::Interface &post_proc_mesh,
                           std::vector<EntityHandle> &map_gauss_pts,
                           bool is_ale = false, bool is_field_disp = true)
        : ELEMENT::UserDataOperator(row_field, UserDataOperator::OPROW),
          dataAtPts(data_at_pts), blockSetsPtr(block_sets_ptr),
          postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts),
          isALE(is_ale), isFieldDisp(is_field_disp) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };
};

MoFEMErrorCode HookeInternalStressElement::OpGetInternalStress::doWork(
    int row_side, EntityType row_type, EntData &row_data) {
  MoFEMFunctionBegin;

  const int nb_integration_pts = row_data.getN().size1();

  const EntityHandle ent = getFEEntityHandle();

  const int val_num = 9 * nb_integration_pts;
  std::vector<double> def_vals(val_num, 0.0);
  Tag th_internal_stress;
  CHKERR inputMesh.tag_get_handle(
      stressTagName, val_num, MB_TYPE_DOUBLE, th_internal_stress,
      MB_TAG_CREAT | MB_TAG_SPARSE, &*def_vals.begin());

  dataAtPts->internalStressMat->resize(9, nb_integration_pts, false);

  CHKERR inputMesh.tag_get_data(
      th_internal_stress, &ent, 1,
      &*(dataAtPts->internalStressMat->data().begin()));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
HookeInternalStressElement::OpInternalStrain_dx::iNtegrate(EntData &row_data) {
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;
  MoFEMFunctionBegin;

  auto get_tensor1 = [](VectorDouble &v, const int r) {
    return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>(
        &v(r + 0), &v(r + 1), &v(r + 2));
  };

  const int nb_integration_pts = getGaussPts().size2();

  auto t_internal_stress =
      getFTensor2FromMat<3, 3>(*dataAtPts->internalStressMat);

  // get element volume
  double vol = getVolume();
  auto t_w = getFTensor0IntegrationWeight();

  nF.resize(nbRows, false);
  nF.clear();

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

    // calculate scalar weight times element volume
    double a = t_w * vol;
    auto t_nf = get_tensor1(nF, 0);

    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {
      t_nf(i) += a * t_row_diff_base(j) * t_internal_stress(i, j);
      ++t_row_diff_base;
      ++t_nf;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    ++t_w;
    ++t_internal_stress;
  }

  MoFEMFunctionReturn(0);
}

template <int S>
MoFEMErrorCode
HookeInternalStressElement::OpGetAnalyticalInternalStress<S>::doWork(
    int row_side, EntityType row_type, EntData &row_data) {
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;
  MoFEMFunctionBegin;

  auto tensor_to_tensor = [](const auto &t1, auto &t2) {
    t2(0, 0) = t1(0, 0);
    t2(1, 1) = t1(1, 1);
    t2(2, 2) = t1(2, 2);
    t2(0, 1) = t2(1, 0) = t1(1, 0);
    t2(0, 2) = t2(2, 0) = t1(2, 0);
    t2(1, 2) = t2(2, 1) = t1(2, 1);
  };

  const int nb_integration_pts = getGaussPts().size2();
  auto t_coords = getFTensor1CoordsAtGaussPts();

  // elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));

  dataAtPts->internalStressMat->resize(9, nb_integration_pts, false);
  auto t_internal_stress =
      getFTensor2FromMat<3, 3>(*dataAtPts->internalStressMat);

  FTensor::Tensor2_symmetric<double, 3> t_internal_stress_symm;

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

    auto t_fun_strain = strainFun(t_coords);
    t_internal_stress_symm(i, j) = -t_D(i, j, k, l) * t_fun_strain(k, l);

    tensor_to_tensor(t_internal_stress_symm, t_internal_stress);

    ++t_coords;
    ++t_D;
    ++t_internal_stress;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HookeInternalStressElement::OpSaveStress::doWork(
    int row_side, EntityType row_type, EntData &row_data) {
  MoFEMFunctionBegin;

  if (row_type != MBVERTEX) {
    MoFEMFunctionReturnHot(0);
  }

  const EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();

  auto tensor_to_tensor = [](const auto &t1, auto &t2) {
    t2(0, 0) = t1(0, 0);
    t2(1, 1) = t1(1, 1);
    t2(2, 2) = t1(2, 2);
    t2(0, 1) = t2(1, 0) = t1(1, 0);
    t2(0, 2) = t2(2, 0) = t1(2, 0);
    t2(1, 2) = t2(2, 1) = t1(2, 1);
  };

  auto tensor_to_vector = [](const auto &t, auto &v) {
    v(0) = t(0, 0);
    v(1) = t(1, 1);
    v(2) = t(2, 2);
    v(3) = t(0, 1);
    v(4) = t(0, 2);
    v(5) = t(1, 2);
  };

  auto get_tag_handle = [&](auto name, auto size) {
    Tag th;
    std::vector<double> def_vals(size, 0.0);
    CHKERR outputMesh.tag_get_handle(name, size, MB_TYPE_DOUBLE, th,
                                     MB_TAG_CREAT | MB_TAG_SPARSE,
                                     def_vals.data());
    return th;
  };

  const int nb_integration_pts = row_data.getN().size1();

  auto th_internal_stress =
      get_tag_handle("INTERNAL_STRESS", 9 * nb_integration_pts);
  auto th_actual_stress =
      get_tag_handle("ACTUAL_STRESS", 9 * nb_integration_pts);
  auto th_deviatoric_stress =
      get_tag_handle("DEVIATORIC_STRESS", 9 * nb_integration_pts);
  auto th_hydrostatic_stress =
      get_tag_handle("HYDROSTATIC_STRESS", 9 * nb_integration_pts);

  auto th_internal_stress_mean = get_tag_handle("MED_INTERNAL_STRESS", 9);
  auto th_actual_stress_mean = get_tag_handle("MED_ACTUAL_STRESS", 9);

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  auto t_h = getFTensor2FromMat<3, 3>(*dataAtPts->hMat);
  auto t_H = getFTensor2FromMat<3, 3>(*dataAtPts->HMat);

  dataAtPts->stiffnessMat->resize(36, 1, false);
  FTensor::Ddg<FTensor::PackPtr<double *, 1>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));
  for (auto &m : (blockSetsPtr)) {
    const double young = m.second.E;
    const double poisson = m.second.PoissonRatio;

    const double coefficient = young / ((1 + poisson) * (1 - 2 * poisson));

    t_D(i, j, k, l) = 0.;
    t_D(0, 0, 0, 0) = t_D(1, 1, 1, 1) = t_D(2, 2, 2, 2) = 1 - poisson;
    t_D(0, 1, 0, 1) = t_D(0, 2, 0, 2) = t_D(1, 2, 1, 2) =
        0.5 * (1 - 2 * poisson);
    t_D(0, 0, 1, 1) = t_D(1, 1, 0, 0) = t_D(0, 0, 2, 2) = t_D(2, 2, 0, 0) =
        t_D(1, 1, 2, 2) = t_D(2, 2, 1, 1) = poisson;
    t_D(i, j, k, l) *= coefficient;

    break; // FIXME: calculates only first block
  }

  double detH = 0.;
  FTensor::Tensor2<double, 3, 3> t_invH;
  FTensor::Tensor2<double, 3, 3> t_F;

  FTensor::Tensor2_symmetric<double, 3> t_stress_symm;
  FTensor::Tensor2_symmetric<double, 3> t_small_strain_symm;

  FTensor::Tensor2<double, 3, 3> t_stress;

  auto t_internal_stress =
      getFTensor2FromMat<3, 3>(*dataAtPts->internalStressMat);

  dataAtPts->actualStressMat->resize(9, nb_integration_pts, false);
  auto t_actual_stress = getFTensor2FromMat<3, 3>(*dataAtPts->actualStressMat);

  dataAtPts->deviatoricStressMat->resize(9, nb_integration_pts, false);
  auto t_deviatoric_stress =
      getFTensor2FromMat<3, 3>(*dataAtPts->deviatoricStressMat);
  dataAtPts->hydrostaticStressMat->resize(9, nb_integration_pts, false);
  auto t_hydrostatic_stress =
      getFTensor2FromMat<3, 3>(*dataAtPts->hydrostaticStressMat);

  FTensor::Tensor2<double, 3, 3> t_internal_stress_mean;
  FTensor::Tensor2<double, 3, 3> t_actual_stress_mean;

  t_internal_stress_mean(i, j) = 0.;
  t_actual_stress_mean(i, j) = 0.;

  auto t_w = getFTensor0IntegrationWeight();

  for (int gg = 0; gg != nb_integration_pts; ++gg) {

    if (!isALE) {
      t_small_strain_symm(i, j) = (t_h(i, j) || t_h(j, i)) / 2.;
    } else {
      CHKERR determinantTensor3by3(t_H, detH);
      CHKERR invertTensor3by3(t_H, detH, t_invH);
      t_F(i, j) = t_h(i, k) * t_invH(k, j);
      t_small_strain_symm(i, j) = (t_F(i, j) || t_F(j, i)) / 2.;
      ++t_H;
    }

    if (isALE || !isFieldDisp) {
      for (auto ii : {0, 1, 2}) {
        t_small_strain_symm(ii, ii) -= 1.;
      }
    }

    // symmetric tensors need improvement
    t_stress_symm(i, j) = t_D(i, j, k, l) * t_small_strain_symm(k, l);
    tensor_to_tensor(t_stress_symm, t_stress);
    t_actual_stress(i, j) = t_stress(i, j) + t_internal_stress(i, j);

    t_actual_stress(i, j) *= scaleFactor;
    t_internal_stress(i, j) *= scaleFactor;

    double hydrostatic_pressure =
        (t_actual_stress(0, 0) + t_actual_stress(1, 1) +
         t_actual_stress(2, 2)) /
        3.;

    t_hydrostatic_stress(i, j) = 0.;
    for (auto ii : {0, 1, 2}) {
      t_hydrostatic_stress(ii, ii) += hydrostatic_pressure;
    }

    t_deviatoric_stress(i, j) = t_actual_stress(i, j);
    for (auto ii : {0, 1, 2}) {
      t_deviatoric_stress(ii, ii) -= hydrostatic_pressure;
    }

    t_actual_stress_mean(i, j) += t_w * t_actual_stress(i, j);
    t_internal_stress_mean(i, j) += t_w * t_internal_stress(i, j);

    ++t_w;
    ++t_h;
    ++t_actual_stress;
    ++t_internal_stress;
  }

  if (saveMean) {
    VectorDouble vec_actual_stress_mean;
    vec_actual_stress_mean.resize(9, false);
    vec_actual_stress_mean.clear();

    VectorDouble vec_internal_stress_mean;
    vec_internal_stress_mean.resize(9, false);
    vec_internal_stress_mean.clear();

    tensor_to_vector(t_actual_stress_mean, vec_actual_stress_mean);
    tensor_to_vector(t_internal_stress_mean, vec_internal_stress_mean);

    CHKERR outputMesh.tag_set_data(th_actual_stress_mean, &ent, 1,
                                   &*(vec_actual_stress_mean.data().begin()));
    CHKERR outputMesh.tag_set_data(th_internal_stress_mean, &ent, 1,
                                   &*(vec_internal_stress_mean.data().begin()));
  } else {
    CHKERR outputMesh.tag_set_data(
        th_internal_stress, &ent, 1,
        &*(dataAtPts->internalStressMat->data().begin()));
    CHKERR outputMesh.tag_set_data(
        th_actual_stress, &ent, 1,
        &*(dataAtPts->actualStressMat->data().begin()));

    CHKERR outputMesh.tag_set_data(
        th_hydrostatic_stress, &ent, 1,
        &*(dataAtPts->hydrostaticStressMat->data().begin()));
    CHKERR outputMesh.tag_set_data(
        th_deviatoric_stress, &ent, 1,
        &*(dataAtPts->deviatoricStressMat->data().begin()));
  }

  MoFEMFunctionReturn(0);
}

template <class ELEMENT>
MoFEMErrorCode
HookeInternalStressElement::OpPostProcHookeElement<ELEMENT>::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX) {
    MoFEMFunctionReturnHot(0);
  }

  auto tensor_to_tensor = [](const auto &t1, auto &t2) {
    t2(0, 0) = t1(0, 0);
    t2(1, 1) = t1(1, 1);
    t2(2, 2) = t1(2, 2);
    t2(0, 1) = t2(1, 0) = t1(1, 0);
    t2(0, 2) = t2(2, 0) = t1(2, 0);
    t2(1, 2) = t2(2, 1) = t1(2, 1);
  };

  auto get_tag_handle = [&](auto name, auto size) {
    Tag th;
    std::vector<double> def_vals(size, 0.0);
    CHKERR postProcMesh.tag_get_handle(name, size, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def_vals.data());
    return th;
  };

  auto th_stress = get_tag_handle("STRESS", 9);
  auto th_strain = get_tag_handle("STRAIN", 9);
  auto th_psi = get_tag_handle("ENERGY", 1);
  auto th_disp = get_tag_handle("DISPLACEMENT", 3);

  const int nb_integration_pts = mapGaussPts.size();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  auto t_h = getFTensor2FromMat<3, 3>(*dataAtPts->hMat);
  auto t_H = getFTensor2FromMat<3, 3>(*dataAtPts->HMat);

  dataAtPts->stiffnessMat->resize(36, 1, false);
  FTensor::Ddg<FTensor::PackPtr<double *, 1>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));
  for (auto &m : (blockSetsPtr)) {
    const double young = m.second.E;
    const double poisson = m.second.PoissonRatio;

    const double coefficient = young / ((1 + poisson) * (1 - 2 * poisson));

    t_D(i, j, k, l) = 0.;
    t_D(0, 0, 0, 0) = t_D(1, 1, 1, 1) = t_D(2, 2, 2, 2) = 1 - poisson;
    t_D(0, 1, 0, 1) = t_D(0, 2, 0, 2) = t_D(1, 2, 1, 2) =
        0.5 * (1 - 2 * poisson);
    t_D(0, 0, 1, 1) = t_D(1, 1, 0, 0) = t_D(0, 0, 2, 2) = t_D(2, 2, 0, 0) =
        t_D(1, 1, 2, 2) = t_D(2, 2, 1, 1) = poisson;
    t_D(i, j, k, l) *= coefficient;

    break; // FIXME: calculates only first block
  }

  double detH = 0.;
  FTensor::Tensor2<double, 3, 3> t_invH;
  FTensor::Tensor2<double, 3, 3> t_F;
  FTensor::Tensor2<double, 3, 3> t_stress;
  FTensor::Tensor2<double, 3, 3> t_small_strain;

  FTensor::Tensor2_symmetric<double, 3> t_stress_symm;
  FTensor::Tensor2_symmetric<double, 3> t_small_strain_symm;

  auto t_spat_pos = getFTensor1FromMat<3>(*dataAtPts->spatPosMat);
  auto t_mesh_node_pos = getFTensor1FromMat<3>(*dataAtPts->meshNodePosMat);
  FTensor::Tensor1<double, 3> t_disp;

  for (int gg = 0; gg != nb_integration_pts; ++gg) {

    if (isFieldDisp) {
      t_h(0, 0) += 1;
      t_h(1, 1) += 1;
      t_h(2, 2) += 1;
    }

    if (!isALE) {
      t_small_strain_symm(i, j) = (t_h(i, j) || t_h(j, i)) / 2.;
    } else {
      CHKERR determinantTensor3by3(t_H, detH);
      CHKERR invertTensor3by3(t_H, detH, t_invH);
      t_F(i, j) = t_h(i, k) * t_invH(k, j);
      t_small_strain_symm(i, j) = (t_F(i, j) || t_F(j, i)) / 2.;
      ++t_H;
    }

    t_small_strain_symm(0, 0) -= 1;
    t_small_strain_symm(1, 1) -= 1;
    t_small_strain_symm(2, 2) -= 1;

    // symmetric tensors need improvement
    t_stress_symm(i, j) = t_D(i, j, k, l) * t_small_strain_symm(k, l);
    tensor_to_tensor(t_stress_symm, t_stress);
    tensor_to_tensor(t_small_strain_symm, t_small_strain);

    t_disp(i) = t_spat_pos(i) - t_mesh_node_pos(i);

    const double psi = 0.5 * t_stress_symm(i, j) * t_small_strain_symm(i, j);

    CHKERR postProcMesh.tag_set_data(th_psi, &mapGaussPts[gg], 1, &psi);
    CHKERR postProcMesh.tag_set_data(th_stress, &mapGaussPts[gg], 1,
                                     &t_stress(0, 0));
    CHKERR postProcMesh.tag_set_data(th_strain, &mapGaussPts[gg], 1,
                                     &t_small_strain(0, 0));
    CHKERR postProcMesh.tag_set_data(th_disp, &mapGaussPts[gg], 1, &t_disp(0));

    ++t_h;
    ++t_spat_pos;
    ++t_mesh_node_pos;
  }

  MoFEMFunctionReturn(0);
}

#endif
