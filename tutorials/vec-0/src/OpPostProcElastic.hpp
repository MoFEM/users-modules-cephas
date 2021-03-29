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
                    boost::shared_ptr<MatrixDouble> m_stress_ptr,
                    double water_table = 0.,
                    double specific_weight_water = 9.81);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<MatrixDouble> mStrainPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;
  double waterTable;
  double specificWeightWater;
  
};
//! [Class definition]

//! [Postprocessing constructor]
template <int DIM>
OpPostProcElastic<DIM>::OpPostProcElastic(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<MatrixDouble> m_strain_ptr,
    boost::shared_ptr<MatrixDouble> m_stress_ptr, double water_table,
    double specific_weight_water)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), mStrainPtr(m_strain_ptr),
      mStressPtr(m_stress_ptr), waterTable(water_table),
      specificWeightWater(specific_weight_water) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}
//! [Postprocessing constructor]

//! [Postprocessing]
template <int DIM>
MoFEMErrorCode OpPostProcElastic<DIM>::doWork(int side, EntityType type,
                                              EntData &data) {
  MoFEMFunctionBegin;

  auto get_tag = [&](const std::string name, int size = 9) {
    std::array<double, 9> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
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


  auto set_tag_scalar = [&](auto th, auto gg,
                            double scalar) { // not intirely sure what these do
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1, &scalar);
  };

  auto mcc_yield_check = [&](const double hyd_p, const double dev_q,
                             const double crit_m, const double p_nought) {
    return ((pow(dev_q, 2) / pow(crit_m, 2)) + hyd_p * (hyd_p - p_nought));
  };

  auto dev_strain_calc = [&](auto th, auto gg, double dev_q) {
    return (dev_q * 2);
  };

  auto vol_strain_calc = [&](const double crit_m, const double p_nought,
                             const double hyd_p) {
    return (pow(crit_m, 2) * (p_nought - 2 * hyd_p));
  };

  auto th_strain = get_tag("STRAIN");
  auto th_stress = get_tag("EFFECTIVE_STRESS");
  auto th_hydrostatic = get_tag("HYDROSTATIC_P", 1);
  auto th_deviatoric = get_tag("DEVIATORIC_Q", 1);
  auto th_yield = get_tag("YIELD", 1);
  auto th_deviatoric_strain = get_tag("DEVIATORIC_STRAIN", 1);
  auto th_volumetric_strain = get_tag("VOLUMETRIC_STRAIN", 1);
  auto th_strain_direction = get_tag("STRAIN_DIRECTION", 1);
  auto th_total_stress = get_tag("TOTAL_STRESS");
  auto th_pore_water_pressure = get_tag("PORE_WATER_PRESSURE", 1);
  double hydrostatic_p = 0.;
  double deviatoric_q = 0.;
  double deviatoric_strain = 0.;
  double volumetric_strain = 0.;
  double plastic_strain_direction = 0.;
  FTensor::Tensor2_symmetric<double, DIM> t_deviatoric;
  FTensor::Tensor2_symmetric<double, DIM> total_stress;

  constexpr double criticalstategradient_m = 1.;
  constexpr double preconsolidation_pressure = 1000.;

  size_t nb_gauss_pts = data.getN().size1();
  auto t_strain = getFTensor2SymmetricFromMat<DIM>(*(mStrainPtr));
  auto t_stress = getFTensor2SymmetricFromMat<DIM>(*(mStressPtr));
  FTensor::Index<'i', DIM> i; 
                              
  FTensor::Index<'j', DIM>
      j; 

  switch (DIM) {
  case 2:
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      CHKERR set_tag(
          th_strain, gg,
          set_plain_stress_strain(set_matrix_symm(t_strain), t_stress));
      CHKERR set_tag(th_stress, gg, set_matrix_symm(t_stress));
      hydrostatic_p = t_stress(i, i);
      hydrostatic_p /= 3.;
      CHKERR set_tag_scalar(th_hydrostatic, gg,
                            hydrostatic_p); // not certain about this?
      t_deviatoric(i, j) =
          t_stress(i, j) -
          hydrostatic_p * FTensor::kronecker_delta_symmetric(
                              i, j); // Am I using the right stress?
      
      double deviatoric_q =
          sqrt(t_deviatoric(i, j) * t_deviatoric(i, j) * 3. / 2.);
      double deviatoric_strain_d = 2 * deviatoric_q;
      CHKERR set_tag_scalar(th_deviatoric_strain, gg, deviatoric_strain_d);
      
      double volumetric_strain_v =
          pow(criticalstategradient_m, 2) *
          (preconsolidation_pressure - 2 * hydrostatic_p);
      CHKERR set_tag_scalar(th_volumetric_strain, gg,
                            vol_strain_calc(criticalstategradient_m,
                                            preconsolidation_pressure,
                                            hydrostatic_p));
      
      double function_value =
          mcc_yield_check(hydrostatic_p, deviatoric_q, criticalstategradient_m,
                          preconsolidation_pressure);
      CHKERR set_tag_scalar(th_yield, gg,
                            function_value);

      double pore_water_pressure= 0.;
       double water_depth = getCoordsAtGaussPts()(gg, 1)- waterTable;
      
        if (water_depth < 0.) {

         pore_water_pressure = specificWeightWater * water_depth;
         CHKERR set_tag_scalar(th_pore_water_pressure, gg, pore_water_pressure);
       }

        total_stress(i, j) =
            t_stress(i, j) +
            pore_water_pressure * FTensor::kronecker_delta_symmetric(i, j);
        CHKERR set_tag(th_total_stress, gg, set_matrix_symm(total_stress));
      
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
//Constructor for water pressure
template <int DIM>
struct OpDomainRhsHydrostaticStress
    : public DomainEleOp { // changed opfaceele to DomainEleOp
public:
  OpDomainRhsHydrostaticStress(std::string field_name1,
                               double specific_weight_water = 9.81,
                               double water_table = 0.)
      : DomainEleOp(field_name1, DomainEleOp::OPROW), specificWeightWater(specific_weight_water),
        waterTable(water_table) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    const int nb_dofs = data.getIndices().size();

    if (nb_dofs) {
      locRhs.resize(nb_dofs, false);
      locRhs.clear();

      // get element area
      const double area = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();

      // get base function
      auto t_base_diff = data.getFTensor1DiffN<DIM>();

      constexpr double g_acceleration = 9.81;

      FTensor::Index<'i', DIM> i;
      // changed getting the base of the function to getting the
      // differential of the base it gives a vector to a pointer of two

      double x, y, z, water_depth ;
      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      
      for (int gg = 0; gg != nb_integration_points; gg++) {
        auto t_rhs = getFTensor1FromArray<DIM, DIM>(locRhs);
        if (DIM == 3)
          water_depth = getCoordsAtGaussPts()(gg, 2) - waterTable;
        else if (DIM == 2)
          water_depth = getCoordsAtGaussPts()(gg, 1) - waterTable;

            if (water_depth > 0.)
            continue;
          
            const double a =
            t_w * area * specificWeightWater * water_depth;
        for (int rr = 0; rr != nb_dofs / DIM; rr++) {
          // each degree of freedom gives a number of shape functions, for two
          // degrees of freedom there is one shape function, so that is why it
          // is devided by two. (this should be DIM)
          // locRhs[rr] += t_base * body_source * a;

          t_rhs(i) -= t_base_diff(i) * a;

          // move to the next base function
          ++t_base_diff; // moves the pointer to the next shape function
          ++t_rhs;
        }

        // move to the weight of the next integration point
        ++t_w;
      }

      // FILL VALUES OF LOCAL VECTOR ENTRIES TO THE GLOBAL VECTOR

      // // Ignoring DOFs on boundary (index -1)
       CHKERR VecSetOption(getKSPf(), VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
       CHKERR VecSetValues(getKSPf(), data, &locRhs(0), ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }

private:
  VectorDouble locRhs;
  double specificWeightWater;
  double waterTable;
};

//CREATING TEMPLATE TO PRODUCE AN OPROWCOL STRUCT WORKING ON THE DOMAIN FOR L
template <int DIM>

struct OpCouplingOp : public DomainEleOp {
  OpCouplingOp(std::string field_name1, string field_name2)
      : DomainEleOp(field_name1, field_name2, DomainEleOp::OPROWCOL) {
    sYmm = false;
  }


    MoFEMErrorCode doWork(
            int row_side, int col_side, EntityType row_type,
            EntityType col_type, EntData &row_data, EntData &col_data) {
      MoFEMFunctionBegin;
      // Both sides are needed since both sides contribute their shape
      // function to the stiffness matrix
      const int nb_row = row_data.getIndices().size();
      const int nb_col = col_data.getIndices().size();
       if (nb_row && nb_col) {
         const int nb_gauss_pts = row_data.getN().size1();
         int nb_base_fun_row = row_data.getFieldData().size() / 3;
         int nb_base_fun_col = col_data.getFieldData().size();
        auto get_tensor_from_mat = [](MatrixDouble &m, const int r,
                                      const int c) {
          return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
                                               &m(r + 2, c));
        };
      //   auto get_tensor_vec = [](VectorDouble &n) {
      //     return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
      //   };
        FTensor::Index<'i', 3> i;
        NN.resize(nb_row, nb_col, false);
        NN.clear();
       auto t_w = getFTensor0IntegrationWeight();
       const double area = getMeasure();
       auto t_base_diff_row = row_data.getFTensor1DiffN<DIM>();
       for (int gg = 0; gg != nb_gauss_pts; ++gg) {
         for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
           auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 0);
           auto t_base_col = col_data.getFTensor0N(gg, 0);
           for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
//const double n = m * t_base_col;
             //t_assemble_m(i) += n * const_unit_n(i);
             ++t_assemble_m;
             ++t_base_col; // update cols slave
           }
           ++t_base_diff_row; // update rows master
         }
         ++t_w;
       }
       //   CHKERR MatSetValues(getSNESB(), row_data, col_data,
       //   &*NN.data().begin(),
       //                       ADD_VALUES);
       }
      MoFEMFunctionReturn(0);
    }

  private:
    MatrixDouble NN;
  };
} // namespace Tutorial