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

/** \file PlasticThermalOps.hpp
 * \example PlasticThermalOps.hpp
 */

namespace SeepageOps {

template <int DIM>
struct OpDomainRhsHydrostaticStress
    : public AssemblyDomainEleOp { // changed opfaceele to AssemblyDomainEleOp
public:
  OpDomainRhsHydrostaticStress(std::string field_name1,
  boost::shared_ptr<VectorDouble> h_ptr,
                               double specific_weight_water = 9.81)
      : AssemblyDomainEleOp(field_name1, field_name1, DomainEleOp::OPROW), hPtr(h_ptr),
        specificWeightWater(specific_weight_water) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const int nb_dofs = data.getIndices().size();

    if (nb_dofs) {
      // locRhs.resize(nb_dofs, false);
      // locRhs.clear();
      auto &nf = AssemblyDomainEleOp::locF;
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

      auto t_h = getFTensor0FromVec(*hPtr);
      for (int gg = 0; gg != nb_integration_points; gg++) {
        auto t_nf = getFTensor1FromPtr<DIM>(&nf[0]);
        // auto t_rhs = getFTensor1FromArray<DIM, DIM>(locRhs);
        
        const double a = t_w * area * specificWeightWater * t_h;
        
        for (int rr = 0; rr != nb_dofs / DIM; rr++) {
          // each degree of freedom gives a number of shape functions, for two
          // degrees of freedom there is one shape function, so that is why it
          // is devided by two. (this should be DIM)
          // locRhs[rr] += t_base * body_source * a;

          t_nf(i) -= t_base_diff(i) * a;

          // move to the next base function
          ++t_base_diff; // moves the pointer to the next shape function
          // ++t_rhs;
          ++t_nf;
        }

        // move to the weight of the next integration point
        ++t_w;
        ++t_h;
      }

      // FILL VALUES OF LOCAL VECTOR ENTRIES TO THE GLOBAL VECTOR

      // // Ignoring DOFs on boundary (index -1)
      // CHKERR VecSetOption(getKSPf(), VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
      // CHKERR VecSetValues(getKSPf(), data, &locRhs(0), ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }
  private:
  // VectorDouble locRhs;
  double specificWeightWater;
  boost::shared_ptr<VectorDouble> hPtr;
};

} // namespace PlasticThermalOps