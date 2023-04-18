/**
 * \file poisson_2d_homogeneous.hpp
 * \example poisson_2d_homogeneous.hpp
 *
 * Solution of poisson equation. Direct implementation of User Data Operators
 * for teaching proposes.
 *
 * \note In practical application we suggest use form integrators to generalise
 * and simplify code. However, here we like to expose user to ways how to
 * implement data operator from scratch.
 */

// Define name if it has not been defined yet
#ifndef __POISSON_2D_HOMOGENEOUS_HPP__
#define __POISSON_2D_HOMOGENEOUS_HPP__

// Use of alias for some specific functions
// We are solving Poisson's equation in 2D so Face element is used
using EntData = EntitiesFieldData::EntData;

template <int DIM>
using ElementsAndOps = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomianParentEle = ElementsAndOps<SPACE_DIM>::DomianParentEle;
using DomainEleOp = DomainEle::UserDataOperator;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

// Namespace that contains necessary UDOs, will be included in the main program
namespace Poisson2DHomogeneousOperators {

// Declare FTensor index for 2D problem
FTensor::Index<'i', 2> i;

// For simplicity, source term f will be constant throughout the domain
const double body_source = 5.;

struct OpDomainLhsMatrixK : public AssemblyDomainEleOp {
public:
  OpDomainLhsMatrixK(std::string row_field_name, std::string col_field_name)
      : AssemblyDomainEleOp(row_field_name, col_field_name,
                            DomainEleOp::OPROWCOL) {}

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

    this->locMat.resize(nb_row_dofs, nb_col_dofs, false);
    this->locMat.clear();

    // get element area
    const double area = getMeasure();

    // get number of integration points
    const int nb_integration_points = getGaussPts().size2();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();

    // get derivatives of base functions on row
    auto t_row_diff_base = row_data.getFTensor1DiffN<2>();

    // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
    for (int gg = 0; gg != nb_integration_points; gg++) {
      const double a = t_w * area;

      for (int rr = 0; rr != nb_row_dofs; ++rr) {
        // get derivatives of base functions on column
        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

        for (int cc = 0; cc != nb_col_dofs; cc++) {
          this->locMat(rr, cc) += t_row_diff_base(i) * t_col_diff_base(i) * a;

          // move to the derivatives of the next base functions on column
          ++t_col_diff_base;
        }

        // move to the derivatives of the next base functions on row
        ++t_row_diff_base;
      }

      // move to the weight of the next integration point
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }
};

struct OpDomainRhsVectorF : public AssemblyDomainEleOp {
public:
  OpDomainRhsVectorF(std::string field_name)
      : AssemblyDomainEleOp(field_name, field_name, DomainEleOp::OPROW) {}

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;

    const int nb_dofs = data.getIndices().size();

    this->locF.resize(nb_dofs, false);
    this->locF.clear();

    // get element area
    const double area = getMeasure();

    // get number of integration points
    const int nb_integration_points = getGaussPts().size2();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();

    // get base function
    auto t_base = data.getFTensor0N();

    // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
    for (int gg = 0; gg != nb_integration_points; gg++) {
      const double a = t_w * area;

      for (int rr = 0; rr != nb_dofs; rr++) {
        this->locF[rr] += t_base * body_source * a;

        // move to the next base function
        ++t_base;
      }

      // move to the weight of the next integration point
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

};

}; // namespace Poisson2DHomogeneousOperators

#endif //__POISSON_2D_HOMOGENEOUS_HPP__