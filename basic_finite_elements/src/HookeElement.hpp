/**
 * \file HookeElement.hpp
 * \example HookeElement.hpp
 *
 * \brief Operators and data structures for linear elastic
 * analysis
 *
 * Implemention of operators for Hooke material. Implementation is extended to
 * the case when the mesh is moving as results of topological changes, also the
 * calculation of material forces and associated tangent matrices are added to
 * implementation.
 *
 * In other words, spatial deformation is small but topological changes large.
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

#ifndef __HOOKE_ELEMENT_HPP
#define __HOOKE_ELEMENT_HPP

#ifndef __BASICFINITEELEMENTS_HPP__
#include <BasicFiniteElements.hpp>
#endif // __BASICFINITEELEMENTS_HPP__

#ifndef __NONLINEAR_ELASTIC_HPP

struct NonlinearElasticElement {

  /** \brief data for calculation heat conductivity and heat capacity elements
   * \ingroup nonlinear_elastic_elem
   */
  struct BlockData {
    int iD;
    double E;
    double PoissonRatio;
    Range tEts; ///< constrains elements in block set
    Range forcesOnlyOnEntitiesRow;
    Range forcesOnlyOnEntitiesCol;
  };
};

#endif // __NONLINEAR_ELASTIC_HPP

/** \brief structure grouping operators and data used for calculation of
 * nonlinear elastic element \ingroup nonlinear_elastic_elem
 *
 * In order to assemble matrices and right hand vectors, the loops over
 * elements, entities over that elements and finally loop over integration
 * points are executed.
 *
 * Following implementation separate those three categories of loops and to each
 * loop attach operator.
 *
 */

#ifndef __CONVECTIVE_MASS_ELEMENT_HPP
struct ConvectiveMassElement {
  /** \brief data for calculation inertia forces
   * \ingroup user_modules
   */
  struct BlockData {
    double rho0;     ///< reference density
    VectorDouble a0; ///< constant acceleration
    Range tEts;      ///< elements in block set
  };
}

#endif //__CONVECTIVE_MASS_ELEMENT_HPP
struct HookeElement {

  using BlockData = NonlinearElasticElement::BlockData;
  using MassBlockData = ConvectiveMassElement::BlockData;

  using EntData = DataForcesAndSourcesCore::EntData;
  using UserDataOperator = ForcesAndSourcesCore::UserDataOperator;
  using VolUserDataOperator =
      VolumeElementForcesAndSourcesCore::UserDataOperator;

  struct DataAtIntegrationPts {

    boost::shared_ptr<MatrixDouble> smallStrainMat;
    boost::shared_ptr<MatrixDouble> hMat;
    boost::shared_ptr<MatrixDouble> FMat;

    boost::shared_ptr<MatrixDouble> HMat;
    boost::shared_ptr<VectorDouble> detHVec;
    boost::shared_ptr<MatrixDouble> invHMat;

    boost::shared_ptr<MatrixDouble> cauchyStressMat;
    boost::shared_ptr<MatrixDouble> stiffnessMat;
    boost::shared_ptr<VectorDouble> energyVec;
    boost::shared_ptr<MatrixDouble> eshelbyStressMat;

    boost::shared_ptr<MatrixDouble> eshelbyStress_dx;

    boost::shared_ptr<MatrixDouble> internalStressMat;
    boost::shared_ptr<MatrixDouble> actualStressMat;
    boost::shared_ptr<MatrixDouble> deviatoricStressMat;
    boost::shared_ptr<MatrixDouble> hydrostaticStressMat;

    boost::shared_ptr<MatrixDouble> spatPosMat;
    boost::shared_ptr<MatrixDouble> meshNodePosMat;

    DataAtIntegrationPts() {

      smallStrainMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      hMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      FMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      HMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      detHVec = boost::shared_ptr<VectorDouble>(new VectorDouble());
      invHMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      cauchyStressMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      stiffnessMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      energyVec = boost::shared_ptr<VectorDouble>(new VectorDouble());
      eshelbyStressMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      // stiffnessMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      eshelbyStress_dx = boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      internalStressMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      actualStressMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      deviatoricStressMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      hydrostaticStressMat =
          boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      spatPosMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      meshNodePosMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    }

    Range forcesOnlyOnEntitiesRow;
    Range forcesOnlyOnEntitiesCol;
  };

  template <bool D = false>
  struct OpCalculateStrain : public VolUserDataOperator {

    OpCalculateStrain(const std::string row_field, const std::string col_field,
                      boost::shared_ptr<DataAtIntegrationPts> &data_at_pts);

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  private:
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
  };

  struct OpCalculateStrainAle : public VolUserDataOperator {

    OpCalculateStrainAle(const std::string row_field,
                         const std::string col_field,
                         boost::shared_ptr<DataAtIntegrationPts> &data_at_pts);

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  private:
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
  };

#define MAT_TO_DDG(SM)                                                         \
  &(*SM)(0, 0), &(*SM)(1, 0), &(*SM)(2, 0), &(*SM)(3, 0), &(*SM)(4, 0),        \
      &(*SM)(5, 0), &(*SM)(6, 0), &(*SM)(7, 0), &(*SM)(8, 0), &(*SM)(9, 0),    \
      &(*SM)(10, 0), &(*SM)(11, 0), &(*SM)(12, 0), &(*SM)(13, 0),              \
      &(*SM)(14, 0), &(*SM)(15, 0), &(*SM)(16, 0), &(*SM)(17, 0),              \
      &(*SM)(18, 0), &(*SM)(19, 0), &(*SM)(20, 0), &(*SM)(21, 0),              \
      &(*SM)(22, 0), &(*SM)(23, 0), &(*SM)(24, 0), &(*SM)(25, 0),              \
      &(*SM)(26, 0), &(*SM)(27, 0), &(*SM)(28, 0), &(*SM)(29, 0),              \
      &(*SM)(30, 0), &(*SM)(31, 0), &(*SM)(32, 0), &(*SM)(33, 0),              \
      &(*SM)(34, 0), &(*SM)(35, 0)

  template <int S = 0> struct OpCalculateStress : public VolUserDataOperator {

    OpCalculateStress(const std::string row_field, const std::string col_field,
                      boost::shared_ptr<DataAtIntegrationPts> data_at_pts);

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  protected:
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
  };

  struct OpCalculateEnergy : public VolUserDataOperator {

    OpCalculateEnergy(const std::string row_field, const std::string col_field,
                      boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
                      Vec ghost_vec = PETSC_NULL);

    ~OpCalculateEnergy();

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  protected:
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
    Vec ghostVec;
  };

  struct OpCalculateEshelbyStress : public VolUserDataOperator {

    OpCalculateEshelbyStress(
        const std::string row_field, const std::string col_field,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts);

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  protected:
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
  };

  template <int S = 0>
  struct OpCalculateHomogeneousStiffness : public VolUserDataOperator {

    OpCalculateHomogeneousStiffness(
        const std::string row_field, const std::string col_field,
        boost::shared_ptr<map<int, BlockData>> &block_sets_ptr,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts);

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  protected:
    boost::shared_ptr<map<int, BlockData>>
        blockSetsPtr; ///< Structure keeping data about problem, like
                      ///< material parameters
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
  };

  /** * @brief Assemble mass matrix for elastic element TODO: CHANGE FORMULA *
   * \f[
   * {\bf{M}} = \int\limits_\Omega
   * \f]
   *
   */
  struct OpCalculateMassMatrix
      : public VolumeElementForcesAndSourcesCore::UserDataOperator {

    MatrixDouble locK;
    MatrixDouble translocK;
    BlockData &dAta;
    MassBlockData &massData;

    boost::shared_ptr<DataAtIntegrationPts> commonData;

    OpCalculateMassMatrix(const std::string row_field,
                          const std::string col_field, BlockData &data,
                          MassBlockData &mass_data,
                          boost::shared_ptr<DataAtIntegrationPts> &common_data,
                          bool symm = true)
        : VolumeElementForcesAndSourcesCore::UserDataOperator(
              row_field, col_field, OPROWCOL, symm),
          commonData(common_data), dAta(data), massData(mass_data) {}

    PetscErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;

      auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
        return FTensor::Tensor2<double *, 3, 3>(
            &m(3 * r + 0, 3 * c + 0), &m(3 * r + 0, 3 * c + 1),
            &m(3 * r + 0, 3 * c + 2), &m(3 * r + 1, 3 * c + 0),
            &m(3 * r + 1, 3 * c + 1), &m(3 * r + 1, 3 * c + 2),
            &m(3 * r + 2, 3 * c + 0), &m(3 * r + 2, 3 * c + 1),
            &m(3 * r + 2, 3 * c + 2));
      };

      const int row_nb_dofs = row_data.getIndices().size();
      if (!row_nb_dofs)
        MoFEMFunctionReturnHot(0);
      const int col_nb_dofs = col_data.getIndices().size();
      if (!col_nb_dofs)
        MoFEMFunctionReturnHot(0);
      if (dAta.tEts.find(getFEEntityHandle()) == dAta.tEts.end()) {
        MoFEMFunctionReturnHot(0);
      }
      if (massData.tEts.find(getFEEntityHandle()) == massData.tEts.end()) {
        MoFEMFunctionReturnHot(0);
      }

      const bool diagonal_block =
          (row_type == col_type) && (row_side == col_side);
      // get number of integration points
      // Set size can clear local tangent matrix
      locK.resize(row_nb_dofs, col_nb_dofs, false);
      locK.clear();

      const int row_nb_gauss_pts = row_data.getN().size1();
      const int row_nb_base_functions = row_data.getN().size2();

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      FTensor::Index<'l', 3> l;

      double density = massData.rho0;

      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();

      // integrate local matrix for entity block
      for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

        auto t_row_base_func = row_data.getFTensor0N(gg, 0);

        // Get volume and integration weight
        double w = getVolume() * t_w;

        for (int row_bb = 0; row_bb != row_nb_dofs / 3; row_bb++) {
          auto t_col_base_func = col_data.getFTensor0N(gg, 0);
          for (int col_bb = 0; col_bb != col_nb_dofs / 3; col_bb++) {
            auto t_assemble = get_tensor2(locK, row_bb, col_bb);
            t_assemble(i, j) += density * t_row_base_func * t_col_base_func * w;
            // Next base function for column
            ++t_col_base_func;
          }
          // Next base function for row
          ++t_row_base_func;
        }
        // Next integration point for getting weight
        ++t_w;
      }

      CHKERR MatSetValues(getKSPB(), row_data, col_data, &locK(0, 0),
                          ADD_VALUES);

      // is symmetric
      if (row_type != col_type || row_side != col_side) {
        translocK.resize(col_nb_dofs, row_nb_dofs, false);
        noalias(translocK) = trans(locK);

        CHKERR MatSetValues(getKSPB(), col_data, row_data, &translocK(0, 0),
                            ADD_VALUES);
      }

      MoFEMFunctionReturn(0);
    }
  };

  struct OpCalculateStiffnessScaledByDensityField : public VolUserDataOperator {
  protected:
    boost::shared_ptr<map<int, BlockData>>
        blockSetsPtr; ///< Structure keeping data about problem, like
                      ///< material parameters
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;

    boost::shared_ptr<VectorDouble> rhoAtGaussPtsPtr;
    const double rhoN; ///< exponent n in E(p) = E * (p / p_0)^n
    const double rHo0; ///< p_0 reference density in E(p) = E * (p / p_0)^n
                       // // where p is density, E - youngs modulus
  public:
    OpCalculateStiffnessScaledByDensityField(
        const std::string row_field, const std::string col_field,
        boost::shared_ptr<map<int, BlockData>> &block_sets_ptr,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
        boost::shared_ptr<VectorDouble> rho_at_gauss_pts, const double rho_n,
        const double rho_0);

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);
  };

  struct OpAssemble : public VolUserDataOperator {

    OpAssemble(const std::string row_field, const std::string col_field,
               boost::shared_ptr<DataAtIntegrationPts> &data_at_pts,
               const char type, bool symm = false);

    /**
     * \brief Do calculations for give operator
     * @param  row_side row side number (local number) of entity on element
     * @param  col_side column side number (local number) of entity on element
     * @param  row_type type of row entity MBVERTEX, MBEDGE, MBTRI or MBTET
     * @param  col_type type of column entity MBVERTEX, MBEDGE, MBTRI or MBTET
     * @param  row_data data for row
     * @param  col_data data for column
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  protected:
    // Finite element stiffness sub-matrix K_ij
    MatrixDouble K;
    MatrixDouble transK;
    VectorDouble nF;

    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;

    VectorInt rowIndices;
    VectorInt colIndices;

    int nbRows;           ///< number of dofs on rows
    int nbCols;           ///< number if dof on column
    int nbIntegrationPts; ///< number of integration points
    bool isDiag;          ///< true if this block is on diagonal

    virtual MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);

    virtual MoFEMErrorCode iNtegrate(EntData &row_data);

    /**
     * \brief Assemble local entity block matrix
     * @param  row_data row data (consist base functions on row entity)
     * @param  col_data column data (consist base functions on column
     * entity)
     * @return          error code
     */
    MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);

    /**
     * \brief Assemble local entity right-hand vector
     * @param  row_data row data (consist base functions on row entity)
     * @param  col_data column data (consist base functions on column
     * entity)
     * @return          error code
     */
    MoFEMErrorCode aSsemble(EntData &row_data);
  };

  struct OpRhs_dx : public OpAssemble {

    OpRhs_dx(const std::string row_field, const std::string col_field,
             boost::shared_ptr<DataAtIntegrationPts> &data_at_pts);

  protected:
    MoFEMErrorCode iNtegrate(EntData &row_data);
  };

  template <int S = 0> struct OpLhs_dx_dx : public OpAssemble {

    OpLhs_dx_dx(const std::string row_field, const std::string col_field,
                boost::shared_ptr<DataAtIntegrationPts> &data_at_pts);

  protected:
    /**
     * \brief Integrate B^T D B operator
     * @param  row_data row data (consist base functions on row entity)
     * @param  col_data column data (consist base functions on column entity)
     * @return error code
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  struct OpAleRhs_dx : public OpAssemble {

    OpAleRhs_dx(const std::string row_field, const std::string col_field,
                boost::shared_ptr<DataAtIntegrationPts> &data_at_pts);

  protected:
    MoFEMErrorCode iNtegrate(EntData &row_data);
  };

  template <int S = 0> struct OpAleLhs_dx_dx : public OpAssemble {

    OpAleLhs_dx_dx(const std::string row_field, const std::string col_field,
                   boost::shared_ptr<DataAtIntegrationPts> &data_at_pts);

  protected:
    /**
     * \brief Integrate B^T D B operator
     * @param  row_data row data (consist base functions on row entity)
     * @param  col_data column data (consist base functions on column entity)
     * @return error code
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  template <int S = 0> struct OpAleLhs_dx_dX : public OpAssemble {

    OpAleLhs_dx_dX(const std::string row_field, const std::string col_field,
                   boost::shared_ptr<DataAtIntegrationPts> &data_at_pts);

  protected:
    /**
     * \brief Integrate tangent stiffness for spatial momentum
     * @param  row_data row data (consist base functions on row entity)
     * @param  col_data column data (consist base functions on column entity)
     * @return error code
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  struct OpAleLhsWithDensity_dx_dX : public OpAssemble {

    boost::shared_ptr<VectorDouble> rhoAtGaussPtsPtr;
    boost::shared_ptr<MatrixDouble> rhoGradAtGaussPtsPtr;
    const double rhoN;
    const double rHo0;

    OpAleLhsWithDensity_dx_dX(
        const std::string row_field, const std::string col_field,
        boost::shared_ptr<DataAtIntegrationPts> &data_at_pts,
        boost::shared_ptr<VectorDouble> rho_at_gauss_pts,
        boost::shared_ptr<MatrixDouble> rho_grad_at_gauss_pts,
        const double rho_n, const double rho_0);

  protected:
    /**
     * \brief Integrate tangent stiffness for spatial momentum
     * @param  row_data row data (consist base functions on row entity)
     * @param  col_data column data (consist base functions on column entity)
     * @return error code
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  struct OpAleLhsWithDensity_dX_dX : public OpAssemble {

    boost::shared_ptr<VectorDouble> rhoAtGaussPtsPtr;
    boost::shared_ptr<MatrixDouble> rhoGradAtGaussPtsPtr;
    const double rhoN;
    const double rHo0;

    OpAleLhsWithDensity_dX_dX(
        const std::string row_field, const std::string col_field,
        boost::shared_ptr<DataAtIntegrationPts> &data_at_pts,
        boost::shared_ptr<VectorDouble> rho_at_gauss_pts,
        boost::shared_ptr<MatrixDouble> rho_grad_at_gauss_pts,
        const double rho_n, const double rho_0);

  protected:
    /**
     * \brief Integrate tangent stiffness for material momentum
     * @param  row_data row data (consist base functions on row entity)
     * @param  col_data column data (consist base functions on column entity)
     * @return error code
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  struct OpAleRhs_dX : public OpAssemble {

    OpAleRhs_dX(const std::string row_field, const std::string col_field,
                boost::shared_ptr<DataAtIntegrationPts> &data_at_pts);

  protected:
    MoFEMErrorCode iNtegrate(EntData &row_data);
  };

  template <int S = 0> struct OpAleLhs_dX_dX : public OpAssemble {

    OpAleLhs_dX_dX(const std::string row_field, const std::string col_field,
                   boost::shared_ptr<DataAtIntegrationPts> &data_at_pts);

  protected:
    /**
     * \brief Integrate tangent stiffness for material momentum
     * @param  row_data row data (consist base functions on row entity)
     * @param  col_data column data (consist base functions on column entity)
     * @return error code
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  template <int S = 0> struct OpAleLhsPre_dX_dx : public VolUserDataOperator {

    OpAleLhsPre_dX_dx(const std::string row_field, const std::string col_field,
                      boost::shared_ptr<DataAtIntegrationPts> &data_at_pts);

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  private:
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
  };

  struct OpAleLhs_dX_dx : public OpAssemble {

    OpAleLhs_dX_dx(const std::string row_field, const std::string col_field,
                   boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
        : OpAssemble(row_field, col_field, data_at_pts, OPROWCOL, false) {}

  protected:
    /**
     * \brief Integrate tangent stiffness for material momentum
     * @param  row_data row data (consist base functions on row entity)
     * @param  col_data column data (consist base functions on column entity)
     * @return error code
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  template <int S> struct OpAnalyticalInternalStain_dx : public OpAssemble {

    typedef boost::function<

        FTensor::Tensor2_symmetric<double, 3>(

            FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> &t_coords

            )

        >
        StrainFunctions;

    OpAnalyticalInternalStain_dx(
        const std::string row_field,
        boost::shared_ptr<DataAtIntegrationPts> &data_at_pts,
        StrainFunctions strain_fun);

  protected:
    MoFEMErrorCode iNtegrate(EntData &row_data);
    StrainFunctions strainFun;
  };

  struct OpInternalStain_dx : public OpAssemble {

    OpInternalStain_dx(const std::string row_field,
                       boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
        : OpAssemble(row_field, row_field, data_at_pts, OPROW, true){};

  protected:
    MoFEMErrorCode iNtegrate(EntData &row_data) {
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

        if (getHoGaussPtsDetJac().size()) {
          // If HO geometry
          a *= getHoGaussPtsDetJac()[gg];
        }

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
    };
  };

  template <int S = 0>
  struct OpGetAnalyticalInternalStress : public VolUserDataOperator {

    typedef boost::function<

        FTensor::Tensor2_symmetric<double, 3>(

            FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> &t_coords

            )

        >
        StrainFunctions;

    OpGetAnalyticalInternalStress(
        const std::string row_field, const std::string col_field,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
        StrainFunctions strain_fun);

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  protected:
    StrainFunctions strainFun;
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
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

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntData &row_data) {
      MoFEMFunctionBegin;

      const int nb_integration_pts = row_data.getN().size1();

      const EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();

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

  protected:
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
    moab::Interface &inputMesh;
    char *stressTagName;
  };

  struct OpClearInternalStress : public VolUserDataOperator {

    OpClearInternalStress(const std::string row_field,
                          const std::string col_field,
                          boost::shared_ptr<DataAtIntegrationPts> data_at_pts)
        : VolUserDataOperator(row_field, col_field, OPROW, true),
          dataAtPts(data_at_pts) {
      std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
    };

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntData &row_data) {
      MoFEMFunctionBegin;

      const int nb_integration_pts = row_data.getN().size1();

      dataAtPts->internalStressMat->resize(9, nb_integration_pts, false);
      dataAtPts->internalStressMat->clear();

      MoFEMFunctionReturn(0);
    }

  protected:
    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
  };

  template <int S> struct OpAnalyticalInternalAleStain_dX : public OpAssemble {

    typedef boost::function<

        FTensor::Tensor2_symmetric<double, 3>(

            FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3> &t_coords

            )

        >
        StrainFunctions;

    OpAnalyticalInternalAleStain_dX(
        const std::string row_field,
        boost::shared_ptr<DataAtIntegrationPts> &data_at_pts,
        StrainFunctions strain_fun,
        boost::shared_ptr<MatrixDouble> mat_pos_at_pts_ptr);

  protected:
    MoFEMErrorCode iNtegrate(EntData &row_data);
    StrainFunctions strainFun;
    boost::shared_ptr<MatrixDouble> matPosAtPtsPtr;
  };

  template <int S> struct OpAnalyticalInternalAleStain_dx : public OpAssemble {

    typedef boost::function<

        FTensor::Tensor2_symmetric<double, 3>(

            FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3> &t_coords

            )

        >
        StrainFunctions;

    OpAnalyticalInternalAleStain_dx(
        const std::string row_field,
        boost::shared_ptr<DataAtIntegrationPts> &data_at_pts,
        StrainFunctions strain_fun,
        boost::shared_ptr<MatrixDouble> mat_pos_at_pts_ptr);

  protected:
    MoFEMErrorCode iNtegrate(EntData &row_data);
    StrainFunctions strainFun;
    boost::shared_ptr<MatrixDouble> matPosAtPtsPtr;
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
                           bool is_ale = false, bool is_field_disp = true);

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
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

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntData &row_data) {
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

      const int nb_integration_pts = row_data.getN().size1();

      const int val_num = 9 * nb_integration_pts;
      std::vector<double> def_vals(val_num, 0.0);

      Tag th_internal_stress;
      CHKERR outputMesh.tag_get_handle(
          "INTERNAL_STRESS", val_num, MB_TYPE_DOUBLE, th_internal_stress,
          MB_TAG_CREAT | MB_TAG_SPARSE, &*def_vals.begin());

      Tag th_actual_stress;
      CHKERR outputMesh.tag_get_handle(
          "ACTUAL_STRESS", val_num, MB_TYPE_DOUBLE, th_actual_stress,
          MB_TAG_CREAT | MB_TAG_SPARSE, &*def_vals.begin());

      Tag th_deviatoric_stress;
      CHKERR outputMesh.tag_get_handle(
          "DEVIATORIC_STRESS", val_num, MB_TYPE_DOUBLE, th_deviatoric_stress,
          MB_TAG_CREAT | MB_TAG_SPARSE, &*def_vals.begin());

      Tag th_hydrostatic_stress;
      CHKERR outputMesh.tag_get_handle(
          "HYDROSTATIC_STRESS", val_num, MB_TYPE_DOUBLE, th_hydrostatic_stress,
          MB_TAG_CREAT | MB_TAG_SPARSE, &*def_vals.begin());

      const int val_num_mean = 9;
      std::vector<double> def_vals_mean(val_num, 0.0);

      Tag th_internal_stress_mean;
      CHKERR outputMesh.tag_get_handle("MED_INTERNAL_STRESS", val_num_mean,
                                       MB_TYPE_DOUBLE, th_internal_stress_mean,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       &*def_vals_mean.begin());

      Tag th_actual_stress_mean;
      CHKERR outputMesh.tag_get_handle("MED_ACTUAL_STRESS", val_num_mean,
                                       MB_TYPE_DOUBLE, th_actual_stress_mean,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       &*def_vals_mean.begin());

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
      auto t_actual_stress =
          getFTensor2FromMat<3, 3>(*dataAtPts->actualStressMat);

      dataAtPts->deviatoricStressMat->resize(9, nb_integration_pts, false);
      auto t_deviatoric_stress =
          getFTensor2FromMat<3, 3>(*dataAtPts->deviatoricStressMat);
      dataAtPts->hydrostaticStressMat->resize(9, nb_integration_pts, false);
      auto t_hydrostatic_stress =
          getFTensor2FromMat<3, 3>(*dataAtPts->hydrostaticStressMat);

      VectorDouble vec_actual_stress_mean;
      vec_actual_stress_mean.resize(9, false);
      vec_actual_stress_mean.clear();

      VectorDouble vec_internal_stress_mean;
      vec_internal_stress_mean.resize(9, false);
      vec_internal_stress_mean.clear();

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
          t_small_strain_symm(0, 0) -= 1;
          t_small_strain_symm(1, 1) -= 1;
          t_small_strain_symm(2, 2) -= 1;
        }

        // symmetric tensors need improvement
        t_stress_symm(i, j) = t_D(i, j, k, l) * t_small_strain_symm(k, l);

        tensor_to_tensor(t_stress_symm, t_stress);

        t_actual_stress(i, j) = t_stress(i, j) + t_internal_stress(i, j);

        t_actual_stress(i, j) *= scaleFactor;
        t_internal_stress(i, j) *= scaleFactor;

        if (!saveMean) {
          t_actual_stress(i, j) *= -1;
        }

        double hydrostatic_pressure =
            (t_actual_stress(0, 0) + t_actual_stress(1, 1) +
             t_actual_stress(2, 2)) /
            3.;

        t_hydrostatic_stress(i, j) = 0.;
        t_hydrostatic_stress(0, 0) += hydrostatic_pressure;
        t_hydrostatic_stress(1, 1) += hydrostatic_pressure;
        t_hydrostatic_stress(2, 2) += hydrostatic_pressure;

        t_deviatoric_stress(i, j) = t_actual_stress(i, j);
        t_deviatoric_stress(0, 0) -= hydrostatic_pressure;
        t_deviatoric_stress(1, 1) -= hydrostatic_pressure;
        t_deviatoric_stress(2, 2) -= hydrostatic_pressure;

        vec_actual_stress_mean[0] += t_actual_stress(0, 0);
        vec_actual_stress_mean[1] += t_actual_stress(1, 1);
        vec_actual_stress_mean[2] += t_actual_stress(2, 2);
        vec_actual_stress_mean[3] += t_actual_stress(0, 1);
        vec_actual_stress_mean[4] += t_actual_stress(0, 2);
        vec_actual_stress_mean[5] += t_actual_stress(1, 2);

        vec_internal_stress_mean[0] += t_internal_stress(0, 0);
        vec_internal_stress_mean[1] += t_internal_stress(1, 1);
        vec_internal_stress_mean[2] += t_internal_stress(2, 2);
        vec_internal_stress_mean[3] += t_internal_stress(0, 1);
        vec_internal_stress_mean[4] += t_internal_stress(0, 2);
        vec_internal_stress_mean[5] += t_internal_stress(1, 2);

        ++t_h;
        ++t_actual_stress;
        ++t_internal_stress;
      }

      for (int ii = 0; ii < 9; ii++) {
        vec_actual_stress_mean[ii] /= nb_integration_pts;
        vec_internal_stress_mean[ii] /= nb_integration_pts;
      }

      if (saveMean) {
        CHKERR outputMesh.tag_set_data(
            th_actual_stress_mean, &ent, 1,
            &*(vec_actual_stress_mean.data().begin()));
        CHKERR outputMesh.tag_set_data(
            th_internal_stress_mean, &ent, 1,
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
    };
  };

  static MoFEMErrorCode
  setBlocks(MoFEM::Interface &m_field,
            boost::shared_ptr<map<int, BlockData>> &block_sets_ptr);

  static MoFEMErrorCode
  addElasticElement(MoFEM::Interface &m_field,
                    boost::shared_ptr<map<int, BlockData>> &block_sets_ptr,
                    const std::string element_name, const std::string x_field,
                    const std::string X_field, const bool ale);

  static MoFEMErrorCode
  setOperators(boost::shared_ptr<ForcesAndSourcesCore> fe_lhs_ptr,
               boost::shared_ptr<ForcesAndSourcesCore> fe_rhs_ptr,
               boost::shared_ptr<map<int, BlockData>> block_sets_ptr,
               const std::string x_field, const std::string X_field,
               const bool ale, const bool field_disp,
               const EntityType type = MBTET,
               boost::shared_ptr<DataAtIntegrationPts> data_at_pts = nullptr);

  static MoFEMErrorCode
  calculateEnergy(DM dm, boost::shared_ptr<map<int, BlockData>> block_sets_ptr,
                  const std::string x_field, const std::string X_field,
                  const bool ale, const bool field_disp, Vec *v_energy_ptr);

private:
  MatrixDouble invJac;
};

template <bool D>
HookeElement::OpCalculateStrain<D>::OpCalculateStrain(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
    : VolUserDataOperator(row_field, col_field, OPROW, true),
      dataAtPts(data_at_pts) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

template <bool D>
MoFEMErrorCode HookeElement::OpCalculateStrain<D>::doWork(int row_side,
                                                          EntityType row_type,
                                                          EntData &row_data) {
  MoFEMFunctionBegin;
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  // get number of integration points
  const int nb_integration_pts = getGaussPts().size2();
  dataAtPts->smallStrainMat->resize(6, nb_integration_pts, false);
  auto t_strain = getFTensor2SymmetricFromMat<3>(*(dataAtPts->smallStrainMat));
  auto t_h = getFTensor2FromMat<3, 3>(*(dataAtPts->hMat));

  for (int gg = 0; gg != nb_integration_pts; ++gg) {
    t_strain(i, j) = (t_h(i, j) || t_h(j, i)) / 2.;

    // If displacement field, not field o spatial positons is given
    if (!D) {
      t_strain(0, 0) -= 1;
      t_strain(1, 1) -= 1;
      t_strain(2, 2) -= 1;
    }

    ++t_strain;
    ++t_h;
  }
  MoFEMFunctionReturn(0);
}

template <int S>
HookeElement::OpAleLhs_dx_dx<S>::OpAleLhs_dx_dx(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
    : OpAssemble(row_field, col_field, data_at_pts, OPROWCOL, true) {}

template <int S>
MoFEMErrorCode HookeElement::OpAleLhs_dx_dx<S>::iNtegrate(EntData &row_data,
                                                          EntData &col_data) {
  MoFEMFunctionBegin;

  // get sub-block (3x3) of local stiffens matrix, here represented by
  // second order tensor
  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  // get element volume
  double vol = getVolume();

  // get intergrayion weights
  auto t_w = getFTensor0IntegrationWeight();

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();

  // Elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));

  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);
  auto &det_H = *dataAtPts->detHVec;

  // iterate over integration points
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {

    // calculate scalar weight times element volume
    double a = t_w * vol * det_H[gg];

    // iterate over row base functions
    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {

      // get sub matrix for the row
      auto t_m = get_tensor2(K, 3 * rr, 0);

      FTensor::Tensor1<double, 3> t_row_diff_base_pulled;
      t_row_diff_base_pulled(i) = t_row_diff_base(j) * t_invH(j, i);

      FTensor::Christof<double, 3, 3> t_rowD;
      // I mix up the indices here so that it behaves like a
      // Dg.  That way I don't have to have a separate wrapper
      // class Christof_Expr, which simplifies things.
      t_rowD(l, j, k) = t_D(i, j, k, l) * (a * t_row_diff_base_pulled(i));

      // get derivatives of base functions for columns
      auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

      // iterate column base functions
      for (int cc = 0; cc != nbCols / 3; ++cc) {

        FTensor::Tensor1<double, 3> t_col_diff_base_pulled;
        t_col_diff_base_pulled(j) = t_col_diff_base(i) * t_invH(i, j);

        // integrate block local stiffens matrix
        t_m(i, j) += t_rowD(i, j, k) * t_col_diff_base_pulled(k);

        // move to next column base function
        ++t_col_diff_base;

        // move to next block of local stiffens matrix
        ++t_m;
      }

      // move to next row base function
      ++t_row_diff_base;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    // move to next integration weight
    ++t_w;
    ++t_D;
    ++t_invH;
  }

  MoFEMFunctionReturn(0);
}

template <int S>
HookeElement::OpCalculateHomogeneousStiffness<S>::
    OpCalculateHomogeneousStiffness(
        const std::string row_field, const std::string col_field,
        boost::shared_ptr<map<int, BlockData>> &block_sets_ptr,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts)
    : VolUserDataOperator(row_field, col_field, OPROW, true),
      blockSetsPtr(block_sets_ptr), dataAtPts(data_at_pts) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

template <int S>
MoFEMErrorCode HookeElement::OpCalculateHomogeneousStiffness<S>::doWork(
    int row_side, EntityType row_type, EntData &row_data) {
  MoFEMFunctionBegin;

  for (auto &m : (*blockSetsPtr)) {
    if (m.second.tEts.find(getFEEntityHandle()) != m.second.tEts.end()) {

      dataAtPts->stiffnessMat->resize(36, 1, false);
      FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
          MAT_TO_DDG(dataAtPts->stiffnessMat));
      const double young = m.second.E;
      const double poisson = m.second.PoissonRatio;

      // coefficient used in intermediate calculation
      const double coefficient = young / ((1 + poisson) * (1 - 2 * poisson));

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      FTensor::Index<'l', 3> l;

      t_D(i, j, k, l) = 0.;

      t_D(0, 0, 0, 0) = 1 - poisson;
      t_D(1, 1, 1, 1) = 1 - poisson;
      t_D(2, 2, 2, 2) = 1 - poisson;

      t_D(0, 1, 0, 1) = 0.5 * (1 - 2 * poisson);
      t_D(0, 2, 0, 2) = 0.5 * (1 - 2 * poisson);
      t_D(1, 2, 1, 2) = 0.5 * (1 - 2 * poisson);

      t_D(0, 0, 1, 1) = poisson;
      t_D(1, 1, 0, 0) = poisson;
      t_D(0, 0, 2, 2) = poisson;
      t_D(2, 2, 0, 0) = poisson;
      t_D(1, 1, 2, 2) = poisson;
      t_D(2, 2, 1, 1) = poisson;

      t_D(i, j, k, l) *= coefficient;

      break;
    }
  }

  MoFEMFunctionReturn(0);
}

template <int S>
HookeElement::OpCalculateStress<S>::OpCalculateStress(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> data_at_pts)
    : VolUserDataOperator(row_field, col_field, OPROW, true),
      dataAtPts(data_at_pts) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

template <int S>
MoFEMErrorCode HookeElement::OpCalculateStress<S>::doWork(int row_side,
                                                          EntityType row_type,
                                                          EntData &row_data) {
  MoFEMFunctionBegin;
  // get number of integration points
  const int nb_integration_pts = getGaussPts().size2();
  auto t_strain = getFTensor2SymmetricFromMat<3>(*(dataAtPts->smallStrainMat));
  dataAtPts->cauchyStressMat->resize(6, nb_integration_pts, false);
  auto t_cauchy_stress =
      getFTensor2SymmetricFromMat<3>(*(dataAtPts->cauchyStressMat));

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  // elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));
  for (int gg = 0; gg != nb_integration_pts; ++gg) {
    t_cauchy_stress(i, j) = t_D(i, j, k, l) * t_strain(k, l);
    ++t_strain;
    ++t_cauchy_stress;
    ++t_D;
  }
  MoFEMFunctionReturn(0);
}

template <int S>
HookeElement::OpLhs_dx_dx<S>::OpLhs_dx_dx(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
    : OpAssemble(row_field, col_field, data_at_pts, OPROWCOL, true) {}

template <int S>
MoFEMErrorCode HookeElement::OpLhs_dx_dx<S>::iNtegrate(EntData &row_data,
                                                       EntData &col_data) {
  MoFEMFunctionBegin;

  // get sub-block (3x3) of local stiffens matrix, here represented by
  // second order tensor
  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  // get element volume
  double vol = getVolume();

  // get intergrayion weights
  auto t_w = getFTensor0IntegrationWeight();

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();

  // Elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));

  // iterate over integration points
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {

    // calculate scalar weight times element volume
    double a = t_w * vol;
    if (getHoGaussPtsDetJac().size()) {
      a *= getHoGaussPtsDetJac()[gg];
    }

    // iterate over row base functions
    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {

      // get sub matrix for the row
      auto t_m = get_tensor2(K, 3 * rr, 0);

      // get derivatives of base functions for columns
      auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

      FTensor::Christof<double, 3, 3> t_rowD;
      // I mix up the indices here so that it behaves like a
      // Dg.  That way I don't have to have a separate wrapper
      // class Christof_Expr, which simplifies things.
      t_rowD(l, j, k) = t_D(i, j, k, l) * (a * t_row_diff_base(i));

      // iterate column base functions
      for (int cc = 0; cc != nbCols / 3; ++cc) {

        // integrate block local stiffens matrix
        t_m(i, j) += t_rowD(i, j, k) * t_col_diff_base(k);

        // move to next column base function
        ++t_col_diff_base;

        // move to next block of local stiffens matrix
        ++t_m;
      }

      // move to next row base function
      ++t_row_diff_base;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    // move to next integration weight
    ++t_w;
    ++t_D;
  }

  MoFEMFunctionReturn(0);
}

template <int S>
HookeElement::OpAleLhs_dx_dX<S>::OpAleLhs_dx_dX(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
    : OpAssemble(row_field, col_field, data_at_pts, OPROWCOL, false) {}

template <int S>
MoFEMErrorCode HookeElement::OpAleLhs_dx_dX<S>::iNtegrate(EntData &row_data,
                                                          EntData &col_data) {
  MoFEMFunctionBegin;

  // get sub-block (3x3) of local stiffens matrix, here represented by
  // second order tensor
  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'n', 3> n;

  // get element volume
  double vol = getVolume();

  // get intergrayion weights
  auto t_w = getFTensor0IntegrationWeight();

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();

  // Elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));

  auto t_cauchy_stress =
      getFTensor2SymmetricFromMat<3>(*(dataAtPts->cauchyStressMat));
  auto t_h = getFTensor2FromMat<3, 3>(*dataAtPts->hMat);
  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);
  auto &det_H = *dataAtPts->detHVec;

  // iterate over integration points
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {

    // calculate scalar weight times element volume
    double a = t_w * vol * det_H[gg];

    FTensor::Tensor4<double, 3, 3, 3, 3> t_F_dX;
    t_F_dX(i, j, k, l) = -(t_h(i, m) * t_invH(m, k)) * t_invH(l, j);

    // iterate over row base functions
    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {

      // get sub matrix for the row
      auto t_m = get_tensor2(K, 3 * rr, 0);

      FTensor::Tensor1<double, 3> t_row_diff_base_pulled;
      t_row_diff_base_pulled(i) = t_row_diff_base(j) * t_invH(j, i);

      FTensor::Tensor1<double, 3> t_row_stress;
      t_row_stress(i) = a * t_row_diff_base_pulled(j) * t_cauchy_stress(i, j);

      FTensor::Tensor3<double, 3, 3, 3> t_row_diff_base_pulled_dX;
      t_row_diff_base_pulled_dX(j, k, l) =
          -(t_invH(i, k) * t_row_diff_base(i)) * t_invH(l, j);

      FTensor::Tensor3<double, 3, 3, 3> t_row_dX_stress;
      t_row_dX_stress(i, k, l) =
          a * (t_row_diff_base_pulled_dX(j, k, l) * t_cauchy_stress(j, i));

      FTensor::Christof<double, 3, 3> t_row_D;
      t_row_D(l, j, k) = (a * t_row_diff_base_pulled(i)) * t_D(i, j, k, l);

      FTensor::Tensor3<double, 3, 3, 3> t_row_stress_dX;
      // FIXME: This operator is not implemented, doing operation by hand
      // t_row_stress_dX(i, m, n) = t_row_D(i, k, l) * t_F_dX(k, l, m, n);
      t_row_stress_dX(i, j, k) = 0;
      for (int ii = 0; ii != 3; ++ii)
        for (int mm = 0; mm != 3; ++mm)
          for (int nn = 0; nn != 3; ++nn) {
            auto &v = t_row_stress_dX(ii, mm, nn);
            for (int kk = 0; kk != 3; ++kk)
              for (int ll = 0; ll != 3; ++ll)
                v += t_row_D(ii, kk, ll) * t_F_dX(kk, ll, mm, nn);
          }

      // get derivatives of base functions for columns
      auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

      // iterate column base functions
      for (int cc = 0; cc != nbCols / 3; ++cc) {

        t_m(i, k) += t_row_stress(i) * (t_invH(j, k) * t_col_diff_base(j));
        t_m(i, k) += t_row_dX_stress(i, k, l) * t_col_diff_base(l);
        t_m(i, k) += t_row_stress_dX(i, k, l) * t_col_diff_base(l);

        // move to next column base function
        ++t_col_diff_base;

        // move to next block of local stiffens matrix
        ++t_m;
      }

      // move to next row base function
      ++t_row_diff_base;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    // move to next integration weight
    ++t_w;
    ++t_D;
    ++t_cauchy_stress;
    ++t_invH;
    ++t_h;
  }

  MoFEMFunctionReturn(0);
}

template <int S>
HookeElement::OpAleLhs_dX_dX<S>::OpAleLhs_dX_dX(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
    : OpAssemble(row_field, col_field, data_at_pts, OPROWCOL, true) {}

template <int S>
MoFEMErrorCode HookeElement::OpAleLhs_dX_dX<S>::iNtegrate(EntData &row_data,
                                                          EntData &col_data) {
  MoFEMFunctionBegin;

  // get sub-block (3x3) of local stiffens matrix, here represented by
  // second order tensor
  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'n', 3> n;

  // get element volume
  double vol = getVolume();

  // get intergrayion weights
  auto t_w = getFTensor0IntegrationWeight();

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();

  // Elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));
  auto t_cauchy_stress =
      getFTensor2SymmetricFromMat<3>(*(dataAtPts->cauchyStressMat));
  auto t_strain = getFTensor2SymmetricFromMat<3>(*(dataAtPts->smallStrainMat));
  auto t_eshelby_stress =
      getFTensor2FromMat<3, 3>(*dataAtPts->eshelbyStressMat);
  auto t_h = getFTensor2FromMat<3, 3>(*dataAtPts->hMat);
  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);
  auto t_F = getFTensor2FromMat<3, 3>(*dataAtPts->FMat);
  auto &det_H = *dataAtPts->detHVec;

  // iterate over integration points
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {

    // calculate scalar weight times element volume
    double a = t_w * vol * det_H[gg];

    FTensor::Tensor4<double, 3, 3, 3, 3> t_F_dX;
    t_F_dX(i, j, k, l) = -(t_h(i, m) * t_invH(m, k)) * t_invH(l, j);

    FTensor::Tensor4<double, 3, 3, 3, 3> t_D_strain_dX;
    t_D_strain_dX(i, j, m, n) = 0.;
    for (int ii = 0; ii != 3; ++ii)
      for (int jj = 0; jj != 3; ++jj)
        for (int ll = 0; ll != 3; ++ll)
          for (int kk = 0; kk != 3; ++kk) {
            auto &v = t_D_strain_dX(ii, jj, kk, ll);
            for (int mm = 0; mm != 3; ++mm)
              for (int nn = 0; nn != 3; ++nn)
                v += t_D(ii, jj, mm, nn) * t_F_dX(mm, nn, kk, ll);
          }

    FTensor::Tensor4<double, 3, 3, 3, 3> t_eshelby_stress_dX;
    t_eshelby_stress_dX(i, j, m, n) = t_F(k, i) * t_D_strain_dX(k, j, m, n);

    for (int ii = 0; ii != 3; ++ii)
      for (int jj = 0; jj != 3; ++jj)
        for (int mm = 0; mm != 3; ++mm)
          for (int nn = 0; nn != 3; ++nn) {
            auto &v = t_eshelby_stress_dX(ii, jj, mm, nn);
            for (int kk = 0; kk != 3; ++kk)
              v += t_F_dX(kk, ii, mm, nn) * t_cauchy_stress(kk, jj);
          }

    t_eshelby_stress_dX(i, j, k, l) *= -1;

    FTensor::Tensor2<double, 3, 3> t_energy_dX;
    t_energy_dX(k, l) = t_F_dX(i, j, k, l) * t_cauchy_stress(i, j);
    t_energy_dX(k, l) +=
        (t_strain(m, n) * t_D(m, n, i, j)) * t_F_dX(i, j, k, l);
    t_energy_dX(k, l) /= 2.;

    for (int kk = 0; kk != 3; ++kk)
      for (int ll = 0; ll != 3; ++ll) {
        auto v = t_energy_dX(kk, ll);
        for (int ii = 0; ii != 3; ++ii)
          t_eshelby_stress_dX(ii, ii, kk, ll) += v;
      }

    // iterate over row base functions
    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {

      // get sub matrix for the row
      auto t_m = get_tensor2(K, 3 * rr, 0);

      FTensor::Tensor1<double, 3> t_row_diff_base_pulled;
      t_row_diff_base_pulled(i) = t_row_diff_base(j) * t_invH(j, i);

      FTensor::Tensor1<double, 3> t_row_stress;
      t_row_stress(i) = a * t_row_diff_base_pulled(j) * t_eshelby_stress(i, j);

      FTensor::Tensor3<double, 3, 3, 3> t_row_diff_base_pulled_dX;
      t_row_diff_base_pulled_dX(j, k, l) =
          -(t_row_diff_base(i) * t_invH(i, k)) * t_invH(l, j);

      FTensor::Tensor3<double, 3, 3, 3> t_row_dX_stress;
      t_row_dX_stress(i, k, l) =
          a * (t_row_diff_base_pulled_dX(j, k, l) * t_eshelby_stress(i, j));

      FTensor::Tensor3<double, 3, 3, 3> t_row_stress_dX;
      t_row_stress_dX(i, m, n) =
          a * t_row_diff_base_pulled(j) * t_eshelby_stress_dX(i, j, m, n);

      // get derivatives of base functions for columns
      auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

      // iterate column base functions
      for (int cc = 0; cc != nbCols / 3; ++cc) {

        t_m(i, k) += t_row_stress(i) * (t_invH(j, k) * t_col_diff_base(j));
        t_m(i, k) += t_row_dX_stress(i, k, l) * t_col_diff_base(l);
        t_m(i, k) += t_row_stress_dX(i, k, l) * t_col_diff_base(l);

        // move to next column base function
        ++t_col_diff_base;

        // move to next block of local stiffens matrix
        ++t_m;
      }

      // move to next row base function
      ++t_row_diff_base;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    // move to next integration weight
    ++t_w;
    ++t_D;
    ++t_cauchy_stress;
    ++t_strain;
    ++t_eshelby_stress;
    ++t_h;
    ++t_invH;
    ++t_F;
  }

  MoFEMFunctionReturn(0);
}

template <int S>
HookeElement::OpAleLhsPre_dX_dx<S>::OpAleLhsPre_dX_dx(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
    : VolUserDataOperator(row_field, col_field, OPROW, true),
      dataAtPts(data_at_pts) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

template <int S>
MoFEMErrorCode HookeElement::OpAleLhsPre_dX_dx<S>::doWork(int row_side,
                                                          EntityType row_type,
                                                          EntData &row_data) {
  MoFEMFunctionBegin;

  const int nb_integration_pts = row_data.getN().size1();

  auto get_eshelby_stress_dx = [this, nb_integration_pts]() {
    FTensor::Tensor4<FTensor::PackPtr<double *, 1>, 3, 3, 3, 3>
        t_eshelby_stress_dx;
    dataAtPts->eshelbyStress_dx->resize(81, nb_integration_pts, false);
    int mm = 0;
    for (int ii = 0; ii != 3; ++ii)
      for (int jj = 0; jj != 3; ++jj)
        for (int kk = 0; kk != 3; ++kk)
          for (int ll = 0; ll != 3; ++ll)
            t_eshelby_stress_dx.ptr(ii, jj, kk, ll) =
                &(*dataAtPts->eshelbyStress_dx)(mm++, 0);
    return t_eshelby_stress_dx;
  };

  auto t_eshelby_stress_dx = get_eshelby_stress_dx();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'n', 3> n;

  // Elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));
  auto t_cauchy_stress =
      getFTensor2SymmetricFromMat<3>(*(dataAtPts->cauchyStressMat));
  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);
  auto t_F = getFTensor2FromMat<3, 3>(*dataAtPts->FMat);

  for (int gg = 0; gg != nb_integration_pts; ++gg) {

    t_eshelby_stress_dx(i, j, m, n) =
        (t_F(k, i) * t_D(k, j, m, l)) * t_invH(n, l);
    for (int ii = 0; ii != 3; ++ii)
      for (int jj = 0; jj != 3; ++jj)
        for (int mm = 0; mm != 3; ++mm)
          for (int nn = 0; nn != 3; ++nn) {
            auto &v = t_eshelby_stress_dx(ii, jj, mm, nn);
            v += t_invH(nn, ii) * t_cauchy_stress(mm, jj);
          }
    t_eshelby_stress_dx(i, j, k, l) *= -1;

    FTensor::Tensor2<double, 3, 3> t_energy_dx;
    t_energy_dx(m, n) = t_invH(n, j) * t_cauchy_stress(m, j);

    for (int mm = 0; mm != 3; ++mm)
      for (int nn = 0; nn != 3; ++nn) {
        auto v = t_energy_dx(mm, nn);
        for (int ii = 0; ii != 3; ++ii)
          t_eshelby_stress_dx(ii, ii, mm, nn) += v;
      }

    ++t_D;
    ++t_invH;
    ++t_cauchy_stress;
    ++t_eshelby_stress_dx;
    ++t_F;
  }

  MoFEMFunctionReturn(0);
}

template <class ELEMENT>
HookeElement::OpPostProcHookeElement<ELEMENT>::OpPostProcHookeElement(
    const string row_field, boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
    map<int, BlockData> &block_sets_ptr, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts, bool is_ale, bool is_field_disp)
    : ELEMENT::UserDataOperator(row_field, UserDataOperator::OPROW),
      dataAtPts(data_at_pts), blockSetsPtr(block_sets_ptr),
      postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts), isALE(is_ale),
      isFieldDisp(is_field_disp) {}

template <class ELEMENT>
MoFEMErrorCode HookeElement::OpPostProcHookeElement<ELEMENT>::doWork(
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

  double def_VAL[9];
  bzero(def_VAL, 9 * sizeof(double));

  Tag th_stress;
  CHKERR postProcMesh.tag_get_handle("STRESS", 9, MB_TYPE_DOUBLE, th_stress,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
  // Tag th_actual_stress;
  // CHKERR postProcMesh.tag_get_handle("STRESS_ACTUAL", 9, MB_TYPE_DOUBLE,
  //                                    th_actual_stress,
  //                                    MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
  // Tag th_internal_stress;
  // CHKERR postProcMesh.tag_get_handle("STRESS_INTERNAL", 9, MB_TYPE_DOUBLE,
  //                                    th_internal_stress,
  //                                    MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
  Tag th_strain;
  CHKERR postProcMesh.tag_get_handle("STRAIN", 9, MB_TYPE_DOUBLE, th_strain,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
  Tag th_psi;
  CHKERR postProcMesh.tag_get_handle("ENERGY", 1, MB_TYPE_DOUBLE, th_psi,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

  double def_VAL_3[3];
  bzero(def_VAL_3, 3 * sizeof(double));
  Tag th_disp;
  CHKERR postProcMesh.tag_get_handle("DISPLACEMENT", 3, MB_TYPE_DOUBLE, th_disp,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL_3);

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
  // FTensor::Tensor2<double, 3, 3> t_actual_stress;

  FTensor::Tensor2_symmetric<double, 3> t_stress_symm;
  FTensor::Tensor2_symmetric<double, 3> t_small_strain_symm;

  // auto t_internal_stress =
  //     getFTensor2FromMat<3, 3>(*dataAtPts->internalStressMat);

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

    // t_actual_stress(i, j) = t_stress(i, j) + t_internal_stress(i, j);

    t_disp(i) = t_spat_pos(i) - t_mesh_node_pos(i);

    const double psi = 0.5 * t_stress_symm(i, j) * t_small_strain_symm(i, j);

    CHKERR postProcMesh.tag_set_data(th_psi, &mapGaussPts[gg], 1, &psi);
    CHKERR postProcMesh.tag_set_data(th_stress, &mapGaussPts[gg], 1,
                                     &t_stress(0, 0));
    // CHKERR postProcMesh.tag_set_data(th_actual_stress, &mapGaussPts[gg], 1,
    //                                  &t_actual_stress(0, 0));
    // CHKERR postProcMesh.tag_set_data(th_internal_stress, &mapGaussPts[gg], 1,
    //                                  &t_internal_stress(0, 0));
    CHKERR postProcMesh.tag_set_data(th_strain, &mapGaussPts[gg], 1,
                                     &t_small_strain(0, 0));
    CHKERR postProcMesh.tag_set_data(th_disp, &mapGaussPts[gg], 1, &t_disp(0));

    ++t_h;
    //++t_internal_stress;
    ++t_spat_pos;
    ++t_mesh_node_pos;
  }

  MoFEMFunctionReturn(0);
}

template <int S>
HookeElement::OpAnalyticalInternalStain_dx<S>::OpAnalyticalInternalStain_dx(
    const std::string row_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts,
    StrainFunctions strain_fun)
    : OpAssemble(row_field, row_field, data_at_pts, OPROW, true),
      strainFun(strain_fun) {}

template <int S>
MoFEMErrorCode
HookeElement::OpAnalyticalInternalStain_dx<S>::iNtegrate(EntData &row_data) {
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

  auto get_coords = [&]() {
    if (getHoCoordsAtGaussPts().size1() == nb_integration_pts)
      return getFTensor1HoCoordsAtGaussPts();
    else
      return getFTensor1CoordsAtGaussPts();
  };
  auto t_coords = get_coords();

  // get element volume
  double vol = getVolume();
  auto t_w = getFTensor0IntegrationWeight();

  nF.resize(nbRows, false);
  nF.clear();

  // elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

    auto t_fun_strain = strainFun(t_coords);
    FTensor::Tensor2_symmetric<double, 3> t_stress;
    t_stress(i, j) = t_D(i, j, k, l) * t_fun_strain(k, l);

    // calculate scalar weight times element volume
    double a = t_w * vol;

    if (getHoGaussPtsDetJac().size()) {
      // If HO geometry
      a *= getHoGaussPtsDetJac()[gg];
    }

    auto t_nf = get_tensor1(nF, 0);

    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {
      t_nf(i) -= a * t_row_diff_base(j) * t_stress(i, j);
      ++t_row_diff_base;
      ++t_nf;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    ++t_w;
    ++t_coords;
    ++t_D;
  }

  MoFEMFunctionReturn(0);
}

template <int S>
HookeElement::OpGetAnalyticalInternalStress<S>::OpGetAnalyticalInternalStress(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
    StrainFunctions strain_fun)
    : VolUserDataOperator(row_field, col_field, OPROW, true),
      dataAtPts(data_at_pts), strainFun(strain_fun) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

template <int S>
MoFEMErrorCode HookeElement::OpGetAnalyticalInternalStress<S>::doWork(
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

  auto get_coords = [&]() {
    if (getHoCoordsAtGaussPts().size1() == nb_integration_pts)
      return getFTensor1HoCoordsAtGaussPts();
    else
      return getFTensor1CoordsAtGaussPts();
  };
  auto t_coords = get_coords();

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

template <int S>
HookeElement::OpAnalyticalInternalAleStain_dX<S>::
    OpAnalyticalInternalAleStain_dX(
        const std::string row_field,
        boost::shared_ptr<DataAtIntegrationPts> &data_at_pts,
        StrainFunctions strain_fun,
        boost::shared_ptr<MatrixDouble> mat_pos_at_pts_ptr)
    : OpAssemble(row_field, row_field, data_at_pts, OPROW, true),
      strainFun(strain_fun), matPosAtPtsPtr(mat_pos_at_pts_ptr) {}

template <int S>
MoFEMErrorCode
HookeElement::OpAnalyticalInternalAleStain_dX<S>::iNtegrate(EntData &row_data) {
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

  auto get_coords = [&]() { return getFTensor1FromMat<3>(*matPosAtPtsPtr); };
  auto t_coords = get_coords();

  // get element volume
  double vol = getVolume();
  auto t_w = getFTensor0IntegrationWeight();

  nF.resize(nbRows, false);
  nF.clear();

  // elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));
  auto t_F = getFTensor2FromMat<3, 3>(*(dataAtPts->FMat));
  auto &det_H = *dataAtPts->detHVec;
  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

    auto t_fun_strain = strainFun(t_coords);
    FTensor::Tensor2_symmetric<double, 3> t_stress;
    t_stress(i, j) = t_D(i, j, k, l) * t_fun_strain(k, l);
    FTensor::Tensor2<double, 3, 3> t_eshelby_stress;
    t_eshelby_stress(i, j) = -t_F(k, i) * t_stress(k, j);

    // calculate scalar weight times element volume
    double a = t_w * vol * det_H[gg];

    auto t_nf = get_tensor1(nF, 0);

    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {
      FTensor::Tensor1<double, 3> t_row_diff_base_pulled;
      t_row_diff_base_pulled(i) = t_row_diff_base(j) * t_invH(j, i);
      t_nf(i) += a * t_row_diff_base_pulled(j) * t_eshelby_stress(i, j);
      ++t_row_diff_base;
      ++t_nf;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    ++t_w;
    ++t_coords;
    ++t_F;
    ++t_invH;
    ++t_D;
  }

  MoFEMFunctionReturn(0);
}

template <int S>
HookeElement::OpAnalyticalInternalAleStain_dx<S>::
    OpAnalyticalInternalAleStain_dx(
        const std::string row_field,
        boost::shared_ptr<DataAtIntegrationPts> &data_at_pts,
        StrainFunctions strain_fun,
        boost::shared_ptr<MatrixDouble> mat_pos_at_pts_ptr)
    : OpAssemble(row_field, row_field, data_at_pts, OPROW, true),
      strainFun(strain_fun), matPosAtPtsPtr(mat_pos_at_pts_ptr) {}

template <int S>
MoFEMErrorCode
HookeElement::OpAnalyticalInternalAleStain_dx<S>::iNtegrate(EntData &row_data) {
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

  auto get_coords = [&]() { return getFTensor1FromMat<3>(*matPosAtPtsPtr); };
  auto t_coords = get_coords();

  // get element volume
  double vol = getVolume();
  auto t_w = getFTensor0IntegrationWeight();

  nF.resize(nbRows, false);
  nF.clear();

  // elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  FTensor::Ddg<FTensor::PackPtr<double *, S>, 3, 3> t_D(
      MAT_TO_DDG(dataAtPts->stiffnessMat));
  auto &det_H = *dataAtPts->detHVec;
  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

    auto t_fun_strain = strainFun(t_coords);
    FTensor::Tensor2_symmetric<double, 3> t_stress;
    t_stress(i, j) = t_D(i, j, k, l) * t_fun_strain(k, l);

    // calculate scalar weight times element volume
    double a = t_w * vol * det_H[gg];

    auto t_nf = get_tensor1(nF, 0);

    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {
      FTensor::Tensor1<double, 3> t_row_diff_base_pulled;
      t_row_diff_base_pulled(i) = t_row_diff_base(j) * t_invH(j, i);
      t_nf(i) += a * t_row_diff_base_pulled(j) * t_stress(i, j);
      ++t_row_diff_base;
      ++t_nf;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    ++t_w;
    ++t_coords;
    ++t_invH;
    ++t_D;
  }

  MoFEMFunctionReturn(0);
}

#endif // __HOOKE_ELEMENT_HPP