/**
 * \file ElasticElement.hpp
 * \example ElasticElement.hpp
 *
 * \brief Operators and data structures for linear elastic
 * analysi
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


#ifndef __ELASTIC_ELEMENT_HPP
#define __ELASTIC_ELEMENT_HPP

#ifndef __BASICFINITEELEMENTS_HPP__
#include <BasicFiniteElements.hpp>
#endif // __BASICFINITEELEMENTS_HPP__

const double nu = 0.1;
const double E = 10;


auto wrap_matrix2_ftensor = [](MatrixDouble &m){
        return FTensor::Tensor2<FTensor::PackPtr<double *, 2>, 2, 2>(
                                       &m(0, 0), &m(0, 1),
                                       &m(1, 0), &m(1, 1));
}; 

auto wrap_matrix3_ftensor = [](MatrixDouble &m){
        return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>(
                                      &m(0, 0), &m(0, 1), &m(0, 2),
                                      &m(1, 0), &m(1, 1), &m(1, 2),
                                      &m(2, 0), &m(2, 1), &m(2, 2));
};

auto extract_subMatrix2_ftensor = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 2>, 2, 2>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), 
        &m(r + 1, c + 0), &m(r + 1, c + 1));
    };

auto wrap_subMatrix3_ftensor = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
        &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
        &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
};

double lambda = nu*E/((1 + nu) * (1 - 2.0 * nu));
double kappa = E/(2. * (1 + nu));        
MatrixDouble unit_mat;
unit_mat.resize(DIM, DIM);
auto t_unit = wrap_matrix2_ftensor(unit_mat);

unit_mat.clear();
unit_mat(0, 0) = unit_mat(1, 1) = 1.;
for(int m = 0; m<DIM; ++m){
    unit_mat(m, m) = 1;
}

FTensor::Index<'i', DIM> i;
FTensor::Index<'j', DIM> j;
FTensor::Index<'k', DIM> k;
FTensor::Index<'l', DIM> l;

tD(i, j, k, l) = mu * (t_unit(i, k) * t_unit(j, l) + 
                        t_unit(i, l) * t_unit(j, k)) + 
                    lambda * t_unit(i, j) * t_unit(k, l);

template <int DIM> 
struct OpElasticStiffness : OpEle
{
    OpElasticStiffness(boost::shared_ptr<CommonData> &data)
    : OpEle("U", "U", OpEle::OPROWCOL), commonData(data)
    {
        sYmm = true;
    }

    MoFEMErrorCode doWork(int                                 row_side, 
                          int                                 col_side,
                          EntityType                          row_type, 
                          EntityType                          col_type,
                          DataForcesAndSourcesCore::EntData   &row_data, 
                          DataForcesAndSourcesCore::EntData   &col_data)
    {
        MoFEMFunctionBeginHot;

        nbRows = row_data.getIndices().size();      
        nbCols = col_data.getIndices().size();
 
        if (nbRows && nbCols) {
            loc_mat.resize(nbRows, nbCols, false);
            loc_mat.clear();
            const int nb_integration_pts = getGaussPts().size2();

            auto t_row_shape_grad = row_data.getFTensor1DiffN<DIM>();

            auto t_w = getFTensor0IntegrationWeight();
            const double ts_a = getFEMethod()->ts_a;
            const double vol = getMeasure();  
            for (int gg = 0; gg != nb_integration_pts; ++gg) {
                const double j_x_w = vol * t_w;
                for (int rr = 0; rr != nb_row_dofs / DIM; ++rr) {
                    auto t_sub_mat = extract_subMatrix2_ftensor(loc_mat, DIM * rr, 0);
                    auto t_col_shape_grad = col_data.getFTensor1DiffN<3>(gg, 0);

                    for (int cc = 0; cc != nbCols / DIM;++cc) {
                        t_sub_mat(i, k) += (tD(i, j, k, l) * (t_row_shape_grad(j) *
                                            t_col_shape_grad(l))) * j_x_w;

                        ++t_col_shape_grad;
                        ++t_sub_mat;                    
                    }

                    ++t_row_shape_grad;
                    
                }
                ++t_w;
            }   
            CHKERR MatSetValues(getFEMethod()->ts_B, row_data, col_data, &loc_mat(0, 0),
                          ADD_VALUES); 
            if (row_side != col_side || row_type != col_type) {
                loc_transMat.resize(nb_col_dofs, nb_row_dofs, false);
                noalias(loc_transMat) = trans(loc_mat);
                CHKERR MatSetValues(getFEMethod()->ts_B, col_data, row_data,
                                    &loc_transMat(0, 0), ADD_VALUES);
      }                  
        }
        MoFEMFunctionReturn(0);
    }

    private:
        MatrixDouble loc_mat, loc_transMat;
        FTensor::Ddg<double, DIM, DIM> tD;
        boost::shared_ptr<CommonData> commonData;
};

template <int DIM> 
struct OpElasticRHS : OpEle {
    OpElasticRHS(boost::shared_ptr<CommonData> &data)
    : OpEle("U", OpEle::OPROW), commonData(data)
    {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data){
        MoFEMFunctionBegin;
        const int nb_dofs = data.getIndices().size();
        if (nb_dofs) {
            vecF.resize(nb_dofs, false);
            vecF.clear();
            const int nb_integration_pts = getGaussPts().size2(); 

            auto t_grad_U = getFTensor2FromMat<DIM, DIM>(commData->FMat);

            auto t_base = data.getFTensor0N();
            auto t_diff_base = data.getFTensor1DiffN<DIM>();
            auto t_w = getFTensor0IntegrationWeight();

            FTensor::Index<'i', DIM> i;
            FTensor::Index<'j', DIM> j;
            FTensor::Index<'k', DIM> k;  

            const double vol = getMeasure();
            for (int gg = 0; gg != nb_integration_pts; ++gg) { 

                const double a = vol * t_w;
                for (int rr = 0; rr != nb_dofs; ++rr) {
                    
                    ++t_diff_base;
                }
                ++t_grad_U;
                ++t_w;
            } 
            CHKERR VecSetOption(getFEMethod()->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
                            PETSC_TRUE);
            CHKERR VecSetValues(getFEMethod()->ts_F, data, &*vecF.begin(),
                            ADD_VALUES);      
        }
        MoFEMFunctionReturn(0);
    }

private:
  boost::shared_ptr<CommonData> commonData;
  VectorDouble vecF;

}



#endif // __ELASTIC_ELEMENT_HPP