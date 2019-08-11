/**
 * \file reaction_diffusion_equation.cpp
 * \example reaction_diffusion_equation.cpp
 *
 **/
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


#include <stdlib.h>
#include <BasicFiniteElements.hpp>
using namespace MoFEM;
static char help[] = "...\n\n";

namespace SoftTissue
{
    using Ele = FaceElementForcesAndSourcesCore;
    using OpEle = FaceElementForcesAndSourcesCore::UserDataOperator;

    using BoundaryEle = EdgeElementForcesAndSourcesCore;
    using OpBoundaryEle = EdgeElementForcesAndSourcesCore::UserDataOperator;

    using EntData = DataForcesAndSourcesCore::EntData;

    const double B = 1e-2;
    const double B0 = 1e-3;
    const double B_epsilon = 0.1; 

    const double D = 2e-3; ///< diffusivity
    const double Dn = 1e-2; ///< non-linear diffusivity
    const double u0 = 0.1; ///< inital vale on blocksets
    double a1, a2, r;
    const double au1 = 1;            const double au2 = 1.5;
    const double av1 = 1;            const double av2 = 1;
    const double ru = 1;             const double rv = 1; 
    const int save_every_nth_step = 4; 
    const int order = 1; ///< approximation order 
    const double natural_bc_values = 0.0; 
    const double essential_bc_values = 0.0;
    // const int dim = 3;
    FTensor::Index<'i', 3> i;

    struct CommonData 
    {
        MatrixDouble flux_values;    ///< flux values at integration points
        VectorDouble flux_divs;      ///< flux divergences at integration points
        VectorDouble mass_values;   ///< mass values at integration points
        MatrixDouble mass_grads;
        VectorDouble mass_dots;      ///< mass rates at integration points



        MatrixDouble invJac; ///< Inverse of element jacobian at integration points

        CommonData() {}

        SmartPetscObj<Mat> M;   ///< Mass matrix
        SmartPetscObj<KSP> ksp; ///< Linear solver
    };

    struct CommonDataEssential
    {
        MatrixDouble flux_e_values;
    };

    struct CommonDataNatural
    {
        VectorDouble mass_n_values;    
    };

///////////////////////////////////////////////////////////////////////////////////
                        /////// Declarations //////////////   
///////////////////////////////////////////////////////////////////////////////////

//Assembly of system mass matrix //***********************************************                                 

    // Mass matrix corresponding to the flux equation. 
    //01. Note that it is an identity matrix
    struct OpAssembleMassTauTau : OpEle 
    {
        OpAssembleMassTauTau(std::string                    flux_field,
                             SmartPetscObj<Mat>             mass_matrix)
        : OpEle(flux_field, flux_field, OpEle::OPROWCOL), 
          M(mass_matrix)
        {
            sYmm = true;
        }

        MoFEMErrorCode doWork(int                  row_side, 
                              int                  col_side, 
                              EntityType           row_type,
                              EntityType           col_type, 
                              EntData              &row_data,
                              EntData              &col_data)
        {
            MoFEMFunctionBegin;
            const int nb_row_dofs = row_data.getIndices().size();
            const int nb_col_dofs = col_data.getIndices().size();
            if (nb_row_dofs && nb_col_dofs && row_type == col_type && 
                row_side == col_side) 
            {
                mat.resize(nb_row_dofs, nb_col_dofs, false);
                mat.clear();
                for (int rr = 0; rr != nb_row_dofs; ++rr) 
                {
                    mat(rr, rr) = 1.0;
                }
                CHKERR MatSetValues(M, row_data, col_data, &mat(0, 0),
                            INSERT_VALUES);
            }
            MoFEMFunctionReturn(0);
        } 

        private:
        MatrixDouble mat, transMat;
        SmartPetscObj<Mat> M;
    };
   
    // Mass matrix corresponding to the balance of mass equation where
    //02. there is a time derivative. 
    struct OpAssembleMassVV : OpEle 
    {
        OpAssembleMassVV(std::string                    mass_field,
                         SmartPetscObj<Mat>             mass_matrix)
        : OpEle(mass_field, mass_field, OpEle::OPROWCOL),
          M(mass_matrix)
        {
            sYmm = true;
        }
        MoFEMErrorCode doWork(int                  row_side, 
                              int                  col_side, 
                              EntityType           row_type,
                              EntityType           col_type, 
                              EntData              &row_data,
                              EntData              &col_data)
        {
            MoFEMFunctionBegin;
            const int nb_row_dofs = row_data.getIndices().size();
            const int nb_col_dofs = col_data.getIndices().size();
            if (nb_row_dofs && nb_col_dofs)
            {
                const int nb_integration_pts = getGaussPts().size2();
                mat.resize(nb_row_dofs, nb_col_dofs, false);
                mat.clear();
                auto t_row_base = row_data.getFTensor0N();
                auto t_w = getFTensor0IntegrationWeight();
                const double vol = getMeasure();
                for (int gg = 0; gg != nb_integration_pts; ++gg) {
                    const double a = t_w * vol;
                    for (int rr = 0; rr != nb_row_dofs; ++rr) {
                        auto t_col_base = col_data.getFTensor0N(gg, 0);
                        for (int cc = 0; cc != nb_col_dofs; ++cc) {
                            mat(rr, cc) += a * t_row_base * t_col_base;
                            ++t_col_base;
                        }
                        ++t_row_base;
                    }
                    ++t_w;
                }
                CHKERR MatSetValues(M, row_data, col_data, &mat(0, 0),
                            ADD_VALUES);
                if (row_side != col_side || row_type != col_type) {
                    transMat.resize(nb_col_dofs, nb_row_dofs, false);
                    noalias(transMat) = trans(mat);
                    CHKERR MatSetValues(M, col_data, row_data, &transMat(0, 0),
                                    ADD_VALUES);
                }
            }
            MoFEMFunctionReturn(0);        
        } 

        private:
        MatrixDouble mat, transMat;
        SmartPetscObj<Mat> M;
    };

//Assembly of RHS for explicit (slow) part//**************************************

    //1. RHS for explicit part of the flux equation excluding the essential boundary
    template <int dim>
    struct OpAssembleSlowRhsTau : OpEle 
    {
        OpAssembleSlowRhsTau(std::string                          flux_field,
                             boost::shared_ptr<CommonData>        &common_data,
                             Range                                &essential_bd_ents)
        : OpEle(flux_field, OpEle::OPROW), 
          commonData(common_data), 
          field(flux_field),
          essential_bd_ents(essential_bd_ents) 
          {}

        MoFEMErrorCode doWork(int             side, 
                              EntityType      type, 
                              EntData         &data)
        {
            MoFEMFunctionBegin;
            const int nb_dofs = data.getIndices().size();

            EntityHandle row_side_ent = data.getFieldDofs()[0]->getEnt();

            bool is_essential = (essential_bd_ents.find(row_side_ent)!=essential_bd_ents.end());
            if(is_essential)
                cerr << "rows are essential" << endl;
            if(nb_dofs && !is_essential){
                vecF.resize(nb_dofs, false);
                vecF.clear();
                const int nb_integration_pts = getGaussPts().size2();
                auto t_mass_grad = getFTensor1FromMat<dim>(commonData->mass_grads);
                auto t_mass_value = getFTensor0FromVec(commonData->mass_values);
                auto t_tau_base = data.getFTensor1N<dim>();
                auto t_w = getFTensor0IntegrationWeight();
                const double vol = getMeasure();
                for(int gg = 0; gg < nb_integration_pts; ++gg)
                {   
                    const double K = B_epsilon + (B0 + B * t_mass_value);
                    const double K_inv = 1. / K; 
                    const double a = vol * t_w;
                    for(int rr = 0; rr < nb_dofs; ++ ++rr)
                    {
                        vecF[rr] += -(B_epsilon * K_inv * t_tau_base(i) * t_mass_grad(i)) * a;
                        ++t_tau_base;
                    }
                    ++t_mass_grad;
                    ++t_mass_value;
                }     
            }
            MoFEMFunctionReturn(0);
        }
        private:
            boost::shared_ptr<CommonData>      commonData;
            VectorDouble                       vecF;
            std::string                        field;
            Range                              &essential_bd_ents;
    };
    
    //2. RHS for explicit part of the mass balance equation
    struct OpAssembleSlowRhsV : OpEle 
    {
        OpAssembleSlowRhsV(std::string                         flux_field,
                              boost::shared_ptr<CommonData>       &common_data)
        : OpEle(field, OpEle::OPROW), 
          commonData(common_data), 
          field(flux_field) 
          {}

        MoFEMErrorCode doWork(int             side, 
                              EntityType      type, 
                              EntData         &data)
        {
            MoFEMFunctionBegin;
            const int nb_dofs = data.getIndices().size();
            if (nb_dofs) {
                vecF.resize(nb_dofs, false);
                vecF.clear();
                const int nb_integration_pts = getGaussPts().size2();
                auto t_mass_value = getFTensor0FromVec(commonData->mass_values);
                auto t_base = data.getFTensor0N();
                auto t_w = getFTensor0IntegrationWeight();
                const double vol = getMeasure();
                for (int gg = 0; gg != nb_integration_pts; ++gg) {
                    const double a = vol * t_w;
                    const double f = a * r * t_mass_value * (1 - t_mass_value);
                    for (int rr = 0; rr != nb_dofs; ++rr) {
                        const double b = f * t_base;
                        vecF[rr] += b;
                        ++t_base;
                    }
                    ++t_mass_value;
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
            boost::shared_ptr<CommonData>      commonData;
            VectorDouble                       vecF;
            std::string                        field;
    };
   

//Assembly of RHS for the implicit (stiff) part excluding the essential boundary //**********************************
    //3. Assembly of F_tau excluding the essential boundary condition
    template<int dim> 
    struct OpAssembleStiffRhsTau : OpEle 
    {
        OpAssembleStiffRhsTau(std::string                    flux_field, 
                              boost::shared_ptr<CommonData>  &data,
                              Range                          &essential_bd_ents)
            : OpEle(flux_field, OpEle::OPROW), 
              commonData(data), 
              field(flux_field),
              essential_bd_ents(essential_bd_ents)
              {}
        
        MoFEMErrorCode doWork(int             side, 
                              EntityType      type, 
                              EntData         &data)
        {
            MoFEMFunctionBegin;
            EntityHandle row_side_ent = data.getFieldDofs()[0]->getEnt();

            bool is_essential = (essential_bd_ents.find(row_side_ent)!=essential_bd_ents.end());
            if(is_essential)
                cerr << "rows are essential" << endl;

            const int nb_dofs = data.getIndices().size();
            if(nb_dofs && !is_essential){
                vecF.resize(nb_dofs, false);
                vecF.clear();
                const int nb_integration_pts = getGaussPts().size2();
                auto t_flux_value = getFTensor1FromMat<3>(commonData->flux_values);
                auto t_mass_value = getFTensor0FromVec(commonData->mass_values);
                auto t_tau_base = data.getFTensor1N<3>();
                auto t_tau_grad = getFTensor2FromMat<3, 2>(data.getDiffN());
                auto t_w = getFTensor0IntegrationWeight();
                const double vol = getMeasure();
                double div_base = t_tau_grad(0, 0) + t_tau_grad(1, 1);
                for(int gg = 0; gg < nb_integration_pts; ++gg)
                {
                    const double K = B_epsilon + (B0 + B * t_mass_value);
                    const double K_inv = 1. / K;
                    const double a = vol * t_w;
                    for(int rr = 0; rr < nb_dofs; ++ ++rr)
                    { 
                        vecF[rr] += (K_inv * t_tau_base(i)*t_flux_value(i) -
                                    div_base * t_mass_value )
                                    * a;
                        ++t_tau_base;
                        ++t_tau_grad;
                    }
                    ++t_flux_value;
                    ++t_mass_value;
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
            VectorDouble                  vecF;
            Range                         essential_bd_ents; 
            std::string                   field;
    };
    //4. Assembly of F_v 
    template<int dim> 
    struct OpAssembleStiffRhsV : OpEle 
    {
        OpAssembleStiffRhsV(std::string                    flux_field, 
                               boost::shared_ptr<CommonData>  &data)
        : OpEle(flux_field, OpEle::OPROW), 
          commonData(data), 
          flux_field(flux_field) 
          {}
        
        MoFEMErrorCode doWork(int        side, 
                              EntityType type, 
                              EntData    &data)
        {
            MoFEMFunctionBegin;
            const int nb_dofs = data.getIndices().size();
            if(nb_dofs){
                vecF.resize(nb_dofs, false);
                vecF.clear();
                const int nb_integration_pts = getGaussPts().size2();
                auto t_mass_dot = getFTensor0FromVec(commonData->mass_dots);
                auto t_flux_div = getFTensor0FromVec(commonData->flux_divs);
                auto t_v_base = data.getFTensor0N();
                auto t_w = getFTensor0IntegrationWeight();
                const double vol = getMeasure();
                for(int gg = 0; gg < nb_integration_pts; ++gg)
                {
                const double a = vol * t_w;
                for(int rr = 0; rr < nb_dofs; ++ ++rr)
                    {
                        vecF[rr] += (- t_v_base * (t_mass_dot + t_flux_div) ) * a;
                        ++t_v_base;
                    }
                    ++t_mass_dot;
                    ++t_flux_div;
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
            VectorDouble                  vecF;
            std::string                   flux_field;
    };

   //5. RHS contribution of the natural boundary condition
   struct OpAssembleNaturalBCRhsTau : OpBoundaryEle 
    {
        OpAssembleNaturalBCRhsTau(std::string                         flux_field,
                                  boost::shared_ptr<CommonData>       &common_data,
                                  Range                               &natural_bd_ents)
        : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW), 
          commonData(common_data), 
          field(flux_field),
          natural_bd_ents(natural_bd_ents) 
          {}

        MoFEMErrorCode doWork(int             side, 
                              EntityType      type, 
                              EntData         &data)
        {
            MoFEMFunctionBegin;
            const int nb_dofs = data.getIndices().size();
            if(nb_dofs){
                vecF.resize(nb_dofs, false);
                vecF.clear();
                const int nb_integration_pts = getGaussPts().size2();
                auto t_tau_base = data.getFTensor1N<3>();

                auto dir = getDirection();
                double len = sqrt(pow(dir[0], 2) + pow(dir[1], 2));
                FTensor::Tensor1<double, 3> t_normal(-dir[1]/len, dir[0]/len, 0.);

                // auto t_normal = getFTensor1FromVec(getDirection());
                auto t_w = getFTensor0IntegrationWeight();
                const double vol = getMeasure();  
                for(int gg = 0; gg < nb_integration_pts; ++gg)
                {
                const double a = vol * t_w;
                for(int rr = 0; rr < nb_dofs; ++ ++rr)
                    {
                        vecF[rr] += (t_tau_base(i) * t_normal(i) * natural_bc_values) * a;
                        ++t_tau_base;
                    }
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
            boost::shared_ptr<CommonData>      commonData;
            VectorDouble                       vecF;
            std::string                        field;
            Range                              natural_bd_ents;
    };

    //6 RHS contribution of the essential boundary condition
    struct OpAssembleEssentialBCRhsTau : OpBoundaryEle 
    {
        OpAssembleEssentialBCRhsTau(std::string                         flux_field,
                                    boost::shared_ptr<CommonData>       &common_data,
                                    Range                               &essential_bd_ents)
        : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW), 
          commonData(common_data), 
          field(flux_field),
          essential_bd_ents(essential_bd_ents) 
          {}

        MoFEMErrorCode doWork(int             side, 
                              EntityType      type, 
                              EntData         &data)
        {
            MoFEMFunctionBegin;
            const int nb_dofs = data.getIndices().size();
            if(nb_dofs){
                vecF.resize(nb_dofs, false);
                vecF.clear();
                const int nb_integration_pts = getGaussPts().size2();
                auto t_flux_value = getFTensor1FromMat<3>(commonData->flux_values);
                auto t_tau_base = data.getFTensor1N<3>();

                auto dir = getDirection();
                double len = sqrt(pow(dir[0], 2) + pow(dir[1], 2));
                FTensor::Tensor1<double, 3> t_normal(-dir[1]/len, dir[0]/len, 0.);

                // auto t_normal = getFTensor1FromVec(getDirection());
                auto t_w = getFTensor0IntegrationWeight();
                const double vol = getMeasure();  
                for(int gg = 0; gg < nb_integration_pts; ++gg)
                {
                const double a = vol * t_w;
                for(int rr = 0; rr < nb_dofs; ++ ++rr)
                    {
                        vecF[rr] += (t_tau_base(i) * t_normal(i) *
                                     (t_flux_value(i) * t_normal(i) - 
                                                    essential_bc_values)) * a;
                        ++t_tau_base;
                    }
                    ++t_flux_value;
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
            boost::shared_ptr<CommonData>      commonData;
            VectorDouble                       vecF;
            std::string                        field;
            Range                              essential_bd_ents;
    };

// Declaration of thangent operator //**********************************************
    //7. Tangent assembly for F_tautau excluding the essential boundary condition
    template <int dim>
    struct OpAssembleLhsTauTau : OpEle
    {
        OpAssembleLhsTauTau(std::string                   flux_field, 
                            Range                         &essential_bd_ents,
                            boost::shared_ptr<CommonData>  &commonData)
        : OpEle(flux_field, flux_field, OpEle::OPROWCOL), 
          field(flux_field),
          essential_bd_ents(essential_bd_ents),
          commonData(commonData)
        {
            sYmm = true;
        }

        MoFEMErrorCode doWork(int                  row_side, 
                              int                  col_side, 
                              EntityType           row_type,
                              EntityType           col_type, 
                              EntData              &row_data,
                              EntData              &col_data)
        {
            MoFEMFunctionBegin;
            EntityHandle row_side_ent = row_data.getFieldDofs()[0]->getEnt();

            bool is_essential = (essential_bd_ents.find(row_side_ent)!=essential_bd_ents.end());
            if(is_essential)
                cerr << "rows are essential" << endl;
            const int nb_row_dofs = row_data.getIndices().size();
            const int nb_col_dofs = col_data.getIndices().size();
            if(nb_row_dofs && nb_col_dofs && !is_essential){
                mat.resize(nb_row_dofs, nb_col_dofs, false);
                mat.clear();
                const int nb_integration_pts = getGaussPts().size2();
                auto t_mass_value = getFTensor0FromVec(commonData->mass_values);

                auto t_row_tau_base = row_data.getFTensor1N<3>();

                auto t_w = getFTensor0IntegrationWeight();
                const double ts_a = getFEMethod()->ts_a;
                const double vol = getMeasure();

                for (int gg = 0; gg != nb_integration_pts; ++gg) {
                    const double a = vol * t_w;
                    const double K = B_epsilon + (B0 + B * t_mass_value);
                    const double K_inv = 1. / K;
                    for (int rr = 0; rr != nb_row_dofs; ++rr) {
                        auto t_col_tau_base = col_data.getFTensor1N<3>(gg, 0);
                        for (int cc = 0; cc != nb_col_dofs; ++cc) {
                            mat(rr, cc) = (K_inv * t_row_tau_base(i) * t_col_tau_base(i)) * a;

                            ++t_col_tau_base;
                        }
                        ++t_row_tau_base;
                    }
                    ++t_mass_value;
                    ++t_w;
                }
                CHKERR MatSetValues(getFEMethod()->ts_B, row_data, col_data, &mat(0, 0),
                    ADD_VALUES);
                if (row_side != col_side || row_type != col_type) {
                    transMat.resize(nb_col_dofs, nb_row_dofs, false);
                    noalias(transMat) = trans(mat);
                    CHKERR MatSetValues(getFEMethod()->ts_B, col_data, row_data,
                                        &transMat(0, 0), ADD_VALUES);
                }
            }            
            MoFEMFunctionReturn(0);
        }
        private: 
            boost::shared_ptr<CommonData> commonData;
            MatrixDouble                  mat, transMat;
            std::string                   field;
            Range                         essential_bd_ents;

    };

    //8. Assembly of tangent matrix for the essential boundary condition
    template <int dim>
    struct OpAssembleLhsEssentialBCTauTau : OpBoundaryEle
    {
        OpAssembleLhsEssentialBCTauTau(std::string                   flux_field, 
                                 Range                         &essential_bd_ents)
        : OpBoundaryEle(flux_field, flux_field, OpBoundaryEle::OPROWCOL), 
          flux_field(flux_field),
          essential_bd_ents(essential_bd_ents)
        {
            sYmm = true;
        }

        MoFEMErrorCode doWork(int                  row_side, 
                              int                  col_side, 
                              EntityType           row_type,
                              EntityType           col_type, 
                              EntData              &row_data,
                              EntData              &col_data)
        {
            MoFEMFunctionBegin;
            const int nb_row_dofs = row_data.getIndices().size();
            const int nb_col_dofs = col_data.getIndices().size();
            if (nb_row_dofs && nb_col_dofs){
                mat.resize(nb_row_dofs, nb_col_dofs, false);
                mat.clear();
                const int nb_integration_pts = getGaussPts().size2();
                auto t_w = getFTensor0IntegrationWeight();
                auto t_row_tau_base = row_data.getFTensor1N<3>();

                auto dir = getDirection();
                double len = sqrt(pow(dir[0], 2) + pow(dir[1], 2));
                FTensor::Tensor1<double, 3> t_normal(-dir[1]/len, dir[0]/len, 0.);

                const double ts_a = getFEMethod()->ts_a;
                const double vol = getMeasure();
                for (int gg = 0; gg != nb_integration_pts; ++gg) {
                    const double a = vol * t_w;
                    for (int rr = 0; rr != nb_row_dofs; ++rr) {
                        auto t_col_tau_base = col_data.getFTensor1N<3>(gg, 0);
                        for (int cc = 0; cc != nb_col_dofs; ++cc) {
                            mat(rr, cc) = ( (t_row_tau_base(i) * t_normal(i)) *
                                            (t_col_tau_base(i) * t_normal(i)) ) * a;

                            ++t_col_tau_base;
                        }
                        ++t_row_tau_base;
                    }
                    ++t_w;
                }
                CHKERR MatSetValues(getFEMethod()->ts_B, row_data, col_data, &mat(0, 0),
                          ADD_VALUES);
                if (row_side != col_side || row_type != col_type) {
                    transMat.resize(nb_col_dofs, nb_row_dofs, false);
                    noalias(transMat) = trans(mat);
                    CHKERR MatSetValues(getFEMethod()->ts_B, col_data, row_data,
                            &transMat(0, 0), ADD_VALUES);
                }
            }
            MoFEMFunctionReturn(0);
        }
        private: 
            MatrixDouble                  mat, transMat;
            std::string                   flux_field;
            Range                         essential_bd_ents;

    };
    
    //9. Assembly of tangent for F_tau_v excluding the essential bc
    template <int dim>
    struct OpAssembleLhsTauV : OpEle
    {
        OpAssembleLhsTauV(std::string                   flux_field, 
                              std::string                   mass_field,
                              boost::shared_ptr<CommonData> &data, 
                              Range                         &essential_bd_ents)
        : OpEle(flux_field, mass_field, OpEle::OPROWCOL), 
          commonData(data),
          flux_field(flux_field),
          mass_field(mass_field),
          essential_bd_ents(essential_bd_ents)
        {
            sYmm = false;
        }  


        MoFEMErrorCode doWork(int                  row_side, 
                              int                  col_side, 
                              EntityType           row_type,
                              EntityType           col_type, 
                              EntData              &row_data,
                              EntData              &col_data);
        private: 
            boost::shared_ptr<CommonData> commonData;
            MatrixDouble                  mat;
            std::string                   flux_field;
            std::string                   mass_field;
            Range                         essential_bd_ents;

    };
    
    //10. Assembly of tangent for F_v_tau
    template <int dim>
    struct OpAssembleLhsVTau : OpEle
    {
        OpAssembleLhsVTau(std::string                   mass_field,
                              std::string                   flux_field)
        : OpEle(mass_field, flux_field, OpEle::OPROWCOL), 
          flux_field(flux_field),
          mass_field(mass_field)
        {
            sYmm = false;
        }

        MoFEMErrorCode doWork(int                  row_side, 
                              int                  col_side, 
                              EntityType           row_type,
                              EntityType           col_type, 
                              EntData              &row_data,
                              EntData              &col_data);
        private: 
            MatrixDouble                  mat;
            std::string                   mass_field;
            std::string                   flux_field;

    };
    
    //11. Assembly of tangent for F_v_v
    template <int dim>
    struct OpAssembleLhsVV : OpEle
    {
        OpAssembleLhsVV(std::string                   mass_field)
        : OpEle(mass_field, mass_field, OpEle::OPROWCOL), 
          mass_field(mass_field)
        {
            sYmm = true;
        }

        MoFEMErrorCode doWork(int                  row_side, 
                              int                  col_side, 
                              EntityType           row_type,
                              EntityType           col_type, 
                              EntData              &row_data,
                              EntData              &col_data);
        private: 
            MatrixDouble                  mat, transMat;
            std::string                   mass_field;
            std::string                   flux_field;

    };

    struct Monitor : public FEMethod {
        Monitor(SmartPetscObj<DM>                            &dm,
                boost::shared_ptr<PostProcFaceOnRefinedMesh> &post_proc)
        : dM(dm), 
        postProc(post_proc)
        {};
        MoFEMErrorCode preProcess() { return 0; }
        MoFEMErrorCode operator()() { return 0; }
        MoFEMErrorCode postProcess() 
        {
            MoFEMFunctionBegin;
            if (ts_step % save_every_nth_step == 0) 
            {
                CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
                CHKERR postProc->writeFile(
                    "out_level_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
            }
            MoFEMFunctionReturn(0);
        }

    private:
        SmartPetscObj<DM> dM;
        boost::shared_ptr<PostProcFaceOnRefinedMesh> postProc;
    };

}; // namespace SoftTissue

using namespace SoftTissue;

int main(int argc, char *argv[]) 
{
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);
  try{
       SmartPetscObj<Mat> local_M;
       SmartPetscObj<KSP> local_Ksp;

       // Create moab and mofem instances
       moab::Core mb_instance;
       moab::Interface &moab = mb_instance;
       MoFEM::Core core(moab);
       MoFEM::Interface &m_field = core;
       // Register DM Manager
       DMType dm_name = "DMMOFEM";
       CHKERR DMRegister_MoFEM(dm_name);
       // Simple interface
       Simple *simple_interface;
       CHKERR m_field.getInterface(simple_interface);
       CHKERR simple_interface->getOptions();
       CHKERR simple_interface->loadFile();
       // add fields
       CHKERR simple_interface->addDomainField("MASS", L2, AINSWORTH_LEGENDRE_BASE, 1);

       CHKERR simple_interface->addDomainField("FLUX", HCURL, AINSWORTH_LEGENDRE_BASE, 1);

       CHKERR simple_interface->addBoundaryField("FLUX", HCURL, AINSWORTH_LEGENDRE_BASE, 1); // traces of Hcurl functions

       CHKERR simple_interface->setFieldOrder("Mass", order-1);
       CHKERR simple_interface->setFieldOrder("Flux", order);

       CHKERR simple_interface->setUp();

       boost::shared_ptr<CommonData> data(new CommonData());

       auto flux_values_ptr = boost::shared_ptr<MatrixDouble>(data, &data->flux_values);
       auto flux_divs_ptr = boost::shared_ptr<VectorDouble>(data, &data->flux_divs);
       auto mass_values_ptr = boost::shared_ptr<VectorDouble>(data, &data->mass_values);
       auto mass_grads_ptr = boost::shared_ptr<MatrixDouble>(data, &data->mass_grads);
       auto mass_dots_ptr = boost::shared_ptr<VectorDouble>(data, &data->mass_dots);

       boost::shared_ptr<Ele> vol_ele_slow_rhs(new Ele(m_field));
       boost::shared_ptr<Ele> vol_ele_stiff_rhs(new Ele(m_field));
       boost::shared_ptr<Ele> vol_ele_stiff_lhs(new Ele(m_field));  

       vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("MASS", mass_values_ptr));

       vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpAssembleSlowRhsV("Mass", data));

       auto solve_for_g = [&]() {
         MoFEMFunctionBegin;
         if (vol_ele_slow_rhs->vecAssembleSwitch) {
            CHKERR VecGhostUpdateBegin(vol_ele_slow_rhs->ts_F, ADD_VALUES,
                                SCATTER_REVERSE);
            CHKERR VecGhostUpdateEnd(vol_ele_slow_rhs->ts_F, ADD_VALUES,
                                SCATTER_REVERSE);
            CHKERR VecAssemblyBegin(vol_ele_slow_rhs->ts_F);
            CHKERR VecAssemblyEnd(vol_ele_slow_rhs->ts_F);
            *vol_ele_slow_rhs->vecAssembleSwitch = false;
          }
          CHKERR KSPSolve(local_Ksp, vol_ele_slow_rhs->ts_F,
                    vol_ele_slow_rhs->ts_F);
          MoFEMFunctionReturn(0);
        };

        vol_ele_slow_rhs->postProcessHook = solve_for_g;

       vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->invJac));
       vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(data->invJac));


  }
  CATCH_ERRORS;
  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();
  return 0;
}


      // AAAAAAAA
// EntityHandle fe_ent = getFEEntityHandle();
// if(essential_bd_ents.find(fe_ent)!=essential_bd_ents.end())
//     cerr << "element is on essenital boundary" << endl;

// EntityHandle row_side_ent = row_data.getFieldData()[0]->getEnt();
// EntityHandle col_side_ent = col_data.getFieldData()[0]->getEnt();
// if(essential_bd_ents.find(row_side_ent)!=essential_bd_ents.end())
//     cerr << "rows are essential" << endl;
// if(essential_bd_ents.find(col_side_ent)!=essential_bd_ents.end())
//     cerr << "cols are essential" << endl;




