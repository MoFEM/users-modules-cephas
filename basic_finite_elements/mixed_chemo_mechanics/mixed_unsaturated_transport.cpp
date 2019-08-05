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


#include <BasicFiniteElements.hpp>

namespace SoftTissue
{
    using namespace MoFEM;
    using Ele = FaceElementForcesAndSourcesCore;
    using OpEle = FaceElementForcesAndSourcesCore::UserDataOperator;

    using BoundaryEle = EdgeElementForcesAndSourcesCore;
    using OpBoundaryEle = EdgeElementForcesAndSourcesCore::UserDataOperator;

    using EntData = DataForcesAndSourcesCore::EntData; 

    const double D = 2e-3; ///< diffusivity
    const double Dn = 1e-2; ///< non-linear diffusivity
    const double u0 = 0.1; ///< inital vale on blocksets
    double a1, a2, r;
    const double au1 = 1;            const double au2 = 1.5;
    const double av1 = 1;            const double av2 = 1;
    const double ru = 1;             const double rv = 1; 
    const int save_every_nth_step = 4; 
    const int order = 1; ///< approximation order 

    struct CommonData 
    {
        MatrixDouble flux_values;    ///< flux values at integration points
        VectorDouble flux_divs;      ///< flux divergences at integration points
        VectorDouble mass_values;   ///< mass values at integration points
        VectorDouble mass_dots;      ///< mass rates at integration points


        MatrixDouble invJac; ///< Inverse of element jacobian at integration points

        CommonData() {}

        SmartPetscObj<Mat> M;   ///< Mass matrix
        SmartPetscObj<KSP> ksp; ///< Linear solver
    };

///////////////////////////////////////////////////////////////////////////////////
                        /////// Declarations //////////////   
///////////////////////////////////////////////////////////////////////////////////

//Assembly of system mass matrix //***********************************************                                 

    // Mass matrix corresponding to the flux equation. 
    // Note that it is an identity matrix
    struct OpAssembleMassFlux : OpEle 
    {
        OpAssembleMassFlux(std::string                    flux_field,
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
                              EntData              &col_data);

        private:
        MatrixDouble mat, transMat;
        SmartPetscObj<Mat> M;
    };
   
    // Mass matrix corresponding to the balance of mass equation where
    // there is a time derivative. 
    struct OpAssembleMassMass : OpEle 
    {
        OpAssembleMassMass(std::string                    mass_field,
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
                              EntData              &col_data);

        private:
        MatrixDouble mat, transMat;
        SmartPetscObj<Mat> M;
    };

//Assembly of RHS for explicit (slow) part//**************************************

    // RHS for explicit part of the flux equation excluding the essential boundary
    struct OpAssembleSlowRhsFlux : OpEle 
    {
        OpAssembleSlowRhsFlux(std::string                         flux_field,
                             boost::shared_ptr<CommonData>        &common_data,
                             Range                                &essential_bd_ents)
        : OpEle(flux_field, OpEle::OPROW), 
          commonData(common_data), 
          field(flux_field),
          essential_bd_ents(essential_bd_ents) 
          {}

        MoFEMErrorCode doWork(int             side, 
                              EntityType      type, 
                              EntData         &data);
        private:
            boost::shared_ptr<CommonData>      commonData;
            VectorDouble                       vecF;
            std::string                        field;
            Range                              &essential_bd_ents;
    };
    
    // RHS for explicit part of the mass balance equation
    struct OpAssembleSlowRhsMass : OpEle 
    {
        OpAssembleSlowRhsMass(std::string                         flux_field,
                              boost::shared_ptr<CommonData>       &common_data)
        : OpEle(field, OpEle::OPROW), 
          commonData(common_data), 
          field(flux_field) 
          {}

        MoFEMErrorCode doWork(int             side, 
                              EntityType      type, 
                              EntData         &data);
        private:
            boost::shared_ptr<CommonData>      commonData;
            VectorDouble                       vecF;
            std::string                        field;
    };
   

//Assembly of RHS for the implicit (stiff) part excluding the essential boundary //**********************************
    template<int dim> 
    struct OpAssembleStiffRhsFlux : OpEle 
    {
        OpAssembleStiffRhsFlux(std::string                    flux_field, 
                               boost::shared_ptr<CommonData>  &data,
                               Range                          &essential_bd_ents)
            : OpEle(flux_field, OpEle::OPROW), 
              commonData(data), 
              field(flux_field),
              essential_bd_ents(essential_bd_ents)
              {}
        
        MoFEMErrorCode doWork(int             side, 
                              EntityType      type, 
                              EntData         &data);

        private:
            boost::shared_ptr<CommonData> commonData;
            VectorDouble                  vecF;
            std::string                   flux_field;
            Range                         essential_bd_ents; 
            std::string                   field;
    };

    template<int dim> 
    struct OpAssembleStiffRhsMass : OpEle 
    {
        OpAssembleStiffRhsMass(std::string                    flux_field, 
                               boost::shared_ptr<CommonData>  &data)
        : OpEle(flux_field, OpEle::OPROW), 
          commonData(data), 
          flux_field(flux_field) 
          {}
        
        MoFEMErrorCode doWork(int        side, 
                              EntityType type, 
                              EntData    &data);

        private:
            boost::shared_ptr<CommonData> commonData;
            VectorDouble                  vecF;
            std::string                   flux_field;
    };

// RHS contribution of the natural boundary condition
   struct OpAssembleNaturalBCRhs : OpBoundaryEle 
    {
        OpAssembleNaturalBCRhs(std::string                         flux_field,
                               boost::shared_ptr<CommonData>       &common_data,
                               Range                               &natural_bd_ents)
        : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW), 
          commonData(common_data), 
          field(flux_field),
          natural_bd_ents(natural_bd_ents) 
          {}

        MoFEMErrorCode doWork(int             side, 
                              EntityType      type, 
                              EntData         &data);
        private:
            boost::shared_ptr<CommonData>      commonData;
            VectorDouble                       vecF;
            std::string                        field;
            Range                              natural_bd_ents;
    };

    // RHS contribution of the essential boundary condition
    struct OpAssembleEssentialBCRhs : OpBoundaryEle 
    {
        OpAssembleEssentialBCRhs(std::string                       flux_field,
                               boost::shared_ptr<CommonData>       &common_data,
                               Range                               &essential_bd_ents)
        : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW), 
          commonData(common_data), 
          field(flux_field),
          essential_bd_ents(essential_bd_ents) 
          {}

        MoFEMErrorCode doWork(int             side, 
                              EntityType      type, 
                              EntData         &data);
        private:
            boost::shared_ptr<CommonData>      commonData;
            VectorDouble                       vecF;
            std::string                        field;
            Range                              essential_bd_ents;
    };

// Declaration of thangent operator //**********************************************

    template <int dim>
    struct OpAssembleLhsFluxFlux : OpEle
    {
        OpAssembleLhsFluxFlux(std::string                   flux_field, 
                              Range                         &essential_bd_ents)
        : OpEle(flux_field, flux_field, OpEle::OPROWCOL), 
          field(flux_field),
          essential_bd_ents(essential_bd_ents)
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
            std::string                   field;
            Range                         essential_bd_ents;

    };

    template <int dim>
    struct OpAssembleLhsFluxFluxEBC : OpBoundaryEle
    {
        OpAssembleLhsFluxFluxEBC(std::string                   flux_field, 
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
                              EntData              &col_data);
        private: 
            MatrixDouble                  mat, transMat;
            std::string                   flux_field;
            Range                         essential_bd_ents;

    };

    template <int dim>
    struct OpAssembleLhsFluxMass : OpEle
    {
        OpAssembleLhsFluxMass(std::string                   flux_field, 
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

    template <int dim>
    struct OpAssembleLhsMassFlux : OpEle
    {
        OpAssembleLhsMassFlux(std::string                   mass_field,
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

    template <int dim>
    struct OpAssembleLhsMassMass : OpEle
    {
        OpAssembleLhsMassMass(std::string                   mass_field)
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

  cout << "Hello World!" << endl;
  return 0;
}