/**
 * \file reaction_diffusion_equation.cpp
 * \example reaction_diffusion_equation.cpp
 *
 * \tableofcontents
 * 
 * \section reaction_diffusion_imp Implementation of reaction-diffusion equation
 * 
 *
 * Calculate level set for initally given surface
 */

#include <BasicFiniteElements.hpp>
using namespace MoFEM;
#include <typeinfo>

static char help[] = "...\n\n";

namespace ReactionDiffusionEquation {

using FaceEle = FaceElementForcesAndSourcesCore;
using OpFaceEle = FaceEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

const double Du = 2e-3; ///< diffusivity
const double Dun = 0;   ///< non-linear diffusivity

const double Dv = 0; ///< diffusivity
const double Dvn = 0;   ///< non-linear flux

const double au1 = 1;
const double au2 = 0;

const double av1 = 0;
const double av2 = 0;

const double ru = 1;
const double rv = 0;



const double r = 1;    ///< rate factor
const double k = 1;    ///< carrying capacity

const double u0 = 0.1; ///< inital vale on blocksets

const int order = 1; ///< approximation order
const int save_every_nth_step = 1;

struct Param
{
  double a1, a2;
  double r;
  double D, Dn;
};




/**
 * @brief Common data
 *
 * Common data are used to keep and pass data between elements
 *
 */
struct CommonData {

  MatrixDouble u_grad, v_grad;
  VectorDouble u_val, v_val;
  VectorDouble u_dot_val, v_dot_val;
  MatrixDouble u_invJac, v_invJac;

  // SmartPetscObj<Mat> M;
  // SmartPetscObj<KSP> ksp;
};

/**
 * @brief Assemble mass matrix
 */
struct OpAssembleMass : OpFaceEle {

    OpAssembleMass(const std::string                   &fieldu, 
                   const std::string                   &fieldv, 
                   const SmartPetscObj<Mat>             M);
  
    MoFEMErrorCode doWork(  int          row_side, 
                            int          col_side, 
                            EntityType   row_type,
                            EntityType   col_type, 
                            EntData &    row_data,
                            EntData &    col_data  ); 

  private:
    MatrixDouble                     mat;  
    MatrixDouble                     transMat;
    SmartPetscObj<Mat> M;
};

//defining the constructor
OpAssembleMass::OpAssembleMass(const std::string               &fieldu, 
                               const std::string               &fieldv, 
                               const SmartPetscObj<Mat>         m)
: OpFaceEle(fieldu, fieldv, OpFaceEle::OPROWCOL)
, M(m)
{
  sYmm = true; // determining the symmetry of the matrix
}

// definition of the main routine doWork()
MoFEMErrorCode OpAssembleMass::doWork(int             row_side, 
                                      int             col_side, 
                                      EntityType      row_type,
                                      EntityType      col_type, 
                                      EntData &       row_data,
                                      EntData &       col_data) 
{
    MoFEMFunctionBegin; //Every mofem function

    const int nb_row_dofs = row_data.getIndices().size(); // number of rows per entity
    const int nb_col_dofs = col_data.getIndices().size(); // number of cols per entity

    if (nb_row_dofs && nb_col_dofs)  
    {
      const int nb_integration_pts = getGaussPts().size2(); // Number of Gauss points
      mat.resize(nb_row_dofs, nb_col_dofs, false);          // resizing mat to the correct size
      mat.clear();                                          // clear mat
      auto t_row_base = row_data.getFTensor0N();            // get the first shape function at 
                                                            // at the first quad point
      //std::cout << typeid(t_row_base).name() << std::endl;

      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();
      for (int gg = 0; gg != nb_integration_pts; ++gg)
      {
        const double a = t_w * vol;
        for (int rr = 0; rr != nb_row_dofs; ++rr) 
        {
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) 
          {
            mat(rr, cc) += a * t_row_base * t_col_base;
            ++t_col_base;
          }
          ++t_row_base;
        }
        ++t_w;
      }

      // CHKERR MatSetValues(commonData->M, row_data, col_data, &mat(0, 0),
      //                     ADD_VALUES);
      CHKERR MatSetValues(M, row_data, col_data, &mat(0, 0),
                          ADD_VALUES);
      if (row_side != col_side || row_type != col_type) 
      {
        transMat.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transMat) = trans(mat);
        // CHKERR MatSetValues(commonData->M, col_data, row_data, &transMat(0, 0),
        //                     ADD_VALUES);
        CHKERR MatSetValues(M, col_data, row_data, &transMat(0, 0),
                            ADD_VALUES);
      }
    }
    MoFEMFunctionReturn(0);
  }



/**
 * @brief Assemble slow part
 *
 * Solve problem \f$ F(t,u,\dot{u}) = G(t,u) \f$ where here the right hand side
 * \f$ G(t,u) \f$ is implemented.
 *
 */
struct OpAssembleSlowRhs : OpFaceEle 
{
  OpAssembleSlowRhs(const std::string                    &fieldu, 
                    boost::shared_ptr<CommonData>        &data, 
                    const Param                          &pr)
  : OpFaceEle(fieldu, OpFaceEle::OPROW)
  , commonData(data)
  , param(pr) 
  , field(fieldu)
  {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) 
  {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) 
    {
      vecF.resize(nb_dofs, false);
      vecF.clear();

      const int nb_integration_pts = getGaussPts().size2();

      auto t_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      if(field == "u"){
         auto t_val_i = getFTensor0FromVec(commonData->u_val);
         auto t_val_j = getFTensor0FromVec(commonData->v_val); 
      
        for (int gg = 0; gg != nb_integration_pts; ++gg) 
        {
          const double a = vol * t_w;
          const double f = a * r * t_val_i * (1 -   t_val_i / k- param.a2 * t_val_j);
          for (int rr = 0; rr != nb_dofs; ++rr) 
          {
            const double b = f * t_base;
            vecF[rr] += b;
            ++t_base;
          }
          ++t_val_i;
          ++t_val_j;
          ++t_w;
        } 
      } 
      if(field == "v"){
         auto t_val_i = getFTensor0FromVec(commonData->v_val);
         auto t_val_j = getFTensor0FromVec(commonData->u_val);

        for (int gg = 0; gg != nb_integration_pts; ++gg) 
        {
          const double a = vol * t_w;
          const double f = a * r * t_val_i * (1 -   t_val_i / k- param.a2 * t_val_j);
          for (int rr = 0; rr != nb_dofs; ++rr) 
          {
            const double b = f * t_base;
            vecF[rr] += b;
            ++t_base;
          }
          ++t_val_i;
          ++t_val_j;
          ++t_w;
        }
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
  Param param;
  std::string field;
};

/**
 * @brief Assemble stiff part
 *
 * Solve problem \f$ F(t,u,\dot{u}) = G(t,u) \f$ where here the left hand side
 * \f$ F(t,u,\dot{u}) \f$ is implemented.
 *
 */
template <int DIM> 
struct OpAssembleStiffRhs : OpFaceEle
 {
    OpAssembleStiffRhs(const std::string                   &fieldu, 
                       boost::shared_ptr<CommonData>       &data, 
                       const Param                         &prm)
    : OpFaceEle(fieldu, OpFaceEle::OPROW)
    , commonData(data)
    , param(prm)
    , field(fieldu) 
    {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) 
    {
      MoFEMFunctionBegin;
      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) 
      {
        vecF.resize(nb_dofs, false);
        vecF.clear();

        const int nb_integration_pts = getGaussPts().size2();
        auto t_base = data.getFTensor0N();
        auto t_diff_base = data.getFTensor1DiffN<DIM>();
        auto t_w = getFTensor0IntegrationWeight();

        FTensor::Index<'i', DIM> i;

        const double vol = getMeasure();



        if(field == "u")
        {
           auto t_dot_val = getFTensor0FromVec(commonData->u_dot_val);
           auto t_grad = getFTensor1FromMat<DIM>(commonData->u_grad);
           auto t_val = getFTensor0FromVec(commonData->u_val);

          for (int gg = 0; gg != nb_integration_pts; ++gg) 
          {
            const double a = vol * t_w;
            for (int rr = 0; rr != nb_dofs; ++rr) 
            {
              vecF[rr] += a * (t_base * t_dot_val + (Du + Dun * t_val) * t_diff_base(i) * t_grad(i));
              ++t_diff_base;
              ++t_base;
            }
            ++t_dot_val;
            ++t_grad;
            ++t_val;
            ++t_w;
          }
        }
        if(field == "v")
        {
           auto t_dot_val = getFTensor0FromVec(commonData->v_dot_val);
           auto t_grad = getFTensor1FromMat<DIM>(commonData->v_grad);
           auto t_val = getFTensor0FromVec(commonData->v_val);

           for (int gg = 0; gg != nb_integration_pts; ++gg)  
           {
              const double a = vol * t_w;
              for (int rr = 0; rr != nb_dofs; ++rr) 
              {
                vecF[rr] += a * (t_base * t_dot_val + (Du + Dun * t_val) * t_diff_base(i) * t_grad(i));
                ++t_diff_base;
                ++t_base;
              }
              ++t_dot_val;
              ++t_grad;
              ++t_val;
              ++t_w;
           }
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
  Param param;
  std::string field;
};

/**
 * @brief Assemble stiff part tangent
 *
 * Solve problem \f$ F(t,u,\dot{u}) = G(t,u) \f$ where here the right hand side
 * \f$ \frac{\textrm{d} F}{\textrm{d} u^n} = a F_{\dot{u}}(t,u,\textrm{u}) +
 * F_{u}(t,u,\textrm{u}) \f$ is implemented.
 *
 */
template <int DIM> struct OpAssembleStiffLhs : OpFaceEle 
{
  OpAssembleStiffLhs(const std::string &fieldu, const std::string &fieldv, boost::shared_ptr<CommonData> &data, const Param &prm)
  : OpFaceEle(fieldu, fieldv, OpFaceEle::OPROWCOL)
  , commonData(data) 
  , param(prm)
  , field(fieldu)
  {
  sYmm = false;
  }
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
  MoFEMFunctionBegin;
  const int nb_row_dofs = row_data.getIndices().size();
  const int nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) 
  {
    mat.resize(nb_row_dofs, nb_col_dofs, false);
    mat.clear();

    const int nb_integration_pts = getGaussPts().size2();
    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<DIM>();
    auto t_w = getFTensor0IntegrationWeight();
    FTensor::Index<'i', DIM> i;

    const double ts_a = getFEMethod()->ts_a;
    const double vol = getMeasure();    

    if(field == "u"){
       auto t_grad = getFTensor1FromMat<DIM>(commonData->u_grad);
       auto t_val = getFTensor0FromVec(commonData->u_val);
        for (int gg = 0; gg != nb_integration_pts; ++gg) 
        {
          const double a = vol * t_w;
          for (int rr = 0; rr != nb_row_dofs; ++rr) 
          {

            auto t_col_base = col_data.getFTensor0N(gg, 0);
            auto t_col_diff_base = col_data.getFTensor1DiffN<DIM>(gg, 0);

            for (int cc = 0; cc != nb_col_dofs; ++cc) 
            {
              mat(rr, cc) += a * (t_row_base * t_col_base * ts_a +
                                  (Du + Dun * t_val) * t_row_diff_base(i) * t_col_diff_base(i)
                                  + Dun * t_col_base * t_row_diff_base(i) * t_grad(i));
              ++t_col_base;
              ++t_col_diff_base;
            }

            ++t_row_base;
            ++t_row_diff_base;
          }
          ++t_val;
          ++t_grad;
          ++t_w;
        }
    }
    if(field == "v"){
       auto t_grad = getFTensor1FromMat<DIM>(commonData->v_grad);
       auto t_val = getFTensor0FromVec(commonData->v_val);

        for (int gg = 0; gg != nb_integration_pts; ++gg) 
        {
          const double a = vol * t_w;
          for (int rr = 0; rr != nb_row_dofs; ++rr) 
          {

            auto t_col_base = col_data.getFTensor0N(gg, 0);
            auto t_col_diff_base = col_data.getFTensor1DiffN<DIM>(gg, 0);

            for (int cc = 0; cc != nb_col_dofs; ++cc) 
            {
              mat(rr, cc) += a * (t_row_base * t_col_base * ts_a +
                                  (Du + Dun * t_val) * t_row_diff_base(i) * t_col_diff_base(i)
                                  + Dun * t_col_base * t_row_diff_base(i) * t_grad(i));
              ++t_col_base;
              ++t_col_diff_base;
            }

            ++t_row_base;
            ++t_row_diff_base;
          }
          ++t_val;
          ++t_grad;
          ++t_w;
        }     
    }



      CHKERR MatSetValues(getFEMethod()->ts_B, row_data, col_data, &mat(0, 0),
                          ADD_VALUES);
      // if (row_side != col_side || row_type != col_type) 
      // {
      //   transMat.resize(nb_col_dofs, nb_row_dofs, false);
      //   noalias(transMat) = trans(mat);
      //   CHKERR MatSetValues(getFEMethod()->ts_B, col_data, row_data,
      //                       &transMat(0, 0), ADD_VALUES);
      // }
    }
    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonData;
  MatrixDouble mat, transMat;
  Param param;
  std::string   field;
};

/**
 * @brief Monitor solution
 *
 * This functions is called by TS solver at the end of each step. It is used
 * to output results to the hard drive.
 */
struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcFaceOnRefinedMesh> &post_proc)
  : dM(dm), postProc(post_proc)
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

}; // namespace ReactionDiffusionEquation

using namespace ReactionDiffusionEquation;

int main(int argc, char *argv[]) {

  // initialize petsc
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  try {

    Param param_u, param_v;

    param_u.a1 = au1;     param_u.a2 = au2;
    param_u.D  = Du;  param_u.Dn = Dun;

    param_v.a1 = av1;     param_v.a2 = av2;
    param_v.D  = Dv;  param_v.Dn = Dvn;

    param_u.r = ru;      param_v.r = rv;


    SmartPetscObj<Mat> l_M;
    SmartPetscObj<KSP> l_ksp;

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
    CHKERR simple_interface->addDomainField("u", H1, AINSWORTH_LEGENDRE_BASE, 1);
    CHKERR simple_interface->addDomainField("v", H1, AINSWORTH_LEGENDRE_BASE, 1);

    // set fields order
    CHKERR simple_interface->setFieldOrder("u", order);
    CHKERR simple_interface->setFieldOrder("v", order);
    
    
    // setup problem
    CHKERR simple_interface->setUp();

    // Create common data structure
    boost::shared_ptr<CommonData> data(new CommonData());
    /// Alias pointers to data in common data structure
    auto val_u_ptr = boost::shared_ptr<VectorDouble>(data, &data->u_val);
    auto dot_u_val_ptr = boost::shared_ptr<VectorDouble>(data, &data->u_dot_val);
    auto grad_u_ptr = boost::shared_ptr<MatrixDouble>(data, &data->u_grad);

    auto val_v_ptr = boost::shared_ptr<VectorDouble>(data, &data->v_val);
    auto dot_v_val_ptr = boost::shared_ptr<VectorDouble>(data, &data->v_dot_val);
    auto grad_v_ptr = boost::shared_ptr<MatrixDouble>(data, &data->v_grad);

    // Create finite element instances to integrate the right-hand side of slow
    // and stiff vector, and the tangent left-hand side for stiff part.
    boost::shared_ptr<FaceEle> vol_ele_slow_rhs(new FaceEle(m_field));
    boost::shared_ptr<FaceEle> vol_ele_stiff_rhs(new FaceEle(m_field));
    boost::shared_ptr<FaceEle> vol_ele_stiff_lhs(new FaceEle(m_field));

    // Push operators to integrate the slow right-hand side vector
    vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("u", val_u_ptr));

    vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("v", val_v_ptr));

    vol_ele_slow_rhs->getOpPtrVector().push_back(new OpAssembleSlowRhs("u", data, param_u));
    vol_ele_slow_rhs->getOpPtrVector().push_back(new OpAssembleSlowRhs("v", data, param_v));

    // PETSc IMAX and Explicit solver demands that g = M^-1 G is provided. So
    // when the slow right-hand side vector (G) is assembled is solved for g
    // vector.
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
      // CHKERR KSPSolve(data_u->ksp, vol_ele_slow_rhs->ts_F,
      //                 vol_ele_slow_rhs->ts_F);
      
      CHKERR KSPSolve(l_ksp, vol_ele_slow_rhs->ts_F,
                      vol_ele_slow_rhs->ts_F);
      MoFEMFunctionReturn(0);
    };
    // Add hook to the element to calculate g.
    vol_ele_slow_rhs->postProcessHook = solve_for_g;

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
         new OpCalculateScalarFieldValues("u", val_u_ptr));
         
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
         new OpCalculateScalarFieldValues("v", val_v_ptr));

    // Add operators to calculate the stiff right-hand side
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->u_invJac)
                                                  );
   vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(data->u_invJac)
                                                  );
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->v_invJac)
                                                  );


    
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(data->v_invJac)
                                                  );


    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarValuesDot("u", dot_u_val_ptr)
                                                  );
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarValuesDot("v", dot_v_val_ptr)
                                                  );

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("u", grad_u_ptr)
                                                  );
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("v", grad_v_ptr)
                                                  );


    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpAssembleStiffRhs<2>("u", data, param_u)
                                                  );
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpAssembleStiffRhs<2>("v", data, param_v)
                                                  );

    // Add operators to calculate the stiff left-hand side



    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->u_invJac)
                                                  );
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->v_invJac)
                                                  );


    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(data->u_invJac)
                                                  );
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(data->v_invJac)
                                                  );

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("u", grad_u_ptr)
                                                  );
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("v", grad_v_ptr)
                                                  );


    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("u", val_u_ptr));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("v", val_v_ptr));


    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpAssembleStiffLhs<2>("u", "u", data, param_u)
                                                  );
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpAssembleStiffLhs<2>("v", "v", data, param_v)
                                                  );


    // Set integration rules
    auto vol_rule = [](int, int, int p) -> int { return 2 * p; };
    vol_ele_slow_rhs->getRuleHook = vol_rule;
    vol_ele_stiff_rhs->getRuleHook = vol_rule;
    vol_ele_stiff_lhs->getRuleHook = vol_rule;

    // Crate element for post-processing
    boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc =
        boost::shared_ptr<PostProcFaceOnRefinedMesh>(
            new PostProcFaceOnRefinedMesh(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    // Genarte post-processing mesh
    post_proc->generateReferenceElementMesh();
    // Postprocess only field values
    post_proc->addFieldValuesPostProc("u");
    post_proc->addFieldValuesPostProc("v");


    // Get PETSc discrete manager
    auto dm = simple_interface->getDM();

    // Get surface entities form blockset, set initial values in those
    // blocksets. To keep it simple is assumed that inital values are on
    // blockset 1
    if (m_field.getInterface<MeshsetsManager>()->checkMeshset(1, BLOCKSET)) {
      Range inner_surface;
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          1, BLOCKSET, 2, inner_surface, true);
      if (!inner_surface.empty()) {
        Range inner_surface_verts;
        CHKERR moab.get_connectivity(inner_surface, inner_surface_verts, false);
        CHKERR m_field.getInterface<FieldBlas>()->setField(
            u0, MBVERTEX, inner_surface_verts, "u");
      }
    }

    // Get skin on the body, i.e. body boundary, and apply homogenous Dirichlet
    // conditions on that boundary.

    Range surface;
    CHKERR moab.get_entities_by_type(0, MBTRI, surface, false);
    Skinner skin(&m_field.get_moab());
    Range edges;
    CHKERR skin.find_skin(0, surface, false, edges);
    Range edges_part;
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    CHKERR pcomm->filter_pstatus(edges, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                 PSTATUS_NOT, -1, &edges_part);
    Range edges_verts;
    CHKERR moab.get_connectivity(edges_part, edges_verts, false);
    // Since Dirichlet b.c. are essential boundary conditions, remove DOFs from
    // the problem.
    CHKERR m_field.getInterface<ProblemsManager>()->removeDofsOnEntities(
        simple_interface->getProblemName(), "u",
        unite(edges_verts, edges_part));

    // // Create mass matrix, calculate and assemble
    // // CHKERR DMCreateMatrix_MoFEM(dm, data_u->M);
    // // CHKERR MatZeroEntries(data_u->M);
    // CHKERR DMCreateMatrix_MoFEM(dm, l_M);
    // CHKERR MatZeroEntries(l_M);

    // CHKERR DMCreateMatrix_MoFEM(dm, data_v->M);
    // CHKERR MatZeroEntries(data_v->M);
    CHKERR DMCreateMatrix_MoFEM(dm, l_M);
    CHKERR MatZeroEntries(l_M);


    boost::shared_ptr<FaceEle> vol_mass_ele(new FaceEle(m_field));
    vol_mass_ele->getOpPtrVector().push_back(new OpAssembleMass("u", "u", l_M));
    vol_mass_ele->getOpPtrVector().push_back(new OpAssembleMass("v", "v", l_M));

    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                    vol_mass_ele);
//     CHKERR MatAssemblyBegin(data_u->M, MAT_FINAL_ASSEMBLY);
//     CHKERR MatAssemblyEnd(data_u->M, MAT_FINAL_ASSEMBLY);

// CHKERR MatAssemblyBegin(data_v->M, MAT_FINAL_ASSEMBLY);
//     CHKERR MatAssemblyEnd(data_v->M, MAT_FINAL_ASSEMBLY);

  CHKERR MatAssemblyBegin(l_M, MAT_FINAL_ASSEMBLY);
  CHKERR MatAssemblyEnd(l_M, MAT_FINAL_ASSEMBLY);

  // CHKERR MatAssemblyBegin(l_M, MAT_FINAL_ASSEMBLY);
  // CHKERR MatAssemblyEnd(l_M, MAT_FINAL_ASSEMBLY);


    // Create and septup KSP (linear solver), we need this to calculate g(t,u) =
    // M^-1G(t,u)
    // data_u->ksp = createKSP(m_field.get_comm());
    // data_v->ksp = createKSP(m_field.get_comm());
    l_ksp = createKSP(m_field.get_comm());

    // CHKERR KSPSetOperators(data_u->ksp, data_u->M, data_u->M);
    // CHKERR KSPSetOperators(data_v->ksp, data_v->M, data_v->M);
    CHKERR KSPSetOperators(l_ksp, l_M, l_M);


    // CHKERR KSPSetFromOptions(data_u->ksp);
    // CHKERR KSPSetUp(data_u->ksp);
    CHKERR KSPSetFromOptions(l_ksp);
    CHKERR KSPSetUp(l_ksp);

    // CHKERR KSPSetFromOptions(data_v->ksp);
    // CHKERR KSPSetUp(data_v->ksp);
    CHKERR KSPSetFromOptions(l_ksp);
    CHKERR KSPSetUp(l_ksp);

    // Create and setup TS solver
    auto ts = createTS(m_field.get_comm());
    // Use IMEX solver, i.e. implicit/explicit solver
    CHKERR TSSetType(ts, TSARKIMEX);
    CHKERR TSARKIMEXSetType(ts, TSARKIMEXA2);

    // Add element to calculate lhs of stiff part
    CHKERR DMMoFEMTSSetIJacobian(dm, simple_interface->getDomainFEName(),
                                 vol_ele_stiff_lhs, null, null);
    // Add element to calculate rhs of stiff part
    CHKERR DMMoFEMTSSetIFunction(dm, simple_interface->getDomainFEName(),
                                 vol_ele_stiff_rhs, null, null);
    // Add element to calculate rhs of slow (nonlinear) part
    CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getDomainFEName(),
                                   vol_ele_slow_rhs, null, null);

    // Add monitor to time solver
    boost::shared_ptr<Monitor> monitor_ptr(new Monitor(dm, post_proc));
    CHKERR DMMoFEMTSSetMonitor(dm, ts, simple_interface->getDomainFEName(),
                               monitor_ptr, null, null);

    // Create solution vector
    SmartPetscObj<Vec> X;
    CHKERR DMCreateGlobalVector_MoFEM(dm, X);
    CHKERR DMoFEMMeshToLocalVector(dm, X, INSERT_VALUES, SCATTER_FORWARD);

    // Solve problem
    double ftime = 1;
    CHKERR TSSetDM(ts, dm);
    CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
    CHKERR TSSetSolution(ts, X);
    CHKERR TSSetFromOptions(ts);
    CHKERR TSSolve(ts, X);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
