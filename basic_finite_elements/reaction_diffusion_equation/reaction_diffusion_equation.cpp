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
using namespace MoFEM;

static char help[] = "...\n\n";

namespace ReactionDiffusionEquation {

using Ele = FaceElementForcesAndSourcesCore;
using OpEle = FaceElementForcesAndSourcesCore::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

const double D = 2e-3; ///< diffusivity
const double r = 1;    ///< rate factor
const double k = 1;    ///< caring capacity

const double u0 = 0.1; ///< inital vale on blocksets

const int order = 1; ///< approximation order
const int save_every_nth_step = 4;

/**
 * @brief Common data
 *
 * Common data are used to keep and pass data between elements
 *
 */
struct CommonData {

  MatrixDouble grad;    ///< Gradients of field "u" at integration points
  VectorDouble val;     ///< Values of field "u" at integration points
  VectorDouble dot_val; ///< Rate of values of field "u" at integration points
  MatrixDouble invJac;  ///< Inverse of element jacobian

  SmartPetscObj<Mat> M;   ///< Mass matrix
  SmartPetscObj<KSP> ksp; ///< Linear solver
};

/**
 * @brief Assemble mass matrix
 */
struct OpAssembleMass : OpEle {
  OpAssembleMass(boost::shared_ptr<CommonData> &data)
      : OpEle("u", "u", OpEle::OPROWCOL), commonData(data) {
    sYmm = true;
  }
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
    if (nb_row_dofs && nb_col_dofs) {
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

      CHKERR MatSetValues(commonData->M, row_data, col_data, &mat(0, 0),
                          ADD_VALUES);
      if (row_side != col_side || row_type != col_type) {
        transMat.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transMat) = trans(mat);
        CHKERR MatSetValues(commonData->M, col_data, row_data, &transMat(0, 0),
                            ADD_VALUES);
      }
    }
    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble mat, transMat;
  boost::shared_ptr<CommonData> commonData;
};

/**
 * @brief Assemble slow part
 *
 * Solve problem \f$ F(t,u,\dot{u}) = G(t,u) \f$ where here the right hand side
 * \f$ G(t,u) \f$ is implemented.
 *
 */
struct OpAssembleSlowRhs : OpEle {
  OpAssembleSlowRhs(boost::shared_ptr<CommonData> &data)
      : OpEle("u", OpEle::OPROW), commonData(data) {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      vecF.resize(nb_dofs, false);
      vecF.clear();

      const int nb_integration_pts = getGaussPts().size2();
      auto t_val = getFTensor0FromVec(commonData->val);
      auto t_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();

      const double vol = getMeasure();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        const double f = a * r * t_val * (1 - t_val / k);
        for (int rr = 0; rr != nb_dofs; ++rr) {
          const double b = f * t_base;
          vecF[rr] += b;
          ++t_base;
        }

        ++t_val;
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
};

/**
 * @brief Assemble stiff part
 *
 * Solve problem \f$ F(t,u,\dot{u}) = G(t,u) \f$ where here the right hand side
 * \f$ F(t,u,\dot{u}) \f$ is implemented.
 *
 */
template <int DIM> struct OpAssembleStiffRhs : OpEle {
  OpAssembleStiffRhs(boost::shared_ptr<CommonData> &data)
      : OpEle("u", OpEle::OPROW), commonData(data) {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      vecF.resize(nb_dofs, false);
      vecF.clear();

      const int nb_integration_pts = getGaussPts().size2();
      auto t_dot_val = getFTensor0FromVec(commonData->dot_val);
      auto t_grad = getFTensor1FromMat<DIM>(commonData->grad);
      auto t_base = data.getFTensor0N();
      auto t_diff_base = data.getFTensor1DiffN<DIM>();
      auto t_w = getFTensor0IntegrationWeight();
      FTensor::Index<'i', DIM> i;

      const double vol = getMeasure();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        for (int rr = 0; rr != nb_dofs; ++rr) {
          vecF[rr] += a * (t_base * t_dot_val + D * t_diff_base(i) * t_grad(i));
          ++t_diff_base;
          ++t_base;
        }
        ++t_dot_val;
        ++t_grad;
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
};

/**
 * @brief Assemble stiff part tangent
 *
 * Solve problem \f$ F(t,u,\dot{u}) = G(t,u) \f$ where here the right hand side
 * \f$ \frac{\textrm{d} F}{\textrm{d} u^n} = a F_{\dot{u}}(t,u,\textrm{u}) +
 * F_{u}(t,u,\textrm{u}) \f$ is implemented.
 *
 */
template <int DIM> struct OpAssembleStiffLhs : OpEle {
  OpAssembleStiffLhs(boost::shared_ptr<CommonData> &data)
      : OpEle("u", "u", OpEle::OPROWCOL), commonData(data) {
    sYmm = true;
  }
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
    if (nb_row_dofs && nb_col_dofs) {
      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();

      const int nb_integration_pts = getGaussPts().size2();
      auto t_row_base = row_data.getFTensor0N();
      auto t_row_diff_base = row_data.getFTensor1DiffN<DIM>();
      auto t_w = getFTensor0IntegrationWeight();
      FTensor::Index<'i', DIM> i;

      const double ts_a = getFEMethod()->ts_a;
      const double vol = getMeasure();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        for (int rr = 0; rr != nb_row_dofs; ++rr) {

          auto t_col_base = col_data.getFTensor0N(gg, 0);
          auto t_col_diff_base = col_data.getFTensor1DiffN<DIM>(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            mat(rr, cc) += a * (t_row_base * t_col_base * ts_a +
                                D * t_row_diff_base(i) * t_col_diff_base(i));
            ++t_col_base;
            ++t_col_diff_base;
          }

          ++t_row_base;
          ++t_row_diff_base;
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
  boost::shared_ptr<CommonData> commonData;
  MatrixDouble mat, transMat;
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
      : dM(dm), postProc(post_proc){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    if (ts_step % save_every_nth_step == 0) {
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
    // set fields order
    CHKERR simple_interface->setFieldOrder("u", order);
    // setup problem
    CHKERR simple_interface->setUp();

    // Create common data structure
    boost::shared_ptr<CommonData> data(new CommonData());
    /// Alias pointers to data in common data structure
    auto val_ptr = boost::shared_ptr<VectorDouble>(data, &data->val);
    auto dot_val_ptr = boost::shared_ptr<VectorDouble>(data, &data->dot_val);
    auto grad_ptr = boost::shared_ptr<MatrixDouble>(data, &data->grad);

    // Create finite element instances to integrate the right-hand side of slow
    // and stiff vector, and the tangent left-hand side for stiff part.
    boost::shared_ptr<Ele> vol_ele_slow_rhs(new Ele(m_field));
    boost::shared_ptr<Ele> vol_ele_stiff_rhs(new Ele(m_field));
    boost::shared_ptr<Ele> vol_ele_stiff_lhs(new Ele(m_field));

    // Push operators to integrate the slow right-hand side vector
    vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("u", val_ptr));
    vol_ele_slow_rhs->getOpPtrVector().push_back(new OpAssembleSlowRhs(data));

    // PETSc IMAX and Explicit solver demans that g = M^-1 G is provided. So
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
      CHKERR KSPSolve(data->ksp, vol_ele_slow_rhs->ts_F,
                      vol_ele_slow_rhs->ts_F);
      MoFEMFunctionReturn(0);
    };
    // Add hook to the element to calculate g.
    vol_ele_slow_rhs->postProcessHook = solve_for_g;

    // Add operators to calculate the stiff right-hand side
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->invJac));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(data->invJac));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarValuesDot("u", dot_val_ptr));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("u", grad_ptr));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpAssembleStiffRhs<2>(data));

    // Add operators to calculate the stiff left-hand side
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->invJac));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(data->invJac));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpAssembleStiffLhs<2>(data));

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
    CHKERR pcomm->filter_pstatus(edges, PSTATUS_SHARED | PSTATUS_MULTISHARED, PSTATUS_NOT, -1, &edges_part);
    Range edges_verts;
    CHKERR moab.get_connectivity(edges_part, edges_verts, false);
    // Since Dirichlet b.c. are essential boundary conditions, remove DOFs from
    // the problem.
    CHKERR m_field.getInterface<ProblemsManager>()->removeDofsOnEntities(
        simple_interface->getProblemName(), "u",
        unite(edges_verts, edges_part));

    // Create mass matrix, calculate and assemble
    CHKERR DMCreateMatrix_MoFEM(dm, data->M);
    CHKERR MatZeroEntries(data->M);
    boost::shared_ptr<Ele> vol_mass_ele(new Ele(m_field));
    vol_mass_ele->getOpPtrVector().push_back(new OpAssembleMass(data));
    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                    vol_mass_ele);
    CHKERR MatAssemblyBegin(data->M, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(data->M, MAT_FINAL_ASSEMBLY);

    // Create and septup KSP (linear solver), we need this to calculate g(t,u) =
    // M^-1G(t,u)
    data->ksp = createKSP(m_field.get_comm());
    CHKERR KSPSetOperators(data->ksp, data->M, data->M);
    CHKERR KSPSetFromOptions(data->ksp);
    CHKERR KSPSetUp(data->ksp);

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

/*! \page reaction_diffusion_imp Implementation of reaction-diffusion equation

\tableofcontents

\section reaction_diffusion_equation Fisher's reaction-diffusion equation

Fisher's equation \cite fisher1937wave, also know as
Kolmogorov–Petrovsky–Piskunov equation is
describing the population dynamics, e.g. mutation, in space and time.
Its predictive power can be applied to model fire propagation, virile mutant
propagation, the evolution of a neutron population in a nuclear reactor, or spread
of Alzheimer's disease in the brain, and many more phenomenon related to growth
and transformation of one infinite and uncountable set species into another.

In this tutorial, we consider equation almost in his original form presented
in \cite fisher1937wave
\f[
\frac{\partial u}{\partial t} - D \nabla^2 u =
ru\left(1-\frac{u}{k}\right)
\f]

where \f$D\f$ is diffusivity, \f$r\f$ is rate factor and \f$k\f$ is carrying
capacity. This equation can be understood that advancing wave with speed no
greater than \f$\sqrt{Dr}\f$, will transform abandon spices into another
"mutant" form, switching state variable \f$u\f$ from equilibrium \f$u=0\f$ state
to another equilibrium \f$u=k\f$. The estimated speed of the advancing wave can be
used further to choose the duration of analysis and length of the time step.

One can understand this equation by example, spreading fire on the plane of
dry grass, isolated from the rest of the domain by the trench of water. Fire
will spread over the whole area, where all grass over time will be consumed
by fire and transformed to ash, where viring (not burn) grass is indicated by
\f$u=0\f$, and grass consumed by fire, i.e. ash is \f$u=1\f$. The front of
the fire will move with the velocity not greater than \f$\sqrt{Dr}\f$. In the
case with the grass is mix with a plant with some resistance to fire, that
would be controlled by parameter \f$k\f$, i.e. caring capacity. Also, if there
is no wind or no inclination of the surface, the diffusivity parameter is
scalar. From the other hand, if the ground has some inclination, or wind is
present, the fire will have a tendency to go up the hill or direction of
the wind, that would be accounted for by tensorial representation of diffusivity.
If the grass has the same hight everywhere, \f$D\f$ is equal everywhere, i.e.
homogenous. If for example grass hight changing from position to position,
\f$D=D(\mathbf{x})\f$ and \f$r=r(\mathbf{x})\f$, diffusivity and rate factor
is heterogeneous. In this example, to keep the problem as simple as
possible, we implemented a homogenous case, with isotropic diffusivity.
Moreover, we assume that fire is initiated in two places, as shown in the
figure \ref Figure1 "Figure 1" below

\anchor Figure1
\image html reaction_diffusion_bc.png "Figure 1: Two yellow spots are places where the fire is initiated. On boundary is assumed Dirichlet boundary condition u = 0." width=500p

\section reaction_diffusion_running_code Running code

The code can work in parallel, so before the start, we have to do mesh partitioning,
such that form very beginning each processor store in the
memory only part of the mesh. In order to partition mesh execute the script
in build directory in \em basic_finite_elements/reaction_diffusion_equation
\code
NBPROC=6 && ../../tools/mofem_part \
-my_file mesh.cub -output_file mesh.h5m -my_nparts $NBPROC -dim 2 -adj_dim 1
\endcode
where variable \em NPROC represents a number of processers. The mesh file \em
mesh.cub is initial mesh from the master, and the partitioned mesh is saved to file
\em mesh.h5m. Having that mesh at hand we can solve the problem \code time
mpirun -np $NBPROC ./reaction_diffusion_equation 2>&1 | tee log \endcode where
parameters to run calculations are set from the file \em param_file.petsc

\include
users_modules/basic_finite_elements/reaction_diffusion_equation/param_file.petsc

where key parameters are
- \em -ts_final_time is time duration
- \em -ts_dt time step size
- \em -ts_arkimex_type ARIMEX time integration type
- \em -ts_adapt_type time step adaptation (if the stiff part is not updated it has to
be set to \em node)
- \em -snes_lag_jacobian set how often jacobian (stiff part) is updated. If set
to -2 jacobian is calculated only once at the beginning of the analysis

\anchor Figure2
\image html reaction_diffusion.gif "Figure 2: Solution of the problem." width=800p

\section reaction_diffusion_discretisation Discretisation

\subsection reaction_diffusion_weak_form Semi-discreate form of the equations

To solve the equation, we apply standard Galerkin method, used in finite
elements approximation, i.e. multiplying both sided by test function and
integrate by parts to reduce demand for the regularity of tested and testing
functions we get
\f[
\int_\Omega v \frac{\partial u}{\partial t} \textrm{d}\Omega +
\int_\Omega \nabla v \cdot D \nabla u \textrm{d}\Omega
=
\int_\Omega v r u \left(1-\frac{u}{k}\right) \textrm{d}\Omega
\f]
where \f$\Omega\f$ is solution domian. To have unique solution, we have to a
priori enforce essential boundary conditions, such as test functions \f$v\f$ and
tested function \f$u\f$ disapiear on boundary \f$\partial\Omega\f$.
Also, approximation and tested functions are in
Hilbert space \f$H^1_0(\Omega)\f$, such that integral of the first derivative and
function value over the domain is bounded. That will make the solution for our week
equation stable and equal to the solution of the strong equation, if smooth enough
initial and boundary conditions are provided. Moreover, the solution to the problem
can be approximated by a finite-dimensional and complete set of the pice-linear
confirming polynomials, which are dense in \f$H^1(\Omega)\f$, thus we will
have convergence to the exact solution with mesh refinement or increasing
polynomial approximation order. The approximation of test and tested functions
is given as follows \f[ v^h = \pmb\Phi
\overline{\mathbf{v}}^\textrm{T}\quad\textrm{and}\quad u^h = \pmb\Phi
\overline{\mathbf{u}}^\textrm{T} \f] where \f$\pmb\Phi\f$ is vector of
hierarchical base functions, and \f$\overline{\mathbf{v}}\f$,
\f$\overline{\mathbf{u}}\f$ are vectors of coefficients at degrees of freedom.
Utilising finite element approximation functions we finally get semi-discreate
form of the Fisher's equation \f[ \mathbf{M} \frac{\partial
\overline{\mathbf{u}}}{\partial t} + \mathbf{K} \overline{\mathbf{u}}
=
\mathbf{G}
\f]
where
\f[
\mathbf{M} :=
\int_\Omega \pmb\Phi^\textrm{T} \pmb\Phi\, \textrm{d}\Omega,\quad
\mathbf{K} :=
\int_\Omega \nabla \pmb\Phi^\textrm{T} \nabla \pmb\Phi \,
\textrm{d}\Omega\quad\textrm{and}\quad
\mathbf{G} := \int_\Omega
\pmb\Phi^\textrm{T} \left\{ r u^h \left(1-\frac{u^h}{k}\right) \right\}
\, \textrm{d}\Omega
\f]
where \f$\mathbf{M}\f$ is so called mass matrix, \f$\mathbf{K}\f$ stiffness
matrix and \f$\mathbf{G}\f$ is source vector.

\subsection reaction_diffusion_time Time discretisation

In this section we fully realay on the time stepping algorithms briefly
described in <a
href=https://www.mcs.anl.gov/petsc/petsc-current/docs/manual.pdf>in PETSc
documentation</a> in section 6. Moreover, we focus attention on IMEX method,
i.e. implicit-explicit time integration. The IMEX method is particularly useful
when two time scales are separate and present in the solution, i.e. one fast
scale, associated with \em stiff part of differential equation, and \em slow
scale usually associated with strongly nonlinear part of the equation. In our
particular case, \em stiff part of the equation is assisted with diffussion
process, and slow is associated with reaction part of the equation. Focussing
attention on our example of fire propagation, in plane of dry grass, \em stiff
part controls speed of advencing fire, and \em slow part is realted to the
length of the burning process itself, which a bit takes longer time. Using
formallism pressented in PETSc documentation we have

\f[ \mathbf{F} \left(
t,\overline{\mathbf{u}^n}, \dot{\overline{\mathbf{u}^n}}
\right)
=
\mathbf{G}
\left(
t,\overline{\mathbf{u}^n}
\right)
\f]
where
\f[
\left.
\frac{\textrm{d}\mathbf{F}}{\textrm{d}\overline{\mathbf{u}}}
\right|_{\overline{\mathbf{u}^n}}
=
\sigma \mathbf{F}_{\dot{\overline{\mathbf{u}^n}}}
\left(
t,\overline{\mathbf{u}^n}, \dot{\overline{\mathbf{u}^n}}
\right)
+
\mathbf{F}_{\overline{\mathbf{u}^n}}
\left(
t,\overline{\mathbf{u}^n}, \dot{\overline{\mathbf{u}^n}}
\right)
=
\sigma \mathbf{M} + \mathbf{K}
\f]
where
\f[
\sigma = \left. \frac{\partial \dot{u}}{\partial u} \right|_{u^n}
\f]
is paramter provided by algortim of time integration implemented in PETSc. The
key advantage of presented algorithm is that implementation of equation
is independent on time integration algorithm, and user can freely and quickly
change time integration method without need of changing a line of the code. The
change of time integration scheme is accounted by \f$\sigma\f$.

\section reaction_diffusion_implementation Implementation

Total control on solution process, that is calling functions to calculate
matrices and vectors in the right order is taken over by PETSc time solver
(TS). TS call MoFEM methods iterating over finite element entities, which are
provided to it by Discrete Manager (DM). The MoFEM keeps responsibility for
managing degrees of freedom, finite elements, and iterating over entities on
the mesh. MOAB is a mesh database where all data are stored on mesh tags at
any point of analysis.

MoFEM and MOAB databases provide low level functions, however in this case
flexibility of \ref MoFEM::Simple interface is adequate. \ref MoFEM::Simple
interface creates approximation fields and finite elements on whole mesh, and
create problem data structures accessed by DM. DM is a bridge between PETSc
and MoFEM data structures and functions, in this particular case TS.

The programmer responsibility is to initialise data structures, methods and
provide users operators called by finite elements to evaluate matrices and
vectors when called by time solver (TS). The relation between TS functions, DM
in MoFEM and finite elements are shown in \ref Figure3 "Figure 3".

\note Implementation of the problem is for PDE in two 2D, however with minimal
effort, by changing type of element can be changed to three dimensional.
Moreover is independent on time integration method, exploiting how PETSc time
solver is implemented.

\anchor Figure3
\image html reaction_diffusion_operators.png "Figure 3: Finite elements and operators. Yellow color indicate functions realted to Time Solver (TS). Read color indicated that functions are menaged by Discrete Manager (DM). Blue color indicate finite element instances. Green color indicate user data operators, where dark green standard user data operators, and lighter green user data operators implemented in this tutorial" width=800p

\subsection reaction_diffusion_mesh Setup problem

MoFEM always has to start with initialisation, and main function has format
\code
int main(int argc, char *argv[]) {
  // initialize petsc
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);
  try {

    // Implementation

  } 
  CATCH_ERRORS;
  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();
  return 0;
}
\endcode
Note that almost all MoFEM, PETSc and MOAB functions should be cheked by
CHKERR, that is essential to write safe code, and error detection.

In this tutorial initialisation of database has floowing steps

-# Main code starts with creating moab database and mofem database instances and
intefaces to each of them
\code
moab::Core mb_instance;
moab::Interface &moab = mb_instance;
MoFEM::Core core(moab);
MoFEM::Interface &m_field = core;
\endcode

-# Registring MoFEM Discrete Manager in PETSc
\code
DMType dm_name = "DMMOFEM";
CHKERR DMRegister_MoFEM(dm_name);
\endcode

-# Loading mesh, and adding fields and setup DM manager
\code
// Simple interface
Simple *simple_interface;
CHKERR m_field.getInterface(simple_interface);
CHKERR simple_interface->getOptions();
CHKERR simple_interface->loadFile();
// add fields
CHKERR simple_interface->addDomainField("u", H1, AINSWORTH_LEGENDRE_BASE,
                                        1);
// set fields order
CHKERR simple_interface->setFieldOrder("u", order);
// setup problem
CHKERR simple_interface->setUp();
\endcode
Note that here we add only one field \em u, in \f$H^1(\Omega)\f$ space,
approximation base is constructed following AINSWORTH_LEGENDRE_BASE recipe, and
field is scalar filed, i.e. number of coefficients for base function is 1.
Simple interface preasumes that field is span over whole domain, and integration
in domain will be over highest dimension entities on the mesh, in this
particular example all triangles. The finite element name is accessed through
\code std::string fe_name = simple_interface->getDomainFEName(); \endcode On
that element will be executed finite element instances, described below, and on
each element user push set of operators to execute. Finite elements instances
for other hand will be called through DM manager by TS solver.

-# Getting access to database from that point can be done exclusively through DM
\code
MoFEM::SmartPetscObj<DM> dm = simple_interface->getDM();
\endcode
Note that MoFEM::SmartPetscObj wraps PETSc object, can be used as regular PETSc
object but user is revived form the need of calling destructor for it. Having
access to DM one can push finite element instaces which can be used by TS to
calculate matrices and vectors at subseqent time steps.

\subsection reaction_diffusion_telling_ts Telling TS what elements should be used

In \ref Figure3 "Figure 3" in yellow boxes are PETSc functions, or yellow
boxes on red background functions beeing part of DM and interfacing Time
Solver (TS) with the MoFEM. In this tutorial TS is set to use IMEX
(implicit/explict) method, <a
href=https://www.mcs.anl.gov/petsc/petsc-current/docs/manual.pdf>see
details</a> in section 6. Methods calculating vector and matrices needed by TS
solver to proceed are set by DM functions

-# \ref MoFEM::DMMoFEMTSSetRHSFunction to calculate \f$\mathbf{G}\f$
-# \ref MoFEM::DMMoFEMTSSetIFunction to calculate \f$\mathbf{F}\f$
-# \ref MoFEM::DMMoFEMTSSetIJacobian to calculate \f$\left.
\frac{\textrm{d}\mathbf{F}}{\textrm{d}\overline{\mathbf{u}}}
\right|_{\overline{\mathbf{u}^n}}\f$
-# \ref MoFEM::DMMoFEMTSSetMonitor to set monitor run at the end of each load step

Those functions provide sequences of finite elements (or just one element for
each term in this particular case) to calculate entries of vectors and
matrices at particular time steps. 

-# Finite elements are marked by blue boxes on \ref Figure3 "Figure 3", and are
created by following code 

\code
// Create finite element instances to integrate the right-hand side of slow
// and stiff vector, and the tangent left-hand side for stiff part.
boost::shared_ptr<Ele> vol_ele_slow_rhs(new Ele(m_field));
boost::shared_ptr<Ele> vol_ele_stiff_rhs(new Ele(m_field));
boost::shared_ptr<Ele> vol_ele_stiff_lhs(new Ele(m_field));
\endcode

where integration rule, controlling number of integration points on each
elment are set by
\code
auto vol_rule = [](int, int, int p) -> int { return 2 * p; };
vol_ele_slow_rhs->getRuleHook = vol_rule;
vol_ele_stiff_rhs->getRuleHook = vol_rule;
vol_ele_stiff_lhs->getRuleHook = vol_rule;
\endcode
Integration rule depends on type of operator evaluated on element, in our
case we evaluated mass matrices, thus we need exactly calculate polynomial
order \f$2p\f$. 

Note that finite elements instances implementation is generic. Elements do
problem specific calculations by providing to them user data operators, what
is described in following sections. Three elements \em vol_ele_slow_rhs, \em
vol_ele_stiff_rhs, \em vol_ele_stiff_lhs are provided to Discrete Manager
(DM) to calculate slow and stiff vectors, and jacobian matrix. Those three
elements are instances of the class MoFEM::FaceElementForcesAndSourcesCore. 
Once elements are created, we can add them to TS manger through DM interface

\code
// Add element to calculate lhs of stiff part
CHKERR DMMoFEMTSSetIJacobian(dm, simple_interface->getDomainFEName(), vol_ele_stiff_lhs, null, null);
// Add element to calculate rhs of stiff part
CHKERR DMMoFEMTSSetIFunction(dm, simple_interface->getDomainFEName(), vol_ele_stiff_rhs, null, null);
// Add element to calculate rhs of slow (nonlinear) part
CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getDomainFEName(), vol_ele_slow_rhs, null, null);
\endcode

\note In case of 3D problem, user has to switch to
MoFEM::VolumeElementForcesAndSourcesCore, to integrate over volume entities.
Note that all users operators implemented in the example are templates, where
template variable is dimension of the element. That makes implementation
dimension independent.

\subsection reaction_diffusion_telling_fe Telling finite elements what operators should be used

The problem specific matrices and vectors are implemented in namespace
\ref ReactionDiffusionEquation, by users data operators \ref
ReactionDiffusionEquation::OpEle. In this tutorial following operators are
implemented

-# ReactionDiffusionEquation::OpAssembleMass used to calculate mass matrix
\f$\mathbf{M}\f$

-# ReactionDiffusionEquation::OpAssembleSlowRhs used to calculate the \em slow
right-hand side vector \f$\mathbf{G}\f$

-# ReactionDiffusionEquation::OpAssembleStiffRhs used to calculate the \em stiff
vector \f$\mathbf{F}\f$

-# ReactionDiffusionEquation::OpAssembleStiffLhs used to calculate the \em stiff
matrix \f$\frac{\textrm{d}\mathbf{F}}{\textrm{d} \overline{\mathbf{u}^n}}\f$

-# ReactionDiffusionEquation::Monitor used to postprocess results at the end of
each time step

Implementation of each operator follow similar pattern,

-# The class is inherithed form ReactionDiffusionEquation::OpEle

-# Two types of operators are implemented
ReactionDiffusionEquation::OpEle::OPROW, to calculate vectors, and
ReactionDiffusionEquation::OpEle::OPROWCOL to calculate matrices.

-# Implementation of user operator overload \em doWork member function

-# In each doWork method user iterate over integration points, and then over
base functions. In case of matrix over base functions on rows and columns.

-# Once local element vector or is assembled, \em doWork function is finalised
with assembling to global vector or global matrix.

\note More derail description how to implement user data operator you can find
in other tutorials, for example \ref poisson_tut2.

\subsection reaction_diffusion_common_data Common data

The data between operator and finite elements are passed thrugh
ReactionDiffusionEquation::CommonData
\code
struct CommonData {

  MatrixDouble grad;    ///< Gradients of field "u" at integration points
  VectorDouble val;     ///< Values of field "u" at integration points
  VectorDouble dot_val; ///< Rate of values of field "u" at integration points
  MatrixDouble invJac;  ///< Inverse of element jacobian

  SmartPetscObj<Mat> M;   ///< Mass matrix
  SmartPetscObj<KSP> ksp; ///< Linear solver

};
\endcode
which is dynamically allocated and keep by shared pointer
\code
boost::shared_ptr<CommonData> data(new CommonData());
\endcode
in addition some other operators need access directly to data, it can be safely
done by so called aliased shared pointers (<a
href=https://stackoverflow.com/questions/27109379/what-is-shared-ptrs-aliasing-constructor-for>see
here</a>) 
\code 
auto val_ptr = boost::shared_ptr<VectorDouble>(data, &data->val); 
auto dot_val_ptr = boost::shared_ptr<VectorDouble>(data, &data->dot_val); 
auto grad_ptr = boost::shared_ptr<MatrixDouble>(data, &data->grad); 
\endcode

\subsection reaction_diffusion_pushing_fe Pushing operators to finite element

Each of those user data operators are added to finite element, for example to
\em vol_ele_stiff_rh we add operators as following
\code
vol_ele_stiff_rhs->getOpPtrVector().push_back(new MoFEM::OpCalculateInvJacForFace(data->invJac));
vol_ele_stiff_rhs->getOpPtrVector().push_back(new MoFEM::OpSetInvJacH1ForFace(data->invJac));
vol_ele_stiff_rhs->getOpPtrVector().push_back(new MoFEM::OpCalculateScalarValuesDot("u", dot_val_ptr));
vol_ele_stiff_rhs->getOpPtrVector().push_back(new MoFEM::OpCalculateScalarFieldGradient<2>("u", grad_ptr));
vol_ele_stiff_rhs->getOpPtrVector().push_back(new ReactionDiffusionEquation::OpAssembleStiffRhs<2>(data));
\endcode
where MoFEM::OpCalculateInvJacForFace and MoFEM::OpSetInvJacH1ForFace are
standard operators to calculate element inverse of jacobian, and pull
direvatives of base functions to element reference configuration \f[
\frac{\partial \pmb\Phi}{\partial \mathbf{x}} = \frac{\partial
\pmb\Phi}{\partial \pmb\xi}\mathbf{J}^{-\textrm{T}} \quad\textrm{where}\quad
\mathbf{J} = \frac{\partial \mathbf{x}}{\partial \pmb\xi},\;
\mathbf{x} = \pmb\Phi(\pmb \xi) \overline{\mathbf{x}}
\f]
where \f$\pmb \xi\f$ are coordinates in local element configuration,
\f$\overline{\mathbf{x}}\f$ ale nodal positions or higher order coefficients in
edges and faces if necessary to describe higher order geometry. Operators
OpCalculateScalarFieldGradient and OpCalculateScalarFieldGradient are standard
operators and are used to calculate field values, and gradients of field
values at integration points, respectively. All operator in \ref Figure3 "Figure
3" marked by dark green color, indicate that are standard operators and are
implemented in basic user modules.

\subsection reaction_diffusion_g Problem with IMAX method in TS

The implementation of IMAX method in TS solver works requires the user to
provide 

\f[ \mathbf{g}(t,\overline{\mathbf{u}}) =
\mathbf{M}^{-1}\mathbf{G}(t,\overline{\mathbf{u}})
\f]

wheras user data operators provie vector
\f$\mathbf{G}(t,\overline{\mathbf{u}})\f$. This issue can be resolved by
exploiting finite elemnet functionality. Each element has derived from
MoFEM::BasicMethod

\code 
struct BasicMethod : public KspMethod, SnesMethod, TSMethod { 
  
  virtual MoFEMErrorCode preProcess() {
    if(preProcessHook)
      CHKERR preProcessHook()
  }
  virtual MoFEMErrorCode operator()();
  virtual MoFEMErrorCode postProcess() {
    if(preProcessHook)
      CHKERR postProcessHook()
  }

};
\endcode

Every element is run in three stages

-# \em preProcess() is executed before code iterate over all given entity finite
elements on the mesh

-# \em operator() is executed for each entity of finite element on the mesh

-# \em postProcess() is executed after code iterate over all given entity finite
elements on the mesh

If user provied hook, after vector \f$\mathbf{G}(t,\overline{\mathbf{u}})\f$ is
calculated, and implementation of following lambda function 

\code 
auto solve_for_g = [&]() { 
  MoFEMFunctionBegin; 
  if (vol_ele_slow_rhs->vecAssembleSwitch) { 
    CHKERR VecGhostUpdateBegin(vol_ele_slow_rhs->ts_F, ADD_VALUES, SCATTER_REVERSE); 
    CHKERR VecGhostUpdateEnd(vol_ele_slow_rhs->ts_F, ADD_VALUES, SCATTER_REVERSE); 
    CHKERR VecAssemblyBegin(vol_ele_slow_rhs->ts_F); 
    CHKERR VecAssemblyEnd(vol_ele_slow_rhs->ts_F); 
    *vol_ele_slow_rhs->vecAssembleSwitch = false;
  }
  CHKERR KSPSolve(data->ksp, vol_ele_slow_rhs->ts_F, vol_ele_slow_rhs->ts_F);
  MoFEMFunctionReturn(0);
};
\endcode
This hooke can be set to the \em slow finite element, i.e. \em vol_ele_slow_rhs as
follows 
\code 
vol_ele_slow_rhs->postProcessHook = solve_for_g; 
\endcode 

Note, that is also indicated on \ref Figure3 "Figure 3". In \em solve_for_g
the KSP (linear) solver is used to solve linear equation

\f[
\mathbf{M}\mathbf{g}(t,\overline{\mathbf{u}}) =
\mathbf{G}(t,\overline{\mathbf{u}})
\f]
where mass matrix \f$\mathbf{M}\f$ is calculated as follows

\code
CHKERR MatZeroEntries(data->M);
CHKERR DMCreateMatrix_MoFEM(dm, data->M);
boost::shared_ptr<Ele> vol_mass_ele(new Ele(m_field));
vol_mass_ele->getOpPtrVector().push_back(new
ReactionDiffusionEquation::OpAssembleMass(data)); CHKERR
DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(), vol_mass_ele);
CHKERR MatAssemblyBegin(data->M, MAT_FINAL_ASSEMBLY);
CHKERR MatAssemblyEnd(data->M, MAT_FINAL_ASSEMBLY);
\endcode

with the use on the fly created finite element \em vol_mass_ele with pushed
one operator ReactionDiffusionEquation::OpAssembleMass. The linear KSP solver
is created and setup as follows

\code 
data->ksp = createKSP(m_field.get_comm()); 
CHKERR KSPSetOperators(data->ksp, data->M, data->M); 
CHKERR KSPSetFromOptions(data->ksp); CHKERR KSPSetUp(data->ksp); 
\endcode

\section reaction_diffusion_initial Initail and boundary conditions

\subsection reaction_diffusion_initial Initial conditions

To set initial conditions we use MoFEM::MeshsetsManager to get entities on the blockset,
which are set by meshing code, for example Cubit, Salome or gMEsh. We assume
that entities on which we set boundary conditions are set on subset of entities,
which are listed in \em BLOCKSET 1. In the following code

\code
if (m_field.getInterface<MoFEM::MeshsetsManager>()->checkMeshset(1, BLOCKSET)) {
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
\endcode

-# We check if BLOCKSET 1 exist. Since problem is solved in parallel, each
processor keeps only part of the mesh, BLOCKSET 1 exist only on some processors,
which the right set of the entities

-# If block exist, we ask MoFEM::MeshsetsManager to give entities dimension 2,
i.e. faces, i.e. triangles in the block set

-# We gat all the vertices on faces, and using MoFEM::FieldBlas interface we set
values to the nodes on vertices. The value which is set is constsat \f$u_0\f$.

\subsection reaction_diffusion_bc Dirichlet boundary conditions

To keep inital assumption, given in the begining paragraphs about fire
propagation, that around plain of dry grass we have trench filled with water
which stop fire, we will apply on boundary homogenous Dirichlet boundary
conditions. In order to do that we follow code

\code
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
CHKERR m_field.getInterface<ProblemsManager>()->removeDofsOnEntities(
    simple_interface->getProblemName(), "u",
    unite(edges_verts, edges_part));
\endcode

-# Take all triangle on the mesh

-# Using MOAB implemented skinner, we take skin from the mesh. Consequently we
have all edges bounding mesh on part of the processor.

-# Since we work on parallel mesh, not all edges are on true domain boundary. In
order to fillter out true boundary we using moab::ParallelComm::filter_pstatus
which is provided with MOAB interface. Using moab::ParallelComm::filter_pstatus
we filter out all entities which are not shared with any other processor. That
entities are physical domain boundary.

-# In order to enforce homogenous essential boundary conditions in finite
element method, the simplest way is to remove deress of freedom (DOFs) from the
problem. That is done by MoFEM::ProblemsManager interface using
MoFEM::ProblemsManager::removeDofsOnEntities function. MoFEM::ProblemsManager is
a MoFEM interface to manage DOFs and finite elements on the problems.

\note MoFEM can manage many problems at once, for example one will have elastic
problem and thermal problem set independently when solve them in staggered
manner. Problems can share fields values, and finite elements, however keep sets
of numbered DOFs independently and have diffrent subsets of DOFs.

\section reaction_diffusion_code The plain program.

The plain program is located in
basic_finite_elements/reaction_diffusion_equation/reaction_diffusion_equation.cpp

\include reaction_diffusion_equation.cpp

*/
