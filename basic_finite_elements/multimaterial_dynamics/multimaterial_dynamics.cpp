/** \file multimaterial_dynamics.cpp
 * \ingroup nonlinear_elastic_elem
 * \example multimaterial_dynamics.cpp
 *
 */
/* MoFEM is free software: you can redistribute it and/or modify it under
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

#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <ElasticMaterials.hpp>
#include <SurfacePressureComplexForLazy.hpp>

using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = FaceElementForcesAndSourcesCore;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcVolumeOnRefinedMesh;
using PostProcSkinEle = PostProcFaceOnRefinedMesh;

constexpr size_t SPACE_DIM = 3;

static char help[] = "...\n\n";

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;
using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<PETSC>::OpBase;

using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpSource<1, 3>;
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;

using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, SPACE_DIM>;
using OpBoundaryVec = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;
using OpBoundaryInternal = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;

#include <MDynamics.hpp>
using namespace MDynamicsFunctions;

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-ksp_atol 1e-10 \n"
                                 "-ksp_rtol 1e-10 \n"
                                 "-snes_monitor \n"
                                 "-snes_max_it 100 \n"
                                 "-snes_linesearch_type bt \n"
                                 "-snes_linesearch_max_it 3 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-8 \n"
                                 "-ts_monitor \n"
                                 "-ts_type alpha2 \n"
                                 "-ts_alpha_radius 1 \n"
                                 "-ts_monitor \n"
                                 "-mat_mumps_icntl_14 800 \n"
                                 "-mat_mumps_icntl_24 1 \n"
                                 "-mat_mumps_icntl_13 \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "DYNAMIC"));
  LogManager::setLog("DYNAMIC");
  MOFEM_LOG_TAG("DYNAMIC", "dynamic");

  try {

    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();
    CHKERR simple->loadFile("");
    simple->getProblemName() = "Multimaterial dynamics";

    MDynamics md(m_field);
    CHKERR md.setupProblem();
    CHKERR md.createCommonData();
    CHKERR md.bC();
    // CHKERR md.OPs();
    CHKERR md.tsSolve();

  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  return 0;
}