/**
 * \file heat_method.cpp
 * \example heat_method.cpp
 *
 * Using PipelineManager interface calculate the divergence of base functions,
 * and integral of flux on the boundary. Since the h-div space is used, volume
 * integral and boundary integral should give the same result.
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

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

constexpr double tol = 1e-6;
constexpr double l2 = 5e-3;

#include <BasicFiniteElements.hpp>

using DomainEle = PipelineManager::FaceEle2D;
using DomainEleOp = DomainEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

using OpHdivHdiv = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<3, 3>;
using OpHdivU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixDivTimesScalar<3>;

using OpQQ = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<3, 3, 1>;
using OpDivQU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpMixDivTimesU<1, 3>;
using OpUDivQ = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesScalarField<1>;

using EdgeEle = PipelineManager::EdgeEle2D;
using EdgeEleOp = EdgeEle::UserDataOperator;

using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  MatrixDouble invJac;
  MatrixDouble jAC;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();

  struct OpRhs : public DomainEleOp {

    OpRhs(boost::shared_ptr<MatrixDouble> q_ptr,
          boost::shared_ptr<VectorDouble> div_ptr,
          boost::shared_ptr<VectorDouble> u_ptr,
          boost::shared_ptr<MatrixDouble> dot_q_ptr,
          boost::shared_ptr<VectorDouble> dot_div_ptr,
          boost::shared_ptr<VectorDouble> dot_u_ptr);

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

  protected:
    boost::shared_ptr<MatrixDouble> qPtr;
    boost::shared_ptr<VectorDouble> divPtr;
    boost::shared_ptr<VectorDouble> uPtr;
    boost::shared_ptr<MatrixDouble> dotQPtr;
    boost::shared_ptr<VectorDouble> dotDivPtr;
    boost::shared_ptr<VectorDouble> dotUPtr;
  };

  struct OpLhs : public DomainEleOp {

    OpLhs(boost::shared_ptr<MatrixDouble> q_ptr,
          boost::shared_ptr<VectorDouble> div_ptr,
          boost::shared_ptr<MatrixDouble> dot_q_ptr);

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

  protected:
    boost::shared_ptr<MatrixDouble> qPtr;
    boost::shared_ptr<VectorDouble> divPtr;
    boost::shared_ptr<MatrixDouble> dotQPtr;
  };
};

//! [Run programme]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR setIntegrationRules();
  CHKERR solveSystem();
  CHKERR outputResults();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  Range tris;
  CHKERR mField.get_moab().get_entities_by_type(0, MBTRI, tris);
  Range one_tri(tris[0], tris[0]);

  Range adj_edges;
  CHKERR mField.get_moab().get_adjacencies(one_tri, 1, false, adj_edges);
  EntityHandle boundary_meshset;
  CHKERR mField.get_moab().create_meshset(MESHSET_SET, boundary_meshset);
  CHKERR mField.get_moab().add_entities(boundary_meshset, adj_edges);
  simpleInterface->getBoundaryMeshSet() = boundary_meshset;

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  // We using AINSWORTH_LEGENDRE_BASE because DEMKOWICZ_JACOBI_BASE for triangle
  // and tet is not yet implemented for L2 space. DEMKOWICZ_JACOBI_BASE and
  // AINSWORTH_LEGENDRE_BASE are construcreed in the same way on quad.
  CHKERR simpleInterface->addDomainField("U", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simpleInterface->addDomainField("FLUX", HCURL, AINSWORTH_LEGENDRE_BASE,
                                         1);

  constexpr int order = 2;
  CHKERR simpleInterface->setFieldOrder("U", order - 1);
  CHKERR simpleInterface->setFieldOrder("FLUX", order);
  CHKERR simpleInterface->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Set integration rule]
MoFEMErrorCode Example::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule = [](int, int, int p) -> int { return 2 * p + 1; };

  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule);

  MoFEMFunctionReturn(0);
}
//! [Set integration rule]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;
  auto prb_mng = mField.getInterface<ProblemsManager>();

  Range tris;
  CHKERR mField.get_moab().get_entities_by_type(0, MBTRI, tris);
  Range one_tri(tris[0], tris[0]);
  one_tri.insert(tris[0]);
  CHKERR prb_mng->removeDofsOnEntities(simpleInterface->getProblemName(), "U",
                                       one_tri, 0, 1);
  Range adj_edges;
  CHKERR mField.get_moab().get_adjacencies(one_tri, 1, false, adj_edges);
  CHKERR prb_mng->removeDofsOnEntities(simpleInterface->getProblemName(),
                                       "FLUX", one_tri, 0, 1);

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();

  auto q_ptr = boost::make_shared<MatrixDouble>();
  auto div_q_ptr = boost::make_shared<VectorDouble>();
  auto u_ptr = boost::make_shared<VectorDouble>();
  auto dot_q_ptr = boost::make_shared<MatrixDouble>();
  auto dot_div_q_ptr = boost::make_shared<VectorDouble>();
  auto dot_u_ptr = boost::make_shared<VectorDouble>();

  auto set_domain_general = [&](auto &pipeline) {
    pipeline.push_back(new OpCalculateJacForFaceEmbeddedIn3DSpace(jAC));
    pipeline.push_back(new OpCalculateInvJacForFaceEmbeddedIn3DSpace(invJac));
    pipeline.push_back(new OpMakeHdivFromHcurl());
    pipeline.push_back(
        new OpSetContravariantPiolaTransformFaceEmbeddedIn3DSpace(jAC));
    pipeline.push_back(new OpSetInvJacHcurlFaceEmbeddedIn3DSpace(invJac));

    pipeline.push_back(new OpCalculateHdivVectorField<3>("FLUX", q_ptr));
    pipeline.push_back(
        new OpCalculateHdivVectorDivergence<3, 3>("FLUX", div_q_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("U", u_ptr));
    pipeline.push_back(new OpCalculateHdivVectorFieldDot<3>("FLUX", dot_q_ptr));
    pipeline.push_back(
        new OpCalculateHdivVectorDivergenceDot<3, 3>("FLUX", dot_div_q_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValuesDot("U", dot_u_ptr));
  };

  auto set_domain_lhs = [&](auto &pipeline) {
    auto beta = [](double,double,double) { return 1; };
    pipeline.push_back(new OpHdivHdiv("FLUX", "FLUX", beta));
    auto minus_one = []() { return -1; };
    pipeline.push_back(new OpHdivU("FLUX", "U", minus_one, false));
    pipeline.push_back(new OpLhs(q_ptr, div_q_ptr, dot_q_ptr));
  };

  auto set_domain_rhs = [&](auto &pipeline) {
    pipeline.push_back(new OpQQ("FLUX", q_ptr));
    pipeline.push_back(new OpDivQU("FLUX", u_ptr, -1));
    pipeline.push_back(new OpRhs(q_ptr, div_q_ptr, u_ptr, dot_q_ptr,
                                 dot_div_q_ptr, dot_u_ptr));

    // auto f = [](const double x, const double y, const double z) {
    //   return 1;
    // };
    // pipeline.push_back(new OpDomainSource("U", f));
  };

  set_domain_general(pipeline_mng->getOpDomainLhsPipeline());
  set_domain_general(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline());
  set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline());

  MoFEMFunctionReturn(0);
};
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;

  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto dm = simpleInterface->getDM();
  auto D = smartCreateDMVector(dm);

  auto solver = pipeline_mng->createTS();

  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR TSSetSolution(solver, D);

  // CHKERR TSPseudoSetTimeStep(solver, PETSC_NULL, PETSC_NULL);

  auto set_section_monitor = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    PetscViewerAndFormat *vf;
    CHKERR PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                      PETSC_VIEWER_DEFAULT, &vf);
    CHKERR SNESMonitorSet(
        snes,
        (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal, void *))SNESMonitorFields,
        vf, (MoFEMErrorCode(*)(void **))PetscViewerAndFormatDestroy);
    MoFEMFunctionReturn(0);
  };

  CHKERR TSSetFromOptions(solver);
  CHKERR TSSetUp(solver);
  CHKERR set_section_monitor(solver);
  CHKERR TSSolve(solver, NULL);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  double d_nrm;
  CHKERR VecNorm(D, NORM_2, &d_nrm);
  MOFEM_LOG("EXAMPLE", Sev::inform) << "Solution norm " << d_nrm;

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();

  auto post_proc_fe =
      boost::make_shared<PostProcFaceOnRefinedMeshFor2D>(mField);
  post_proc_fe->generateReferenceElementMesh();

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateJacForFaceEmbeddedIn3DSpace(jAC));
  post_proc_fe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
  post_proc_fe->getOpPtrVector().push_back(
      new OpSetContravariantPiolaTransformFaceEmbeddedIn3DSpace(jAC));

  post_proc_fe->addFieldValuesPostProc("FLUX");
  post_proc_fe->addFieldValuesPostProc("U");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_heat_method.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check results]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Check results]

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  LogManager::setLog("EXAMPLE");
  MOFEM_LOG_TAG("EXAMPLE", "example");

  try {

    //! [Register MoFEM discrete manager in PETSc]
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);
    //! [Register MoFEM discrete manager in PETSc

    //! [Create MoAB]
    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface
    //! [Create MoAB]

    //! [Create MoFEM]
    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database insterface
    //! [Create MoFEM]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}

Example::OpRhs::OpRhs(boost::shared_ptr<MatrixDouble> q_ptr,
                      boost::shared_ptr<VectorDouble> div_ptr,
                      boost::shared_ptr<VectorDouble> u_ptr,
                      boost::shared_ptr<MatrixDouble> dot_q_ptr,
                      boost::shared_ptr<VectorDouble> dot_div_ptr,
                      boost::shared_ptr<VectorDouble> dot_u_ptr)
    : DomainEleOp("U", DomainEleOp::OPROW), qPtr(q_ptr), divPtr(div_ptr),
      uPtr(u_ptr), dotQPtr(dot_q_ptr), dotDivPtr(dot_div_ptr),
      dotUPtr(dot_u_ptr) {}

MoFEMErrorCode Example::OpRhs::doWork(int side, EntityType type,
                                      DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;

  auto nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    
    auto t_q = getFTensor1FromMat<3>(*qPtr);
    auto t_div = getFTensor0FromVec(*divPtr);

    auto t_dot_div = getFTensor0FromVec(*dotDivPtr);

    auto nb_base_functions = data.getN().size2();
    auto nb_gauss_pts = getGaussPts().size2();
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(nf.begin(), &nf[nb_dofs], 0);

    auto t_base = data.getFTensor0N();
    auto t_w = getFTensor0IntegrationWeight();
    auto a = getMeasure();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      double alpha = t_w * a;

      const double dot = t_q(i) * t_q(i) + tol;
      const double res_log = log(dot);

      size_t bb = 0;
      for (; bb != nb_dofs; ++bb) {
        nf[bb] += alpha * t_base * res_log;
        nf[bb] += alpha * t_base * (l2 * t_div + t_dot_div);
        ++t_base;
      }

      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_w;
      ++t_q;
      ++t_div;
      ++t_dot_div;
    }

    CHKERR VecSetValues<MoFEM::EssentialBcStorage>(getKSPf(), data, &nf[0],
                                                   ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

Example::OpLhs::OpLhs(boost::shared_ptr<MatrixDouble> q_ptr,
                      boost::shared_ptr<VectorDouble> div_ptr,
                      boost::shared_ptr<MatrixDouble> dot_q_ptr)
    : DomainEleOp("U", "FLUX", DomainEleOp::OPROWCOL), qPtr(q_ptr),
      divPtr(div_ptr), dotQPtr(dot_q_ptr) {
  sYmm = false;
}

MoFEMErrorCode Example::OpLhs::doWork(int row_side, int col_side,
                                      EntityType row_type, EntityType col_type,
                                      EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  auto row_nb_dofs = row_data.getIndices().size();
  auto col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    auto t_q = getFTensor1FromMat<3>(*qPtr);
    auto t_div = getFTensor0FromVec(*divPtr);

    auto nb_gauss_pts = getGaussPts().size2();
    MatrixDouble loc_mat(row_nb_dofs, col_nb_dofs);
    loc_mat.clear();

    auto nb_base_functions = row_data.getN().size2();
    auto t_row_base = row_data.getFTensor0N();
    auto t_w = getFTensor0IntegrationWeight();
    auto a = getMeasure();

    auto ts_a = getFEMethod()->ts_a;

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      double alpha = t_w * a;

      size_t rr = 0;
      for (; rr != row_nb_dofs; ++rr) {

        auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
        auto t_col_diff_base = col_data.getFTensor2DiffN<3, 3>(gg, 0);

        const double dot = t_q(i) * t_q(i) + tol;
        double d_res_dot = 1 / dot;

        for (size_t cc = 0; cc != col_nb_dofs; ++cc) {
          loc_mat(rr, cc) +=
              alpha * t_row_base * d_res_dot * (2 * t_q(i) * t_col_base(i));
          loc_mat(rr, cc) +=
              alpha * t_row_base * (l2 + ts_a) * t_col_diff_base(i, i);

          ++t_col_base;
          ++t_col_diff_base;
        }

        ++t_row_base;
      }

      for (; rr < nb_base_functions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_q;
      ++t_div;
    }

    CHKERR MatSetValues<MoFEM::EssentialBcStorage>(
        getKSPB(), row_data, col_data, &loc_mat(0, 0), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}
