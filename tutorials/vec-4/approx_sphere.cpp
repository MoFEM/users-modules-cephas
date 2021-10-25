/**
 * \file approx_sphere.cpp
 * \example approx_sphere.cpp
 *
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

#include <BasicFiniteElements.hpp>

constexpr int FM_DIM = 2;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = MoFEM::FaceElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

using DomainEle = ElementsAndOps<FM_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

constexpr double a = 1;
constexpr double a2 = a * a;
constexpr double a4 = a2 * a2;

constexpr double A = 6371220;

FTensor::Index<'i', 3> i;
FTensor::Index<'j', 3> j;
FTensor::Index<'k', 3> k;

auto res_J = [](const double x, const double y, const double z) {
  const double res = (x * x + y * y + z * z - a2);
  return res;
};

auto res_J_dx = [](const double x, const double y, const double z) {
  const double res = res_J(x, y, z);
  return FTensor::Tensor1<double, 3>{res * (2 * x), res * (2 * y),
                                     res * (2 * z)};
};

auto lhs_J_dx2 = [](const double x, const double y, const double z) {
  const double res = res_J(x, y, z);
  return FTensor::Tensor2<double, 3, 3>{

      (res * 2 + (4 * x * x)),
      (4 * y * x),
      (4 * z * x),

      (4 * x * y),
      (2 * res + (4 * y * y)),
      (4 * z * y),

      (4 * x * z),
      (4 * y * z),
      (2 * res + (4 * z * z))};
};

struct OpRhs : public AssemblyDomainEleOp {

  OpRhs(const std::string field_name, boost::shared_ptr<MatrixDouble> x_ptr,
        boost::shared_ptr<MatrixDouble> dot_x_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        xPtr(x_ptr), xDotPtr(dot_x_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data) {
    MoFEMFunctionBegin;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();

    auto t_x0 = getFTensor1CoordsAtGaussPts();
    auto t_x = getFTensor1FromMat<3>(*xPtr);
    auto t_dot_x = getFTensor1FromMat<3>(*xDotPtr);
    auto t_normal = getFTensor1NormalsAtGaussPts();

    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      FTensor::Tensor1<double, 3> t_n{t_x0(0), t_x0(1), t_x0(2)};
      t_n.normalize();
      FTensor::Tensor2<double, 3, 3> t_P, t_Q;
      t_P(i, j) = t_n(i) * t_n(j);
      t_Q(i, j) = t_kd(i, j) - t_P(i, j);

      auto t_J_res = res_J_dx(t_x(0), t_x(1), t_x(2));

      const double alpha = t_w;
      auto t_nf = getFTensor1FromArray<3, 3>(locF);
      double l = sqrt(t_normal(i) * t_normal(i));

      FTensor::Tensor1<double, 3> t_res;
      t_res(i) =
          alpha * l * ((t_P(i, k) * t_J_res(k) + t_Q(i, k) * t_dot_x(k)));

      int rr = 0;
      for (; rr != nbRows / 3; ++rr) {

        t_nf(j) += t_row_base * t_res(j);

        ++t_row_base;
        ++t_nf;
      }
      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
      }

      ++t_w;
      ++t_x;
      ++t_dot_x;
      ++t_x0;
      ++t_normal;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> xPtr;
  boost::shared_ptr<MatrixDouble> xDotPtr;
};

struct OpLhs : public AssemblyDomainEleOp {

  OpLhs(const std::string field_name, boost::shared_ptr<MatrixDouble> x_ptr,
        boost::shared_ptr<MatrixDouble> dot_x_ptr)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        xPtr(x_ptr), xDotPtr(dot_x_ptr) {
    this->sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();

    auto t_x0 = getFTensor1CoordsAtGaussPts();
    auto t_x = getFTensor1FromMat<3>(*xPtr);
    auto t_dot_x = getFTensor1FromMat<3>(*xDotPtr);
    auto t_normal = getFTensor1NormalsAtGaussPts();

    auto get_t_mat = [&](const int rr) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>{
          &locMat(rr + 0, 0), &locMat(rr + 0, 1), &locMat(rr + 0, 2),

          &locMat(rr + 1, 0), &locMat(rr + 1, 1), &locMat(rr + 1, 2),

          &locMat(rr + 2, 0), &locMat(rr + 2, 1), &locMat(rr + 2, 2)};
    };

    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
    const double ts_a = getTSa();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      FTensor::Tensor1<double, 3> t_n{t_x0(0), t_x0(1), t_x0(2)};
      t_n.normalize();
      FTensor::Tensor2<double, 3, 3> t_P, t_Q;
      t_P(i, j) = t_n(i) * t_n(j);
      t_Q(i, j) = t_kd(i, j) - t_P(i, j);

      auto t_J_lhs = lhs_J_dx2(t_x(0), t_x(1), t_x(2));
      double l = sqrt(t_normal(i) * t_normal(i));

      const double alpha = t_w;
      FTensor::Tensor2<double, 3, 3> t_lhs;
      t_lhs(i, j) =
          (alpha * l) * (t_P(i, k) * t_J_lhs(k, j) + t_Q(i, j) * ts_a);

      int rr = 0;
      for (; rr != nbRows / 3; rr++) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_mat = get_t_mat(3 * rr);

        for (int cc = 0; cc != nbCols / 3; cc++) {

          const double rc = t_row_base * t_col_base;
          t_mat(i, j) += rc * t_lhs(i, j);

          ++t_col_base;
          ++t_mat;
        }
        ++t_row_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_x;
      ++t_x0;
      ++t_normal;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> xPtr;
  boost::shared_ptr<MatrixDouble> xDotPtr;
};

struct OpError : public DomainEleOp {

  OpError(const std::string field_name, boost::shared_ptr<MatrixDouble> x_ptr)
      : DomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        xPtr(x_ptr) {

    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {

    MoFEMFunctionBegin;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_x = getFTensor1FromMat<3>(*xPtr);
    auto t_normal = getFTensor1NormalsAtGaussPts();
    auto nb_integration_pts = getGaussPts().size2();

    double error = 0;

    for (int gg = 0; gg != nb_integration_pts; gg++) {

      double l = sqrt(t_normal(i) * t_normal(i));
      error += t_w * l * std::abs((t_x(i) * t_x(i) - A * A));

      ++t_w;
      ++t_x;
      ++t_normal;
    }

    CHKERR VecSetValue(errorVec, 0, error, ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

  static SmartPetscObj<Vec> errorVec;

private:
  boost::shared_ptr<MatrixDouble> xPtr;
};

SmartPetscObj<Vec> OpError::errorVec;

struct ApproxSphere {

  ApproxSphere(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode getOptions();
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setOPs();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
};

//! [Run programme]
MoFEMErrorCode ApproxSphere::runProblem() {
  MoFEMFunctionBegin;
  CHKERR getOptions();
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR setOPs();
  CHKERR solveSystem();
  CHKERR outputResults();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

MoFEMErrorCode ApproxSphere::getOptions() {
  MoFEMFunctionBeginHot;
  MoFEMFunctionReturnHot(0);
}

//! [Read mesh]
MoFEMErrorCode ApproxSphere::readMesh() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();
  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode ApproxSphere::setupProblem() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  CHKERR simple->addDomainField("HO_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE, 3);
  CHKERR simple->addDataField("HO_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE, 3);

  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("HO_POSITIONS", order);
  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Push operators to pipeline]
MoFEMErrorCode ApproxSphere::setOPs() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  auto integration_rule = [](int, int, int approx_order) {
    return 3 * approx_order;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

  auto x_ptr = boost::make_shared<MatrixDouble>();
  auto dot_x_ptr = boost::make_shared<MatrixDouble>();
  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  auto def_ops = [&](auto &pipeline) {
    pipeline.push_back(
        new OpCalculateVectorFieldValues<3>("HO_POSITIONS", x_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldValuesDot<3>("HO_POSITIONS", dot_x_ptr));
  };

  def_ops(pipeline_mng->getOpDomainRhsPipeline());
  def_ops(pipeline_mng->getOpDomainLhsPipeline());

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpRhs("HO_POSITIONS", x_ptr, dot_x_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpLhs("HO_POSITIONS", x_ptr, dot_x_ptr));

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode ApproxSphere::solveSystem() {
  MoFEMFunctionBegin;

  // // Project HO geometry from mesh
  Projection10NodeCoordsOnField ent_method_material(mField, "HO_POSITIONS");
  CHKERR mField.loop_dofs("HO_POSITIONS", ent_method_material);

  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto dm = simple->getDM();
  MoFEM::SmartPetscObj<TS> ts;
  ts = pipeline_mng->createTS();

  double ftime = 1;
  CHKERR TSSetMaxSteps(ts, 1);
  CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

  auto T = smartCreateDMVector(simple->getDM());
  CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                 SCATTER_FORWARD);
  CHKERR TSSetSolution(ts, T);
  CHKERR TSSetFromOptions(ts);

  CHKERR TSSolve(ts, NULL);
  CHKERR TSGetTime(ts, &ftime);

  CHKERR mField.getInterface<FieldBlas>()->fieldScale(A, "HO_POSITIONS");

  MoFEMFunctionReturn(0);
}

//! [Solve]
MoFEMErrorCode ApproxSphere::outputResults() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto dm = simple->getDM();

  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc("HO_POSITIONS");
  CHKERR DMoFEMLoopFiniteElements(dm, "dFE", post_proc_fe);
  CHKERR post_proc_fe->writeFile("out_approx.h5m");

  auto error_fe = boost::make_shared<DomainEle>(mField);

  auto x_ptr = boost::make_shared<MatrixDouble>();
  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  error_fe->getOpPtrVector().push_back(
      new OpGetHONormalsOnFace("HO_POSITIONS"));
  error_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<3>("HO_POSITIONS", x_ptr));
  error_fe->getOpPtrVector().push_back(new OpError("HO_POSITIONS", x_ptr));

  error_fe->preProcessHook = [&]() {
    MoFEMFunctionBegin;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Create vec ";
    OpError::errorVec = createSmartVectorMPI(
        mField.get_comm(), (!mField.get_comm_rank()) ? 1 : 0, 1);
    VecZeroEntries(OpError::errorVec);
    MoFEMFunctionReturn(0);
  };

  error_fe->postProcessHook = [&]() {
    MoFEMFunctionBegin;
    CHKERR VecAssemblyBegin(OpError::errorVec);
    CHKERR VecAssemblyEnd(OpError::errorVec);
    double error2;
    CHKERR VecSum(OpError::errorVec, &error2);
    MOFEM_LOG("EXAMPLE", Sev::inform)
        << "Error " << sqrt(error2 / (4 * M_PI * A * A));
    OpError::errorVec.reset();
    MoFEMFunctionReturn(0);
  };

  CHKERR DMoFEMLoopFiniteElements(dm, "dFE", error_fe);

  CHKERR simple->deleteDM();
  CHKERR simple->deleteFiniteElements();

  CHKERR mField.get_moab().write_file("out_ho_mesh.h5m");

  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

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

    //! [ApproxSphere]
    ApproxSphere ex(m_field);
    CHKERR ex.runProblem();
    //! [ApproxSphere]
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
}
