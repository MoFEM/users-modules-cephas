/**
 * \file helmholtz.cpp
 * \example helmholtz.cpp
 *
 * Using PipelineManager interface calculate Helmholtz problem. Example show how
 * to manage complex variable fields.
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

using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;
using EdgeEle = EdgeElementForcesAndSourcesCoreBase;
using EdgeEleOp = EdgeEle::UserDataOperator;

constexpr std::array<int, 3> d1_savitzky_golay_w3_p2 = {-1, 0, 1};
constexpr std::array<int, 5> d1_savitzky_golay_w5_p2 = {-2, -1, 0, 1, 2};
constexpr std::array<int, 7> d1_savitzky_golay_w7_p2 = {-3, -2, -1, 0, 1, 2, 3};
constexpr std::array<int, 9> d1_savitzky_golay_w9_p2 = {-4, -3, -2, -1, 0,
                                                         1,  2,  3,  4};

constexpr std::array<int, 5> d1_savitzky_golay_w5_p4 = {1, -8, 0, 8, -1};
constexpr std::array<int, 7> d1_savitzky_golay_w7_p4 = {22, -67, -58, 0,
                                                        58, 67,  -22};
constexpr std::array<int, 9> d1_savitzky_golay_w9_p4 = {
    86, -142, -193, -126, 0, 126, 193, 142, -86};

constexpr std::array<int, 10> d1_normalisation_p2 = {0,  0, 0,  2, 0,
                                                    10, 0, 28, 0, 60};
constexpr std::array<int, 10> d1_normalisation_p4 = {0, 0,   0, 0,   12,
                                                     0, 252, 0, 1188};

const int *d1_savitzky_golay_p2[] = {nullptr, nullptr,
                                     nullptr, d1_savitzky_golay_w3_p2.data(),
                                     nullptr, d1_savitzky_golay_w5_p2.data(),
                                     nullptr, d1_savitzky_golay_w7_p2.data(),
                                     nullptr, d1_savitzky_golay_w9_p2.data()};

const int *d1_savitzky_golay_p4[] = {nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     d1_savitzky_golay_w5_p4.data(),
                                     nullptr,
                                     d1_savitzky_golay_w7_p4.data(),
                                     nullptr,
                                     d1_savitzky_golay_w9_p4.data()};

using EntData = EntitiesFieldData::EntData;
using AssemblyBoundaryEleOp =
    FormsIntegrators<EdgeEleOp>::Assembly<PETSC>::OpBase;
using OpHdivHdiv = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<3, 3>;
using OpHdivU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixDivTimesScalar<2>;
using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

static double k = 1; // wave number

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystemIntensity();
  MoFEMErrorCode assembleSystemFlux();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode calculateFlux(double &calc_flux);
  MoFEMErrorCode outputResults(const int i);

  enum BoundingBox {
    CENTER_X = 0,
    CENTER_Y,
    MAX_X,
    MAX_Y,
    MIN_X,
    MIN_Y,
    LAST_BB
  };

  static std::vector<double> rZ;
  static std::vector<MatrixInt> iI;
  static std::array<double, LAST_BB> aveMaxMin;

  static int focalIndex;
  static std::pair<int, int> getCoordsInImage(double x, double y);

  static double rhsSource(const double x, const double y, const double);
  static double lhsFlux(const double x, const double y, const double);

  SmartPetscObj<Mat> B;
  SmartPetscObj<Vec> F;

  struct BoundaryOp;
  struct BoundaryRhs;
};

std::vector<double> Example::rZ;
std::vector<MatrixInt> Example::iI;
std::array<double, Example::BoundingBox::LAST_BB> Example::aveMaxMin;
int Example::focalIndex;

std::pair<int, int> Example::getCoordsInImage(double x, double y) {

  auto &m = iI[focalIndex];
  x -= aveMaxMin[MIN_X];
  y -= aveMaxMin[MIN_Y];
  x *= (m.size1() - 1) / (aveMaxMin[MAX_X] - aveMaxMin[MIN_X]);
  y *= (m.size2() - 1) / (aveMaxMin[MAX_Y] - aveMaxMin[MIN_Y]);
  const auto p = std::make_pair<int, int>(std::round(x), std::round(y));

#ifndef NDEBUG
  if (p.first < 0 && p.first >= m.size1())
    THROW_MESSAGE("Wrong index");
  if (p.second < 0 && p.second >= m.size2())
    THROW_MESSAGE("Wrong index");
#endif

  return p;
}

double Example::rhsSource(const double x, const double y, const double) {
  const auto idx = getCoordsInImage(x, y);
  const auto &up = iI[focalIndex + 1];
  const auto &down = iI[focalIndex - 1];
  const auto dz = rZ[focalIndex + 1] - rZ[focalIndex - 1];
  return -k *
         ((up(idx.first, idx.second) - down(idx.first, idx.second)) / (2 * dz));
}

double Example::lhsFlux(const double x, const double y, const double) {
  const auto idx = getCoordsInImage(x, y);
  const auto &m = iI[focalIndex];
  return 1. / m(idx.first, idx.second);
}

struct Example::BoundaryOp : public EdgeEleOp {
  BoundaryOp(boost::shared_ptr<MatrixDouble> flux_ptr, double &glob_flux)
      : EdgeEleOp(NOSPACE, OPLAST), fluxPtr(flux_ptr), globFlux(glob_flux) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const auto nb_gauss_pts = getGaussPts().size2();
    auto t_flux = getFTensor1FromMat<3>(*fluxPtr);
    auto t_normal = getFTensor1Normal();
    auto t_w = getFTensor0IntegrationWeight();
    FTensor::Index<'i', 3> i;
    for (auto gg = 0; gg != nb_gauss_pts; ++gg) {
      globFlux += t_w * t_normal(i) * t_flux(i);
      ++t_flux;
      ++t_w;
    };
    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> fluxPtr;
  double &globFlux;
};

struct Example::BoundaryRhs : public AssemblyBoundaryEleOp {
  BoundaryRhs(const std::string flux_field)
      : AssemblyBoundaryEleOp(flux_field, flux_field, OPROW) {}

  MoFEMErrorCode iNtegrate(EntData &row_data) {
    MoFEMFunctionBegin;
    const size_t nb_base_functions = row_data.getN().size2() / 3;
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor1N<3>();
    auto t_normal = getFTensor1Normal();
    FTensor::Index<'i', 3> i;
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      const double alpha = t_w;
      int rr = 0;
      for (; rr != nbRows; ++rr) {
        locF[rr] += alpha * t_row_base(i) * t_normal(i);
        ++t_row_base;
      }
      for (; rr < nb_base_functions; ++rr)
        ++t_row_base;
      ++t_w; // move to another integration weight
    }
    MoFEMFunctionReturn(0);
  }
};

//! [run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();

  for (auto i = 1; i != rZ.size() - 1; ++i) {

    MOFEM_LOG_CHANNEL("WORLD");
    MOFEM_LOG("WORLD", Sev::inform) << i << " focal length zR = " << rZ[i];

    focalIndex = i;
    CHKERR solveSystem();
    CHKERR outputResults(i);
  }

  MoFEMFunctionReturn(0);
}
//! [run problem]

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;

  auto read_data = []() {
    std::ifstream in;
    in.open("out_arrays.txt");
    std::vector<int> values;
    values.insert(values.end(), std::istream_iterator<int>(in),
                  std::istream_iterator<int>());
    MOFEM_LOG("WORLD", Sev::inform) << "Read data size " << values.size();
    in.close();
    return values;
  };

  auto struture_data = [&](auto &&data) {
    constexpr double scale = 1e4; // scale to float
    auto it = data.begin();
    if (it == data.end()) {
      THROW_MESSAGE("No images");
    }
    rZ.reserve(*it);
    iI.reserve(*it);
    MOFEM_LOG("WORLD", Sev::inform) << "Number of images " << *it;
    it++;
    for (; it != data.end();) {
      rZ.emplace_back(*(it++) / scale);
      const auto r = *(it++);
      const auto c = *(it++);
      iI.emplace_back(r, c);
      MOFEM_LOG("WORLD", Sev::inform)
          << "Read data set " << rZ.back() << " size " << r << " by " << c;
      auto &m = iI.back();
      for (auto rit = m.begin1(); rit != m.end1(); ++rit) {
        for (auto cit = rit.begin(); cit != rit.end(); ++cit) {
          *cit = *(it++);
        }
      }
    }
  };

  struture_data(read_data());

  auto get_bounding_box = [&]() {
    auto &moab = mField.get_moab();
    Range verts;
    MOAB_THROW(moab.get_entities_by_type(0, MBVERTEX, verts));

    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);

    Range verts_part;
    CHKERR pcomm->filter_pstatus(verts, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                 PSTATUS_NOT, -1, &verts_part);

    MatrixDouble coords(verts_part.size(), 3);
    CHKERR moab.get_coords(verts_part, &*coords.data().begin());

    std::array<double, 2> ave_coords{0, 0};
    for (auto v = 0; v != verts_part.size(); ++v) {
      ave_coords[0] += coords(v, 0);
      ave_coords[1] += coords(v, 1);
    }

    auto comm = mField.get_comm();

    int local_count = verts_part.size();
    int global_count;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, comm);
    std::array<double, 2> ave_coords_glob{0, 0};
    MPI_Allreduce(ave_coords.data(), ave_coords.data(), 2, MPI_DOUBLE, MPI_SUM,
                  comm);
    ave_coords_glob[0] /= global_count;
    ave_coords_glob[1] /= global_count;

    std::array<double, 2> max_coords{ave_coords_glob[0], ave_coords_glob[1]};
    for (auto v = 0; v != verts_part.size(); ++v) {
      max_coords[0] = std::max(max_coords[0], coords(v, 0));
      max_coords[1] = std::max(max_coords[1], coords(v, 1));
    }
    std::array<double, 2> max_coords_glob{0, 0};
    MPI_Allreduce(max_coords.data(), max_coords_glob.data(), 2, MPI_DOUBLE,
                  MPI_MAX, comm);

    std::array<double, 2> min_coords{max_coords_glob[0], max_coords_glob[1]};
    for (auto v = 0; v != verts_part.size(); ++v) {
      min_coords[0] = std::min(min_coords[0], coords(v, 0));
      min_coords[1] = std::min(min_coords[1], coords(v, 1));
    }
    std::array<double, 2> min_coords_glob{0, 0};
    MPI_Allreduce(min_coords.data(), min_coords_glob.data(), 2, MPI_DOUBLE,
                  MPI_MIN, comm);

    return std::array<double, LAST_BB>{ave_coords_glob[0], ave_coords_glob[1],
                                       max_coords_glob[0], max_coords_glob[1],
                                       min_coords_glob[0], min_coords_glob[1]};
  };

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  aveMaxMin = get_bounding_box();

  MOFEM_LOG("WORLD", Sev::inform)
      << "Centre " << aveMaxMin[CENTER_X] << " " << aveMaxMin[CENTER_Y];
  MOFEM_LOG("WORLD", Sev::inform)
      << "Max " << aveMaxMin[MAX_X] << " " << aveMaxMin[MAX_Y];
  MOFEM_LOG("WORLD", Sev::inform)
      << "Min " << aveMaxMin[MIN_X] << " " << aveMaxMin[MIN_Y];

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;

  // Note that in 2D case HDIV and HCURL spaces are isomorphic, and therefore
  // only base for HCURL has been implemented in 2D. Base vectors for HDIV space
  // are be obtained after rotation of HCURL base vectors by a right angle
  CHKERR simpleInterface->addDomainField("S", HCURL, DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simpleInterface->addBoundaryField("S", HCURL, DEMKOWICZ_JACOBI_BASE,
                                           1);
  // We use AINSWORTH_LEGENDRE_BASE since DEMKOWICZ_JACOBI_BASE for triangle
  // is not yet implemented for L2 space. For quads DEMKOWICZ_JACOBI_BASE and
  // AINSWORTH_LEGENDRE_BASE are construcreed in the same way
  CHKERR simpleInterface->addDomainField("PHI", L2, DEMKOWICZ_JACOBI_BASE, 1);

  int base_order = 1;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-base_order", &base_order,
                            PETSC_NULL);

  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-k", &k, PETSC_NULL);

  MOFEM_LOG("WORLD", Sev::inform) << "Base order " << base_order;
  MOFEM_LOG("WORLD", Sev::inform) << "Wave number " << k;

  CHKERR simpleInterface->setFieldOrder("S", base_order);
  CHKERR simpleInterface->setFieldOrder("PHI", base_order - 1);
  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Applying essential BC]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;

  MoFEMFunctionReturn(0);
}
//! [Applying essential BC]

//! [Calculate flux on boundary]
MoFEMErrorCode Example::calculateFlux(double &calc_flux) {
  MoFEMFunctionBegin;
  auto pipeline_mng = mField.getInterface<PipelineManager>();

  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();

  auto rule_vol = [](int, int, int order) { return 2 * (order + 1); };
  pipeline_mng->setBoundaryRhsIntegrationRule(rule_vol);

  auto flux_ptr = boost::make_shared<MatrixDouble>();
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpSetContravariantPiolaTransformOnEdge2D());
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpCalculateHVecVectorField<3>("S", flux_ptr));
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new BoundaryOp(flux_ptr, calc_flux));

  calc_flux = 0;
  CHKERR pipeline_mng->loopFiniteElements();
  double global_flux_assembeld = 0;
  MPI_Allreduce(&calc_flux, &global_flux_assembeld, 1, MPI_DOUBLE, MPI_SUM,
                mField.get_comm());
  calc_flux = global_flux_assembeld;

  MoFEMFunctionReturn(0);
}
//! [Calculate flux on boundary]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystemIntensity() {
  MoFEMFunctionBegin;

  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();

  auto rule_vol = [](int, int, int order) { return 2 * (order + 1); };
  pipeline_mng->setDomainLhsIntegrationRule(rule_vol);
  pipeline_mng->setDomainRhsIntegrationRule(rule_vol);

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateHOJacForFace(jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpMakeHdivFromHcurl());
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetInvJacHcurlFace(inv_jac_ptr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpSetHOWeightsOnFace());

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpHdivHdiv("S", "S", lhsFlux));
  auto unity = []() { return 1; };
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpHdivU("S", "PHI", unity, true));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainSource("PHI", rhsSource));

  auto dm = simpleInterface->getDM();
  F = smartCreateDMVector(dm);
  B = smartCreateDMMatrix(dm);

  pipeline_mng->getDomainLhsFE()->ksp_A = B;
  pipeline_mng->getDomainLhsFE()->ksp_B = B;
  pipeline_mng->getDomainRhsFE()->ksp_f = F;

  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
  CHKERR MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
  CHKERR VecAssemblyBegin(F);
  CHKERR VecAssemblyEnd(F);
  CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystemFlux() {
  MoFEMFunctionBegin;

  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();

  auto rule_vol = [](int, int, int order) { return 2 * (order + 1); };
  pipeline_mng->setBoundaryRhsIntegrationRule(rule_vol);
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpSetContravariantPiolaTransformOnEdge2D());
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(new BoundaryRhs("S"));

  auto dm = simpleInterface->getDM();
  F = smartCreateDMVector(dm);
  pipeline_mng->getBoundaryRhsFE()->ksp_f = F;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR VecAssemblyBegin(F);
  CHKERR VecAssemblyEnd(F);
  CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);

  double fnorm;
  CHKERR VecNorm(F, NORM_2, &fnorm);
  MOFEM_LOG("WORLD", Sev::inform) << "Flux F norm " << std::scientific << fnorm;

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;

  CHKERR assembleSystemIntensity();

  auto dm = simpleInterface->getDM();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  auto solver = MoFEM::createKSP(mField.get_comm());
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPSetOperators(solver, B, B);
  CHKERR KSPSetUp(solver);

  auto iD = smartCreateDMVector(dm);
  CHKERR KSPSolve(solver, F, iD);
  CHKERR VecGhostUpdateBegin(iD, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(iD, INSERT_VALUES, SCATTER_FORWARD);

  CHKERR DMoFEMMeshToLocalVector(dm, iD, INSERT_VALUES, SCATTER_REVERSE);
  double i_lambda_flux;
  CHKERR calculateFlux(i_lambda_flux);
  MOFEM_LOG_CHANNEL("WORLD");
  MOFEM_LOG("WORLD", Sev::inform)
      << "iD flux " << std::scientific << i_lambda_flux;

  auto lambdaD = smartVectorDuplicate(iD);
  CHKERR assembleSystemFlux();
  CHKERR KSPSolve(solver, F, lambdaD);
  CHKERR VecGhostUpdateBegin(lambdaD, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(lambdaD, INSERT_VALUES, SCATTER_FORWARD);

  CHKERR DMoFEMMeshToLocalVector(dm, lambdaD, INSERT_VALUES, SCATTER_REVERSE);
  double lambda_flux;
  CHKERR calculateFlux(lambda_flux);
  MOFEM_LOG_CHANNEL("WORLD");
  MOFEM_LOG("WORLD", Sev::inform)
      << "lambdaD flux " << std::scientific << lambda_flux;

  double lambda = -i_lambda_flux / lambda_flux;

  // CHKERR VecAXPY(iD, lambda, lambdaD);
  // CHKERR DMoFEMMeshToLocalVector(dm, iD, INSERT_VALUES, SCATTER_REVERSE);
  // double zero_flux;
  // CHKERR calculateFlux(zero_flux);
  // MOFEM_LOG_CHANNEL("WORLD");
  // MOFEM_LOG("WORLD", Sev::inform)
  //     << "Zero flux " << std::scientific << zero_flux;

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults(const int i) {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc("S");
  post_proc_fe->addFieldValuesPostProc("PHI");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_" + boost::lexical_cast<std::string>(i) +
                                 ".h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

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
