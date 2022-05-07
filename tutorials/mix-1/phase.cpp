/**
 * \file phase.cpp
 * \example phase.cpp
 *
 * Calculate light phase using <a
 * href="https://en.wikipedia.org/wiki/Transport-of-intensity_equation">Transport
 * intensity equation</a>.
 * 
 * See paper \cite parvizi2015practical
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
constexpr std::array<int, 10> d1_normalisation_p4 = {0,  0, 0,   0, 0,
                                                     12, 0, 252, 0, 1188};

const int *d1_savitzky_golay_p2[] = {nullptr, nullptr,
                                     nullptr, d1_savitzky_golay_w3_p2.data(),
                                     nullptr, d1_savitzky_golay_w5_p2.data(),
                                     nullptr, d1_savitzky_golay_w7_p2.data(),
                                     nullptr, d1_savitzky_golay_w9_p2.data()};

const int *d1_savitzky_golay_p4[] = {nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     d1_savitzky_golay_w5_p4.data(),
                                     nullptr,
                                     d1_savitzky_golay_w7_p4.data(),
                                     nullptr,
                                     d1_savitzky_golay_w9_p4.data()};

const int **d1_savitzky_golay[] = {nullptr, nullptr, d1_savitzky_golay_p2,
                                   nullptr, d1_savitzky_golay_p4};
const int *d1_normalisation[] = {nullptr, nullptr, d1_normalisation_p2.data(),
                                 nullptr, d1_normalisation_p4.data()};

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
static int order_savitzky_golay = 2;
static int window_savitzky_golay = 3;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
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
  static int savitzkyGolayNormalisation;
  static const int *savitzkyGolayWeights;

  static std::pair<int, int> getCoordsInImage(double x, double y);

  static double rhsSource(const double x, const double y, const double);
  static double lhsFlux(const double x, const double y, const double);

  struct BoundaryOp;
};

std::vector<double> Example::rZ;
std::vector<MatrixInt> Example::iI;
std::array<double, Example::BoundingBox::LAST_BB> Example::aveMaxMin;
int Example::focalIndex;
int Example::savitzkyGolayNormalisation;
const int *Example::savitzkyGolayWeights;

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

  double v = 0;
  for (auto w = 0; w != window_savitzky_golay; ++w) {
    const auto i = focalIndex - (window_savitzky_golay - 1) / 2 + w;
    const auto &intensity = iI[i];
    v += intensity(idx.first, idx.second) * savitzkyGolayWeights[w];
  }
  v = static_cast<double>(v) / savitzkyGolayNormalisation;

  const auto dz = rZ[focalIndex + 1] - rZ[focalIndex - 1];
  return -k * v / dz;
}

double Example::lhsFlux(const double x, const double y, const double) {
  const auto idx = getCoordsInImage(x, y);
  const auto &m = iI[focalIndex];
  return 1. / m(idx.first, idx.second);
}

struct Example::BoundaryOp : public EdgeEleOp {
  BoundaryOp(boost::shared_ptr<MatrixDouble> flux_ptr, double &glob_flux)
      : EdgeEleOp(NOSPACE, OPSPACE), fluxPtr(flux_ptr), globFlux(glob_flux) {}

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

//! [run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();

  char images_array_files[255] = "out_arrays.txt";
  CHKERR PetscOptionsGetString(PETSC_NULL, "", "-images_arrays",
                               images_array_files, 255, PETSC_NULL);

  // Look into "extract_data.ipynb" file, it is used to generate data file with
  // combined stack of the images.
  auto read_images = [&]() {
    std::ifstream in;
    in.open(images_array_files);
    std::vector<int> values;
    values.insert(values.end(), std::istream_iterator<int>(in),
                  std::istream_iterator<int>());
    MOFEM_LOG("WORLD", Sev::inform) << "Read data size " << values.size();
    in.close();
    return values;
  };

  auto structure_data = [&](auto &&data) {
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

  structure_data(read_images());

  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-k", &k, PETSC_NULL);
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order_savitzky_golay",
                               &order_savitzky_golay, PETSC_NULL);
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-window_savitzky_golay",
                            &window_savitzky_golay, PETSC_NULL);

  int window_shift = 0;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-window_shift", &window_shift,
                            PETSC_NULL);

  MOFEM_LOG("WORLD", Sev::inform) << "Wave number " << k;
  MOFEM_LOG("WORLD", Sev::inform)
      << "Order Savitzky Golay " << order_savitzky_golay;
  MOFEM_LOG("WORLD", Sev::inform)
      << "Window Savitzky Golay " << window_savitzky_golay;
  MOFEM_LOG("WORLD", Sev::inform) << "Window shift " << window_shift;

  if ((rZ.size() - 1) % 2)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Expected even number of images");

  focalIndex = (rZ.size() - 1) / 2 + window_shift;
  MOFEM_LOG("WORLD", Sev::inform) << "zR for mid plane " << rZ[focalIndex];

  const auto d1_sg_data = d1_savitzky_golay[order_savitzky_golay];
  if (!d1_sg_data)
    SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
            "Wrong Savitzky Golay order");

  auto d1_sg_data_window = d1_sg_data[window_savitzky_golay];
  if (!d1_sg_data_window)
    SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
            "Wrong Savitzky Golay window");

  savitzkyGolayNormalisation =
      d1_normalisation[order_savitzky_golay][window_savitzky_golay];
  savitzkyGolayWeights = d1_sg_data_window;

  CHKERR assembleSystemIntensity();
  CHKERR solveSystem();
  CHKERR outputResults(0);

  MoFEMFunctionReturn(0);
}
//! [run problem]

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;

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


  MOFEM_LOG("WORLD", Sev::inform) << "Base order " << base_order;

  CHKERR simpleInterface->setFieldOrder("S", base_order);
  CHKERR simpleInterface->setFieldOrder("PHI", base_order - 1);
  CHKERR simpleInterface->setUp();

  // Block PHI-PHI is empty
  CHKERR simpleInterface->addFieldToEmptyFieldBlocks("PHI", "PHI");

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

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
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpSetHOWeightsOnFace());
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

  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpSetHOWeightsOnFace());
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainSource("PHI", rhsSource));

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;

  auto dm = simpleInterface->getDM();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  auto solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPSetUp(solver);

  auto iD = smartCreateDMVector(dm);
  auto F = smartVectorDuplicate(iD);
  CHKERR KSPSolve(solver, F, iD);
  CHKERR VecGhostUpdateBegin(iD, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(iD, INSERT_VALUES, SCATTER_FORWARD);

  CHKERR DMoFEMMeshToLocalVector(dm, iD, INSERT_VALUES, SCATTER_REVERSE);
  double i_lambda_flux;
  CHKERR calculateFlux(i_lambda_flux);
  MOFEM_LOG_CHANNEL("WORLD");
  MOFEM_LOG("WORLD", Sev::inform)
      << "iD flux " << std::scientific << i_lambda_flux;

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

  auto jac_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateHOJacForFace(jac_ptr));
  post_proc_fe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
  post_proc_fe->getOpPtrVector().push_back(
      new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));

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
