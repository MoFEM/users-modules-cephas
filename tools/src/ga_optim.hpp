/** \file CPCutMesh.hpp
  \brief Cutting mesh
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

#ifndef __CP_CUTMESH_HPP__
#define __CP_CUTMESH_HPP__

namespace MoFEM {


  static MoFEMErrorCode cutMeshRout(const double tolCut, const double tolCutClose,
                             const double tolTrim, const double tolTrimClose,
                             double &fitness);

  struct CutWithGA : public UnknownInterface {

    MoFEMErrorCode query_interface(const MOFEMuuid &uuid,
                                   UnknownInterface **iface) const;

    MoFEM::Core &cOre;
    CutWithGA(const MoFEM::Core &core);
    ~CutWithGA() {}

    int lineSearchSteps;
    int nbMaxMergingCycles;
    int nbMaxTrimSearchIterations;

    /**
     * \brief Get options from command line
     * @return error code
     */
  MoFEMErrorCode getOptions() {
    MoFEMFunctionBegin;
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "MOFEM Cut mesh options",
                             "none");

    CHKERR PetscOptionsInt("-cut_linesearch_steps",
                           "number of bisection steps which line search do to "
                           "find optimal merged nodes position",
                           "", lineSearchSteps, &lineSearchSteps, PETSC_NULL);

    CHKERR PetscOptionsInt("-cut_max_merging_cycles",
                           "number of maximal merging cycles", "",
                           nbMaxMergingCycles, &nbMaxMergingCycles, PETSC_NULL);

    CHKERR PetscOptionsInt(
        "-cut_max_trim_iterations", "number of maximal merging cycles", "",
        nbMaxTrimSearchIterations, &nbMaxTrimSearchIterations, PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRG(ierr);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief set surface entities
   * @param  surface entities which going to be added
   * @return         error code
   */
  MoFEMErrorCode setSurface(const Range &surface);

  /**
   * \brief copy surface entities
   * @param  surface entities which going to be added
   * @return         error code
   */
  MoFEMErrorCode copySurface(const Range &surface, Tag th = NULL,
                             double *shift = NULL, double *origin = NULL,
                             double *transform = NULL,
                             const std::string save_mesh = "");

  /**
   * \brief set volume entities
   * @param  volume entities which going to be added
   * @return         error code
   */
  MoFEMErrorCode setVolume(const Range &volume);

  /**
   * \brief merge surface entities
   * @param  surface entities which going to be added
   * @return         error code
   */
  MoFEMErrorCode mergeSurface(const Range &surface);

  /**
   * \brief merge volume entities
   * @param  volume entities which going to be added
   * @return         error code
   */
  MoFEMErrorCode mergeVolumes(const Range &volume);

  /**
   * \brief build tree
   * @return error code
   */
  MoFEMErrorCode buildTree();

  MoFEMErrorCode
  cutAndTrim(const BitRefLevel &bit_level1, const BitRefLevel &bit_level2,
             Tag th, const double tol_cut, const double tol_cut_close,
             const double tol_trim, const double tol_trim_close,
             Range *fixed_edges = NULL, Range *corner_nodes = NULL,
             const bool update_meshsets = false, const bool debug = true);

  MoFEMErrorCode
  cutTrimAndMerge(const int fraction_level, const BitRefLevel &bit_level1,
                  const BitRefLevel &bit_level2, const BitRefLevel &bit_level3,
                  Tag th, const double tol_cut, const double tol_cut_close,
                  const double tol_trim, const double tol_trim_close,
                  double &fitness, Range fixed_edges, Range corner_nodes,
                  const bool update_meshsets = false, const bool debug = false);

  /**
   * \brief find edges to cut
   * @param  verb verbosity level
   * @return      error code
   */
  MoFEMErrorCode findEdgesToCut(Range *fixed_edges, Range *corner_nodes,
                                const double low_tol = 0, int verb = 0,
                                const bool debug = false);

  MoFEMErrorCode projectZeroDistanceEnts(Range *fixed_edges,
                                         Range *corner_nodes,
                                         const double low_tol = 0,
                                         const int verb = QUIET,
                                         const bool debug = false);

  /**
   * \brief cut edges
   *
   * For edges to cut (calculated by findEdgesToCut), edges are split in the
   * middle and then using MoFEM::MeshRefinement interface, tetrahedra mesh
   * are cut.
   *
   * @param  bit BitRefLevel of new mesh created by cutting edges
   * @return     error code
   */
  MoFEMErrorCode cutEdgesInMiddle(const BitRefLevel bit,
                                  const bool debug = false);

  /**
   * \brief projecting of mid edge nodes on new mesh on surface
   * @return error code
   */
  MoFEMErrorCode moveMidNodesOnCutEdges(Tag th = NULL);

  /**
   * \brief Find edges to trimEdges

   * To make this work, you need to find edges to cut (findEdgesToCut), then
   * cut edges in the middle (cutEdgesInMiddle) and finally project edges on
   * the surface (moveMidNodesOnCutEdges)

   * @param  verb verbosity level
   * @return      error code
   */
  MoFEMErrorCode findEdgesToTrim(Range *fixed_edges, Range *corner_nodes,
                                 Tag th = NULL, const double tol = 1e-4,
                                 int verb = 0);

  /**
   * \brief trim edges
   * @param  bit bit level of the trimmed mesh
   * @return     error code
   */
  MoFEMErrorCode trimEdgesInTheMiddle(const BitRefLevel bit, Tag th = NULL,
                                      const double tol = 1e-4,
                                      const bool debug = false);

  /**
   * \brief move trimmed edges mid nodes
   * @return error code
   */
  MoFEMErrorCode moveMidNodesOnTrimmedEdges(Tag th = NULL);

  /**
   * \brief Remove pathological elements on surface internal front
   *
   * Internal surface skin is a set of edges in iterior of the body on boundary
   * of surface. This set of edges is called surface front. If surface face has
   * three nodes on surface front, non of the face nodes is split and should be
   * removed from surface if it is going to be split.
   *
   * @param  split_bit split bit level
   * @param  bit       bit level of split mesh
   * @param  ents      ents on the surface which is going to be split
   * @return           error code
   */
  MoFEMErrorCode removePathologicalFrontTris(const BitRefLevel split_bit,
                                             Range &ents);

  /**
   * \brief split sides
   * @param  split_bit split bit level
   * @param  bit       bit level of split mesh
   * @param  ents      ents on the surface which is going to be split
   * @return           error code
   */
  MoFEMErrorCode splitSides(const BitRefLevel split_bit, const BitRefLevel bit,
                            const Range &ents, Tag th = NULL);

  /**
   * @brief Merge edges
   *
   * Sort all edges, where sorting is by quality calculated as edge length times
   * quality of tets adjacent to the edge. Edge is merged if quality if the mesh
   * is improved.
   *
   * @param fraction_level Fraction of edges attemt to be merged at iteration
   * @param tets Tets of the mesh which edges are merged
   * @param surface Surface created by edge spliting
   * @param fixed_edges edges which are geometrical corners of the body
   * @param corner_nodes vertices on the corners
   * @param merged_nodes  merged nodes
   * @param out_tets  returned test after merge
   * @param new_surf  new surface without merged edges
   * @param th  tag with nodal positons
   * @param bit_ptr set bit ref level to mesh without merged edges
   * @param debug
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode mergeBadEdges(const int fraction_level, const Range &tets,
                               const Range &surface, const Range &fixed_edges,
                               const Range &corner_nodes, Range &merged_nodes,
                               Range &out_tets, Range &new_surf, Tag th,
                               const bool update_meshsets = false,
                               const BitRefLevel *bit_ptr = NULL,
                               const bool debug = false);

  /**
   * @brief Merge edges
   *
   * Sort all edges, where sorting is by quality calculated as edge length times
   * quality of tets adjacent to the edge. Edge is merged if quality if the mesh
   * is improved.
   */
  MoFEMErrorCode
  mergeBadEdges(const int fraction_level, const BitRefLevel cut_bit,
                const BitRefLevel trim_bit, const BitRefLevel bit,
                const Range &surface, const Range &fixed_edges,
                const Range &corner_nodes, Tag th = NULL,
                const bool update_meshsets = false, const bool debug = false);

#ifdef WITH_TETGEN

  MoFEMErrorCode
  rebuildMeshWithTetGen(vector<string> &switches, const BitRefLevel &mesh_bit,
                        const BitRefLevel &bit, const Range &surface,
                        const Range &fixed_edges, const Range &corner_nodes,
                        Tag th = NULL, const bool debug = false);

#endif

  /**
   * \brief set coords to tag
   * @param  th tag handle
   * @return    error code
   */
  MoFEMErrorCode setTagData(Tag th, const BitRefLevel bit = BitRefLevel());

  /**
   * \brief set coords from tag
   * @param  th tag handle
   * @return    error code
   */
  MoFEMErrorCode setCoords(Tag th, const BitRefLevel bit = BitRefLevel(),
                           const BitRefLevel mask = BitRefLevel().set());

  inline const Range &getVolume() const { return vOlume; }
  inline const Range &getSurface() const { return sUrface; }

  inline const Range &getCutEdges() const { return cutEdges; }
  inline const Range &getCutVolumes() const { return cutVolumes; }
  inline const Range &getNewCutVolumes() const { return cutNewVolumes; }
  inline const Range &getNewCutSurfaces() const { return cutNewSurfaces; }
  inline const Range &getNewCutVertices() const { return cutNewVertices; }
  inline const Range &projectZeroDistanceEnts() const {
    return zeroDistanceEnts;
  }

  inline const Range &getTrimEdges() const { return trimEdges; }
  inline const Range &getNewTrimVolumes() const { return trimNewVolumes; }
  inline const Range &getNewTrimSurfaces() const { return trimNewSurfaces; }
  inline const Range &getNewTrimVertices() const { return trimNewVertices; }

  inline const Range &getMergedVolumes() const { return mergedVolumes; }
  inline const Range &getMergedSurfaces() const { return mergedSurfaces; }

  inline const Range &getTetgenSurfaces() const { return tetgenSurfaces; }

  MoFEMErrorCode saveCutEdges();

  MoFEMErrorCode saveTrimEdges();

  inline boost::shared_ptr<OrientedBoxTreeTool> &getTreeSurfPtr() {
    return treeSurfPtr;
  }

  MoFEMErrorCode clearMap();

private:
  Range sUrface;
  Range vOlume;

  boost::shared_ptr<OrientedBoxTreeTool> treeSurfPtr;
  EntityHandle rootSetSurf;

  Range cutEdges;
  Range cutVolumes;
  Range cutNewVolumes;
  Range cutNewSurfaces;
  Range zeroDistanceEnts;
  Range zeroDistanceVerts;
  Range cutNewVertices;

  Range trimNewVolumes;
  Range trimNewVertices;
  Range trimNewSurfaces;
  Range trimEdges;

  Range mergedVolumes;
  Range mergedSurfaces;

  Range tetgenSurfaces;

  struct TreeData {
    double dIst;
    double lEngth;
    VectorDouble3 unitRayDir;
    VectorDouble3 rayPoint;
  };

  map<EntityHandle, TreeData> edgesToCut;
  map<EntityHandle, TreeData> verticesOnCutEdges;
  map<EntityHandle, TreeData> edgesToTrim;
  map<EntityHandle, TreeData> verticesOnTrimEdges;

#ifdef WITH_TETGEN

  map<EntityHandle, unsigned long> moabTetGenMap;
  map<unsigned long, EntityHandle> tetGenMoabMap;
  boost::ptr_vector<tetgenio> tetGenData;

#endif

  MoFEMErrorCode getRayForEdge(const EntityHandle ent, VectorAdaptor &ray_point,
                               VectorAdaptor &unit_ray_dir,
                               double &ray_length) const;

  // /**
  //  * Find if segment in on the plain
  //  * @param  s0 segment first point
  //  * @param  s1 segment second point
  //  * @param  x0 point on the plain
  //  * @param  n  normal on the plain
  //  * @param  s  intersect point
  //  * @return    1 - intersect, 2 - segment on the plain, 0 - no intersect
  //  */
  // int segmentPlane(
  //   VectorAdaptor s0,
  //   VectorAdaptor s1,
  //   VectorAdaptor x0,
  //   VectorAdaptor n,
  //   double &s
  // ) const;

  double aveLength; ///< Average edge length
  double maxLength; ///< Maximal edge length
};

class Population;

// struct DataForCut {
//   CutWithGA *Cut_mesh;
//   int &fractionLevel;
//   BitRefLevel &Bit0;
//   BitRefLevel &Bit1;
//   BitRefLevel &Bit2;
//   BitRefLevel &Bit3;
//   Tag &tAg;
//   Range &Edges;
//   Range &Nodes;
//   double &minEdgeL;

//   DataForCut(CutWithGA *cut_mesh, int &fraction_level,BitRefLevel &bit0, BitRefLevel &bit1,
//              BitRefLevel &bit2, BitRefLevel &bit3, Tag &tag, Range &edges,
//              Range &nodes, double &min_edge_l)
//       : Cut_mesh(cut_mesh), fractionLevel(fraction_level), Bit0(bit0),Bit1(bit1),
//         Bit2(bit2), Bit3(bit3), tAg(tag), Edges(edges), Nodes(nodes),
//         minEdgeL(min_edge_l) {}
// };

random_device rd;  // only used once to initialise (seed) engine
mt19937 rng(rd()); // random-number engine used (Mersenne-Twister in this case)

int random(int min, int max) // range : [min, max)
{
  uniform_int_distribution<int> uni(min, max); // guaranteed unbiased

  return uni(rng);
}

double random_d() // range : [min, max)
{

  //   tolCut = 1e-4;
  // tolCutClose = 1e-2;
  // tolTrim = 1e-3;
  // tolTrimClose = 1e-3;

  // uniform_real_distribution<double> uni(0.0001, 0.1); // guaranteed unbiased
  vector<double> poss{0.01, 0.001, 0.0001};

  int mult = random(1, 4);
  int idx = random(0, poss.size() - 1);


  return 2 * mult * poss[idx];
  // return uni(rng);
}

double random(double max) // range : [min, max)
{
  uniform_real_distribution<double> uni(0, max); // guaranteed unbiased

  return uni(rng);
}

inline const int randChar() { return random(32, 126); }
class DNA {
  const int sIze;
  vector<double> gEnes;
  double fItness;
  // string &tArget;
  // shared_ptr<DataForCut> dataC_ptr;

public:
  // static double bestFitness;
  // generate random DNA //TODO: target size could be determined during the
  // compile time
  DNA(const int size /*, shared_ptr<DataForCut> data_c*/)
      : sIze(size), gEnes(size), fItness(0) {
    // gEnes.resize(size);
    // dataC_ptr = data_c;
    generate(gEnes.begin(), gEnes.end(), [&] { return random_d(); });
    // for (auto &g : gEnes)
    //   g = randChar();
  };
  const double &getFitness() { return fItness; }
  // convert gene (string of numbers) to
  string printDNA() {
    // string phrase;
    std::ostringstream stm;
    for (auto &i : gEnes) {
      stm << i << ", ";
    }
    stm << "\n";
    return stm.str();
  }

  void calcFitness() {
    double fitness = 0;
    const auto &vec = gEnes;
    // for(auto &v : vec)
    //   v *= dataC_ptr->minEdgeL;
    try {

      ierr = cutMeshRout(vec[0], vec[1], vec[2], vec[3], fitness);
      double check = fitness;
      
      // Tag tag = dataC_ptr->tAg;
      // ierr = dataC_ptr->Cut_mesh->CutWithGA::cutTrimAndMerge(
      //     dataC_ptr->fractionLevel, dataC_ptr->Bit1, dataC_ptr->Bit2,
      //     dataC_ptr->Bit3, dataC_ptr->tAg, 1e-2, 1e-1, 1e-1, 1e-2, fitness,
      //     dataC_ptr->Edges, dataC_ptr->Nodes, false, false);
      // check = fitness;
      // ierr = dataC_ptr->Cut_mesh->CutWithGA::cutTrimAndMerge(
      //     dataC_ptr->fractionLevel, dataC_ptr->Bit1, dataC_ptr->Bit2,
      //     dataC_ptr->Bit3, dataC_ptr->tAg, vec[0], vec[1], vec[2], vec[3],
      //     fitness, dataC_ptr->Edges, dataC_ptr->Nodes, false, false);
      if (ierr) {
        ierr = 0;
        fItness = 0;
      }

    } catch (MoFEMExceptionInitial const &ex) {
      cout << " here1 " << endl;
    } catch (MoFEMExceptionRepeat const &ex) {

      cout << " here2 " << endl;
    } catch (MoFEMException const &ex) {

      cout << " here3 " << endl;
    } catch (std::exception const &ex) {
      cout << " here4 " << endl;
    }

    // if (fitness < 0 || fitness != fitness)
    if (fitness != fitness)
      fitness = 0;
    fItness = fitness * fitness * fitness;
    // no no no no no no no no no no
    // fItness = (fItness + 1) * (fItness + 1);
  }

  DNA crossOver(DNA &partner) {
    // new child
    DNA child(this->sIze /*, this->dataC_ptr*/);
    // pick a midpoint TODO: play with it
    int midpoint = random(0, this->sIze);
    for (int i = 0; i != sIze; i++) {
      if (i > midpoint)
        // child.gEnes[i] = partner.gEnes[i];
      child.gEnes[i] = 0.5 * (this->gEnes[i] + partner.gEnes[i]); //AVERAGE //FIXME:
      // the genes
      else
        child.gEnes[i] = this->gEnes[i];
    }
    return child;
  }

  void mutate(const double &mutation_rate) {
    for (auto &gen : gEnes) {
      if (random(1.) < mutation_rate) {
        gen = random_d();
      }
    }
  }
};

// double DNA::bestFitness = 0;

class Population {

  const vector<double> Target;

  double mutationRate;
  int populationSize;
  int gEnerations;
  vector<DNA> pOpulation;
  vector<int> matingPool;
  bool is_finished;
  int perfectScore;
  double bestFitness;
  // shared_ptr<DataForCut> DataC;

public:
  Population(const vector<double> &target, /*shared_ptr<DataForCut> data_c,*/
             const int population_size = 10, const double &mutation_rate = 0.01)
      : Target(target),/* DataC(data_c),*/ mutationRate(mutation_rate),
        populationSize(population_size), gEnerations(0), is_finished(false),
        perfectScore(1), bestFitness(0) {
    int target_size = Target.size();
    pOpulation.reserve(population_size);
    for (size_t i = 0; i != populationSize; i++) {
      pOpulation.push_back(DNA(target_size/*,DataC*/));
    }
  }
  // Fill our fitness array with a value for every member of the population
  void calcFitness() {
    for (auto &pop : pOpulation) {
      pop.calcFitness();
    }
  }

  // Generate a mating pool
  // void initializeInterface(CutWithGA &cut_mesh, BitRefLevel &bit1,
  // BitRefLevel &bit2, Tag &tag, Range &edges, Range &nodes) {
  //   this->dataC = shared_ptr<DataForCut>(new DataForCut(cut_mesh, bit1, bit2,
  //   tag, edges, nodes));
  // }

  void generateNewPopulation() {

    double sum = 0;
    for (auto &&pop : pOpulation) {
      sum += pop.getFitness();
    }

    vector<DNA> new_population;
    new_population.reserve(populationSize);
    for (int i = 0; i != populationSize; ++i) {
      auto &partnerA = pickOne(sum); // pick DNA with respect to fitness
      auto &partnerB = pickOne(sum);
      auto child = partnerA.crossOver(partnerB);
      child.mutate(mutationRate);
      child.calcFitness();
      new_population.push_back(child);
    }
    pOpulation = move(new_population);
    ++gEnerations;
  }

  string getBest(double &fit) {
    double record = 0.;
    int idx = 0;
    int i = 0;
    for (auto &pop : pOpulation) {
      if (pop.getFitness() > record) {
        idx = i;
        record = pop.getFitness();
      }
      ++i;
    }

    if (record == perfectScore)
      is_finished = true;
    fit = pOpulation[idx].getFitness();
    return pOpulation[idx].printDNA();
  }

  const bool &finished() { return is_finished; }
  const int &getGenerations() { return gEnerations; }
  double getAverageFitness() {
    double tot = 0;

    for (auto &&pop : pOpulation) {
      tot += pop.getFitness();
    }
    return tot / populationSize;
  }

  const string printAll() {
    string everything;
    int displayLimit = populationSize < 25 ? populationSize : 25;

    for (size_t i = 0; i < displayLimit; i++) {
      everything += pOpulation[i].printDNA() + "\n";
    }
    return everything;
  }

  DNA &acceptReject(const double &max_fitness) {
    while (true) {
      int idx = random(0, populationSize - 1);
      DNA &partner = pOpulation[idx];
      double r = random(max_fitness);
      if (r < partner.getFitness())
        return partner;
    }
  }

  DNA &pickOne(const double &sum) {
    auto it = pOpulation.begin();
    double r = random(sum);
    while (r > 0) {
      r -= it->getFitness();
      ++it;
    }
    if (it != pOpulation.begin()) // FIXME:
      --it;
    return *it;
  }
};

MoFEMErrorCode CutWithGA::query_interface(const MOFEMuuid &uuid,
                                          UnknownInterface **iface) const {
  MoFEMFunctionBeginHot;
  *iface = NULL;
  if (uuid == IDD_MOFEMCutMesh) {
    *iface = const_cast<CutWithGA *>(this);
    MoFEMFunctionReturnHot(0);
  }
  SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "unknown interface");
  MoFEMFunctionReturnHot(0);
}

CutWithGA::CutWithGA(const Core &core) : cOre(const_cast<Core &>(core)) {
  lineSearchSteps = 10;
  nbMaxMergingCycles = 200;
  nbMaxTrimSearchIterations = 20;
}

MoFEMErrorCode CutWithGA::clearMap() {
  MoFEMFunctionBegin;
  treeSurfPtr.reset();
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::setSurface(const Range &surface) {
  MoFEMFunctionBeginHot;
  sUrface = surface;
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode CutWithGA::copySurface(const Range &surface, Tag th,
                                      double *shift, double *origin,
                                      double *transform,
                                      const std::string save_mesh) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;
  std::map<EntityHandle, EntityHandle> verts_map;
  for (Range::const_iterator tit = surface.begin(); tit != surface.end();
       tit++) {
    int num_nodes;
    const EntityHandle *conn;
    CHKERR moab.get_connectivity(*tit, conn, num_nodes, true);
    MatrixDouble coords(num_nodes, 3);
    if (th) {
      CHKERR moab.tag_get_data(th, conn, num_nodes, &coords(0, 0));
    } else {
      CHKERR moab.get_coords(conn, num_nodes, &coords(0, 0));
    }
    EntityHandle new_verts[num_nodes];
    for (int nn = 0; nn != num_nodes; nn++) {
      if (verts_map.find(conn[nn]) != verts_map.end()) {
        new_verts[nn] = verts_map[conn[nn]];
      } else {
        if (transform) {
          ublas::matrix_row<MatrixDouble> mr(coords, nn);
          if (origin) {
            VectorAdaptor vec_origin(
                3, ublas::shallow_array_adaptor<double>(3, origin));
            mr = mr - vec_origin;
          }
          MatrixAdaptor mat_transform = MatrixAdaptor(
              3, 3, ublas::shallow_array_adaptor<double>(9, transform));
          mr = prod(mat_transform, mr);
          if (origin) {
            VectorAdaptor vec_origin(
                3, ublas::shallow_array_adaptor<double>(3, origin));
            mr = mr + vec_origin;
          }
        }
        if (shift) {
          ublas::matrix_row<MatrixDouble> mr(coords, nn);
          VectorAdaptor vec_shift(
              3, ublas::shallow_array_adaptor<double>(3, shift));
          mr = mr + vec_shift;
        }
        CHKERR moab.create_vertex(&coords(nn, 0), new_verts[nn]);
        verts_map[conn[nn]] = new_verts[nn];
      }
    }
    EntityHandle ele;
    CHKERR moab.create_element(MBTRI, new_verts, num_nodes, ele);
    sUrface.insert(ele);
  }
  if (!save_mesh.empty()) {
    EntityHandle meshset;
    CHKERR m_field.get_moab().create_meshset(MESHSET_SET, meshset);
    CHKERR m_field.get_moab().add_entities(meshset, sUrface);
    CHKERR m_field.get_moab().write_file(save_mesh.c_str(), "VTK", "", &meshset,
                                         1);
    CHKERR m_field.get_moab().delete_entities(&meshset, 1);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::setVolume(const Range &volume) {
  MoFEMFunctionBeginHot;
  vOlume = volume;
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode CutWithGA::mergeSurface(const Range &surface) {
  MoFEMFunctionBeginHot;
  sUrface.merge(surface);
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode CutWithGA::mergeVolumes(const Range &volume) {
  MoFEMFunctionBeginHot;
  vOlume.merge(volume);
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode CutWithGA::buildTree() {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;
  treeSurfPtr = boost::shared_ptr<OrientedBoxTreeTool>(
      new OrientedBoxTreeTool(&moab, "ROOTSETSURF", true));
  CHKERR treeSurfPtr->build(sUrface, rootSetSurf);
  MoFEMFunctionReturn(0);
}

struct UpdateMeshsets {
  MoFEMErrorCode operator()(Core &core, const BitRefLevel &bit) const {
    MoFEMFunctionBeginHot;
    ierr = core.getInterface<MeshsetsManager>()
               ->updateAllMeshsetsByEntitiesChildren(bit);
    CHKERRG(ierr);

    MoFEMFunctionReturnHot(0);
  }
};

MoFEMErrorCode CutWithGA::cutAndTrim(
    const BitRefLevel &bit_level1, const BitRefLevel &bit_level2, Tag th,
    const double tol_cut, const double tol_cut_close, const double tol_trim,
    const double tol_trim_close, Range *fixed_edges, Range *corner_nodes,
    const bool update_meshsets, const bool debug) {
  CoreInterface &m_field = cOre;
  MoFEMFunctionBegin;

  // cut mesh
  CHKERR findEdgesToCut(fixed_edges, corner_nodes, tol_cut, QUIET, debug);
  CHKERR projectZeroDistanceEnts(fixed_edges, corner_nodes, tol_cut_close,
                                 QUIET, debug);
  CHKERR cutEdgesInMiddle(bit_level1, debug);
  if (fixed_edges) {
    CHKERR cOre.getInterface<BitRefManager>()->updateRange(*fixed_edges,
                                                           *fixed_edges);
  }
  if (corner_nodes) {
    CHKERR cOre.getInterface<BitRefManager>()->updateRange(*corner_nodes,
                                                           *corner_nodes);
  }
  if (update_meshsets) {
    CHKERR UpdateMeshsets()(cOre, bit_level1);
  }
  CHKERR moveMidNodesOnCutEdges(th);

  auto get_min_quality = [&m_field](const BitRefLevel bit, Tag th) {
    Range tets_level; // test at level
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit, BitRefLevel().set(), MBTET, tets_level);
    double min_q = 1;
    CHKERR m_field.getInterface<Tools>()->minTetsQuality(tets_level, min_q, th);
    return min_q;
  };

  // PetscPrintf(PETSC_COMM_WORLD, "Min quality cut %6.4g\n",
  //             get_min_quality(bit_level1, th));

  // trim mesh
  CHKERR findEdgesToTrim(fixed_edges, corner_nodes, th, tol_trim);
  CHKERR trimEdgesInTheMiddle(bit_level2, th, tol_trim_close, debug);
  if (fixed_edges) {
    CHKERR cOre.getInterface<BitRefManager>()->updateRange(*fixed_edges,
                                                           *fixed_edges);
  }
  if (corner_nodes) {
    CHKERR cOre.getInterface<BitRefManager>()->updateRange(*corner_nodes,
                                                           *corner_nodes);
  }
  if (update_meshsets) {
    CHKERR UpdateMeshsets()(cOre, bit_level2);
  }
  CHKERR moveMidNodesOnTrimmedEdges(th);

  // PetscPrintf(PETSC_COMM_WORLD, "Min quality trim %3.2g\n",
  //             get_min_quality(bit_level2, th));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::cutTrimAndMerge(
    const int fraction_level, const BitRefLevel &bit_level1,
    const BitRefLevel &bit_level2, const BitRefLevel &bit_level3, Tag th,
    const double tol_cut, const double tol_cut_close, const double tol_trim,
    const double tol_trim_close, double &fitness, Range fixed_edges,
    Range corner_nodes, const bool update_meshsets, const bool debug) {
  CoreInterface &m_field = cOre;
  MoFEMFunctionBegin;
  CHKERR cutAndTrim(bit_level1, bit_level2, th, tol_cut, tol_cut_close,
                    tol_trim, tol_trim_close, &fixed_edges, &corner_nodes,
                    update_meshsets, debug);

  CHKERR mergeBadEdges(fraction_level, bit_level2, bit_level1, bit_level3,
                       getNewTrimSurfaces(), fixed_edges, corner_nodes, th,
                       update_meshsets, debug);
  // CHKERRABORT(PETSC_COMM_WORLD, ierr);
  // CHKERR removePathologicalFrontTris(bit_level3,
  //  const_cast<Range &>(getMergedSurfaces()));

  auto get_min_quality = [&m_field, debug](const BitRefLevel bit, Tag th) {
    Range tets_level; // test at level
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit, BitRefLevel().set(), MBTET, tets_level);
    double min_q = 1;
    CHKERR m_field.getInterface<Tools>()->minTetsQuality(tets_level, min_q, th);
    if (min_q < 0 && debug) {
      CHKERR m_field.getInterface<Tools>()->writeTetsWithQuality(
          "negative_tets.vtk", "VTK", "", tets_level, th);
    }
    return min_q;
  };

  // PetscPrintf(PETSC_COMM_WORLD, "Min quality node merge %6.4g\n",
  //             get_min_quality(bit_level1, th));
  fitness = get_min_quality(bit_level1, th);
  // CHKERR
  // cOre.getInterface<BitRefManager>()->updateRange(fixed_edges, fixed_edges);
  // CHKERR cOre.getInterface<BitRefManager>()->updateRange(corner_nodes,
  //                                                        corner_nodes);

  MoFEMFunctionReturn(0);
}

static double get_ave_edge_length(const EntityHandle ent,
                                  const Range &vol_edges,
                                  moab::Interface &moab) {
  Range adj_edges;
  if (moab.type_from_handle(ent) == MBVERTEX) {
    CHKERR moab.get_adjacencies(&ent, 1, 1, false, adj_edges);
  } else {
    Range nodes;
    CHKERR moab.get_connectivity(&ent, 1, nodes);
    CHKERR moab.get_adjacencies(&ent, 1, 1, false, adj_edges,
                                moab::Interface::UNION);
  }
  adj_edges = intersect(adj_edges, vol_edges);
  double ave_l = 0;
  for (auto e : adj_edges) {
    int num_nodes;
    const EntityHandle *conn;
    CHKERR moab.get_connectivity(e, conn, num_nodes, true);
    VectorDouble6 coords(6);
    CHKERR moab.get_coords(conn, num_nodes, &coords[0]);
    FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_n0(
        &coords[0], &coords[1], &coords[2]);
    FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_n1(
        &coords[3], &coords[4], &coords[5]);
    FTensor::Index<'i', 3> i;
    t_n0(i) -= t_n1(i);
    ave_l += sqrt(t_n0(i) * t_n0(i));
  }
  return ave_l / adj_edges.size();
};

MoFEMErrorCode CutWithGA::findEdgesToCut(Range *fixed_edges,
                                         Range *corner_nodes,
                                         const double low_tol, int verb,
                                         const bool debug) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;

  edgesToCut.clear();
  cutEdges.clear();

  zeroDistanceVerts.clear();
  zeroDistanceEnts.clear();
  verticesOnCutEdges.clear();

  double ray_length;
  double ray_point[3], unit_ray_dir[3];
  VectorAdaptor vec_unit_ray_dir(
      3, ublas::shallow_array_adaptor<double>(3, unit_ray_dir));
  VectorAdaptor vec_ray_point(
      3, ublas::shallow_array_adaptor<double>(3, ray_point));

  Tag th_dist;
  rval = moab.tag_get_handle("DIST", th_dist);
  if (rval == MB_SUCCESS) {
    CHKERR moab.tag_delete(th_dist);
  } else {
    rval = MB_SUCCESS;
  }
  Tag th_dist_normal;
  rval = moab.tag_get_handle("DIST_NORMAL", th_dist_normal);
  if (rval == MB_SUCCESS) {
    CHKERR moab.tag_delete(th_dist_normal);
  } else {
    rval = MB_SUCCESS;
  }

  double def_val[] = {0, 0, 0};
  CHKERR moab.tag_get_handle("DIST", 1, MB_TYPE_DOUBLE, th_dist,
                             MB_TAG_CREAT | MB_TAG_SPARSE, def_val);
  CHKERR moab.tag_get_handle("DIST_NORMAL", 3, MB_TYPE_DOUBLE, th_dist_normal,
                             MB_TAG_CREAT | MB_TAG_SPARSE, def_val);

  Range vol_vertices;
  CHKERR moab.get_connectivity(vOlume, vol_vertices, true);
  for (auto v : vol_vertices) {
    VectorDouble3 point_in(3);
    CHKERR moab.get_coords(&v, 1, &point_in[0]);
    VectorDouble3 point_out(3);
    EntityHandle facets_out;
    CHKERR treeSurfPtr->closest_to_location(&point_in[0], rootSetSurf,
                                            &point_out[0], facets_out);
    VectorDouble3 n(3);
    Util::normal(&moab, facets_out, n[0], n[1], n[2]);
    VectorDouble3 delta = point_out - point_in;
    double dist = norm_2(delta);
    VectorDouble3 dist_normal = inner_prod(delta, n) * n;
    CHKERR moab.tag_set_data(th_dist, &v, 1, &dist);
    CHKERR moab.tag_set_data(th_dist_normal, &v, 1, &dist_normal[0]);
  }

  auto get_normal_dist = [](const double *normal) {
    FTensor::Tensor1<double, 3> t_n(normal[0], normal[1], normal[2]);
    FTensor::Index<'i', 3> i;
    return sqrt(t_n(i) * t_n(i));
  };

  auto get_edge_crossed = [&moab, get_normal_dist,
                           th_dist_normal](EntityHandle v0, EntityHandle v1) {
    VectorDouble3 dist_normal0(3);
    CHKERR moab.tag_get_data(th_dist_normal, &v0, 1, &dist_normal0[0]);
    VectorDouble3 dist_normal1(3);
    CHKERR moab.tag_get_data(th_dist_normal, &v1, 1, &dist_normal1[0]);
    return (inner_prod(dist_normal0, dist_normal1) < 0);
  };

  auto get_normal_dist_from_conn = [&moab, get_normal_dist,
                                    th_dist_normal](EntityHandle v) {
    double dist_normal[3];
    CHKERR moab.tag_get_data(th_dist_normal, &v, 1, dist_normal);
    return get_normal_dist(dist_normal);
  };

  auto project_node = [this, &moab, th_dist_normal](const EntityHandle v) {
    MoFEMFunctionBegin;
    VectorDouble3 dist_normal(3);
    rval = moab.tag_get_data(th_dist_normal, &v, 1, &dist_normal[0]);
    VectorDouble3 s0(3);
    CHKERR moab.get_coords(&v, 1, &s0[0]);
    double dist = norm_2(dist_normal);
    verticesOnCutEdges[v].dIst = dist;
    verticesOnCutEdges[v].lEngth = dist;
    verticesOnCutEdges[v].unitRayDir =
        dist > 0 ? dist_normal / dist : dist_normal;
    verticesOnCutEdges[v].rayPoint = s0;
    MoFEMFunctionReturn(0);
  };

  auto not_project_node = [this, &moab](const EntityHandle v) {
    MoFEMFunctionBegin;
    VectorDouble3 s0(3);
    CHKERR moab.get_coords(&v, 1, &s0[0]);
    verticesOnCutEdges[v].dIst = 0;
    verticesOnCutEdges[v].lEngth = 0;
    verticesOnCutEdges[v].unitRayDir = s0;
    verticesOnCutEdges[v].rayPoint = s0;
    MoFEMFunctionReturn(0);
  };

  auto check_if_is_on_fixed_edge = [fixed_edges](const EntityHandle e) {
    if (fixed_edges) {
      if (fixed_edges->find(e) != fixed_edges->end()) {
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  };

  auto check_if_is_on_cornet_node = [corner_nodes](const EntityHandle v) {
    if (corner_nodes) {
      if (corner_nodes->find(v) != corner_nodes->end()) {
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  };

  Range vol_edges;
  CHKERR moab.get_adjacencies(vOlume, 1, true, vol_edges,
                              moab::Interface::UNION);

  aveLength = 0;
  maxLength = 0;
  int nb_ave_length = 0;
  for (auto e : vol_edges) {
    int num_nodes;
    const EntityHandle *conn;
    CHKERR moab.get_connectivity(e, conn, num_nodes, true);
    double dist[num_nodes];
    CHKERR moab.tag_get_data(th_dist, conn, num_nodes, dist);
    CHKERR getRayForEdge(e, vec_ray_point, vec_unit_ray_dir, ray_length);
    const double tol = ray_length * low_tol;
    if (get_edge_crossed(conn[0], conn[1])) {
      std::vector<double> distances_out;
      std::vector<EntityHandle> facets_out;
      CHKERR treeSurfPtr->ray_intersect_triangles(distances_out, facets_out,
                                                  rootSetSurf, tol, ray_point,
                                                  unit_ray_dir, &ray_length);
      if (!distances_out.empty()) {
        const auto dist_ptr =
            std::min_element(distances_out.begin(), distances_out.end());
        const double dist = *dist_ptr;
        if (dist <= ray_length) {
          aveLength += ray_length;
          maxLength = fmax(maxLength, ray_length);
          nb_ave_length++;
          edgesToCut[e].dIst = dist;
          edgesToCut[e].lEngth = ray_length;
          edgesToCut[e].unitRayDir = vec_unit_ray_dir;
          edgesToCut[e].rayPoint = vec_ray_point;
          cutEdges.insert(e);
        }
      }
    }

    if (fabs(dist[0]) < tol && fabs(dist[1]) < tol) {
      aveLength += ray_length;
      maxLength = fmax(maxLength, ray_length);
      if (check_if_is_on_fixed_edge(e)) {
        CHKERR not_project_node(conn[0]);
        CHKERR not_project_node(conn[1]);
      } else {
        CHKERR project_node(conn[0]);
        CHKERR project_node(conn[1]);
      }
      zeroDistanceEnts.insert(e);
    }
  }
  aveLength /= nb_ave_length;

  Range cut_edges_verts;
  CHKERR moab.get_connectivity(unite(cutEdges, zeroDistanceEnts),
                               cut_edges_verts, true);
  vol_vertices = subtract(vol_vertices, cut_edges_verts);

  for (auto v : vol_vertices) {
    double dist;
    CHKERR moab.tag_get_data(th_dist, &v, 1, &dist);
    const double tol = get_ave_edge_length(v, vol_edges, moab) * low_tol;
    if (fabs(dist) < tol) {

      if (check_if_is_on_cornet_node(v)) {
        CHKERR not_project_node(v);
      } else {
        CHKERR project_node(v);
      }

      zeroDistanceVerts.insert(v);
    }
  }

  cutVolumes.clear();
  // take all volumes adjacent to cut edges
  CHKERR moab.get_adjacencies(cutEdges, 3, false, cutVolumes,
                              moab::Interface::UNION);
  CHKERR moab.get_adjacencies(zeroDistanceVerts, 3, false, cutVolumes,
                              moab::Interface::UNION);
  {
    Range verts;
    CHKERR moab.get_connectivity(unite(cutEdges, zeroDistanceEnts), verts,
                                 true);
    CHKERR moab.get_adjacencies(verts, 3, false, cutVolumes,
                                moab::Interface::UNION);
  }
  cutVolumes = intersect(cutVolumes, vOlume);

  // get edges on the cut volumes
  Range edges;
  CHKERR moab.get_adjacencies(cutVolumes, 1, false, edges,
                              moab::Interface::UNION);
  edges = subtract(edges, cutEdges);

  // add to cut set edges which are cutted by extension of cutting surface
  for (auto e : edges) {
    int num_nodes;
    const EntityHandle *conn;
    CHKERR moab.get_connectivity(e, conn, num_nodes, true);
    const double tol = get_ave_edge_length(e, vol_edges, moab) * low_tol;
    double dist_normal[2];
    dist_normal[0] = get_normal_dist_from_conn(conn[0]);
    dist_normal[1] = get_normal_dist_from_conn(conn[1]);
    if (get_edge_crossed(conn[0], conn[1])) {
      CHKERR getRayForEdge(e, vec_ray_point, vec_unit_ray_dir, ray_length);
      double s =
          fabs(dist_normal[0]) / (fabs(dist_normal[0]) + fabs(dist_normal[1]));
      edgesToCut[e].dIst = s * ray_length;
      edgesToCut[e].lEngth = ray_length;
      edgesToCut[e].unitRayDir = vec_unit_ray_dir;
      edgesToCut[e].rayPoint = vec_ray_point;
      cutEdges.insert(e);
    } else if (fabs(dist_normal[0]) < tol && fabs(dist_normal[1]) < tol) {
      if (check_if_is_on_fixed_edge(e)) {
        CHKERR not_project_node(conn[0]);
        CHKERR not_project_node(conn[1]);
      } else {
        CHKERR project_node(conn[0]);
        CHKERR project_node(conn[1]);
      }
      zeroDistanceEnts.insert(e);
    }
  }

  CHKERR moab.get_adjacencies(cutVolumes, 1, false, edges,
                              moab::Interface::UNION);
  Range add_verts;
  CHKERR moab.get_connectivity(edges, add_verts, true);
  add_verts = subtract(add_verts, zeroDistanceVerts);
  CHKERR moab.get_connectivity(unite(cutEdges, zeroDistanceEnts),
                               cut_edges_verts, true);
  add_verts = subtract(add_verts, cut_edges_verts);

  for (auto v : add_verts) {
    double dist_normal = get_normal_dist_from_conn(v);
    const double tol = get_ave_edge_length(v, vol_edges, moab) * low_tol;
    if (fabs(dist_normal) < tol) {

      if (check_if_is_on_cornet_node(v)) {
        CHKERR not_project_node(v);
      } else {
        CHKERR project_node(v);
      }

      zeroDistanceVerts.insert(v);
    }
  }

  for (auto f : zeroDistanceEnts) {
    int num_nodes;
    const EntityHandle *conn;
    CHKERR moab.get_connectivity(f, conn, num_nodes, true);
    Range adj_edges;
    CHKERR moab.get_adjacencies(conn, num_nodes, 1, false, adj_edges,
                                moab::Interface::UNION);
    for (auto e : adj_edges) {
      cutEdges.erase(e);
      edgesToCut.erase(e);
    }
  }

  for (auto v : zeroDistanceVerts) {
    Range adj_edges;
    CHKERR moab.get_adjacencies(&v, 1, 1, false, adj_edges,
                                moab::Interface::UNION);
    for (auto e : adj_edges) {
      cutEdges.erase(e);
      edgesToCut.erase(e);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::projectZeroDistanceEnts(Range *fixed_edges,
                                                  Range *corner_nodes,
                                                  const double low_tol,
                                                  const int verb,
                                                  const bool debug) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;

  // Get entities on body skin
  Skinner skin(&moab);
  Range tets_skin;
  rval = skin.find_skin(0, vOlume, false, tets_skin);
  Range tets_skin_edges;
  CHKERR moab.get_adjacencies(tets_skin, 1, false, tets_skin_edges,
                              moab::Interface::UNION);
  Range tets_skin_verts;
  CHKERR moab.get_connectivity(tets_skin, tets_skin_verts, true);

  // Get entities in volume
  Range vol_faces, vol_edges, vol_nodes;
  CHKERR moab.get_adjacencies(vOlume, 2, false, vol_faces,
                              moab::Interface::UNION);
  CHKERR moab.get_adjacencies(vOlume, 1, false, vol_edges,
                              moab::Interface::UNION);
  CHKERR moab.get_connectivity(vOlume, vol_nodes, true);

  // Get nodes on cut edges
  Range cut_edge_verts;
  CHKERR moab.get_connectivity(cutEdges, cut_edge_verts, true);

  Range fixed_edges_nodes;
  if (fixed_edges) {
    CHKERR moab.get_connectivity(*fixed_edges, fixed_edges_nodes, true);
  }

  // Get faces and edges
  Range cut_edges_faces;
  CHKERR moab.get_adjacencies(cut_edge_verts, 2, true, cut_edges_faces,
                              moab::Interface::UNION);
  cut_edges_faces = intersect(cut_edges_faces, vol_faces);
  Range cut_edges_faces_verts;
  CHKERR moab.get_connectivity(cut_edges_faces, cut_edges_faces_verts, true);
  cut_edges_faces_verts = subtract(cut_edges_faces_verts, cut_edge_verts);
  Range to_remove_cut_edges_faces;
  CHKERR moab.get_adjacencies(cut_edges_faces_verts, 2, true,
                              to_remove_cut_edges_faces,
                              moab::Interface::UNION);
  cut_edges_faces = subtract(cut_edges_faces, to_remove_cut_edges_faces);
  cut_edges_faces.merge(cutEdges);

  Tag th_dist_normal;
  CHKERR moab.tag_get_handle("DIST_NORMAL", th_dist_normal);

  auto get_quality_change =
      [this, &m_field,
       &moab](const Range &adj_tets,
              map<EntityHandle, TreeData> vertices_on_cut_edges) {
        vertices_on_cut_edges.insert(verticesOnCutEdges.begin(),
                                     verticesOnCutEdges.end());
        double q0 = 1;
        CHKERR m_field.getInterface<Tools>()->minTetsQuality(adj_tets, q0);
        double q = 1;
        for (auto t : adj_tets) {
          int num_nodes;
          const EntityHandle *conn;
          CHKERR m_field.get_moab().get_connectivity(t, conn, num_nodes, true);
          VectorDouble12 coords(12);
          CHKERR moab.get_coords(conn, num_nodes, &coords[0]);
          // cerr << coords << endl;
          for (int n = 0; n != 4; ++n) {
            bool ray_found = false;
            auto mit = vertices_on_cut_edges.find(conn[n]);
            if (mit != vertices_on_cut_edges.end()) {
              ray_found = true;
            }
            if (ray_found) {
              auto n_coords = getVectorAdaptor(&coords[3 * n], 3);
              double dist = mit->second.dIst;
              noalias(n_coords) =
                  mit->second.rayPoint + dist * mit->second.unitRayDir;
            }
          }
          q = std::min(q, Tools::volumeLengthQuality(&coords[0]));
        }
        if (std::isnormal(q))
          return q / q0;
        else
          return -1.;
      };

  auto get_conn = [&moab](const EntityHandle &e, const EntityHandle *&conn,
                          int &num_nodes) {
    MoFEMFunctionBegin;
    EntityType type = moab.type_from_handle(e);
    if (type == MBVERTEX) {
      conn = &e;
      num_nodes = 1;
    } else {
      CHKERR moab.get_connectivity(e, conn, num_nodes, true);
    }
    MoFEMFunctionReturn(0);
  };

  auto get_normal_dist = [](const double *normal) {
    FTensor::Tensor1<double, 3> t_n(normal[0], normal[1], normal[2]);
    FTensor::Index<'i', 3> i;
    return sqrt(t_n(i) * t_n(i));
  };

  auto get_normal_dist_from_conn = [&moab, get_normal_dist,
                                    th_dist_normal](EntityHandle v) {
    double dist_normal[3];
    CHKERR moab.tag_get_data(th_dist_normal, &v, 1, dist_normal);
    return get_normal_dist(dist_normal);
  };

  auto project_node = [&moab, th_dist_normal](
                          const EntityHandle v,
                          map<EntityHandle, TreeData> &vertices_on_cut_edges) {
    MoFEMFunctionBegin;
    VectorDouble3 dist_normal(3);
    rval = moab.tag_get_data(th_dist_normal, &v, 1, &dist_normal[0]);
    VectorDouble3 s0(3);
    CHKERR moab.get_coords(&v, 1, &s0[0]);
    double dist = norm_2(dist_normal);
    vertices_on_cut_edges[v].dIst = dist;
    vertices_on_cut_edges[v].lEngth = dist;
    vertices_on_cut_edges[v].unitRayDir =
        dist > 0 ? dist_normal / dist : dist_normal;
    vertices_on_cut_edges[v].rayPoint = s0;
    MoFEMFunctionReturn(0);
  };

  for (int d = 2; d >= 0; --d) {

    Range ents;
    if (d > 0)
      ents = cut_edges_faces.subset_by_dimension(d);
    else
      ents = cut_edge_verts;

    // make list of entities
    multimap<double, EntityHandle> ents_to_check;
    for (auto f : ents) {
      int num_nodes;
      const EntityHandle *conn;
      CHKERR get_conn(f, conn, num_nodes);
      VectorDouble3 dist(3);
      for (int n = 0; n != num_nodes; ++n) {
        dist[n] = get_normal_dist_from_conn(conn[n]);
      }
      double max_dist = 0;
      for (int n = 0; n != num_nodes; ++n) {
        max_dist = std::max(max_dist, fabs(dist[n]));
      }
      if (max_dist < low_tol * get_ave_edge_length(f, vol_edges, moab)) {
        ents_to_check.insert(std::pair<double, EntityHandle>(max_dist, f));
      }
    }

    double ray_point[3], unit_ray_dir[3];
    VectorAdaptor vec_unit_ray_dir(
        3, ublas::shallow_array_adaptor<double>(3, unit_ray_dir));
    VectorAdaptor vec_ray_point(
        3, ublas::shallow_array_adaptor<double>(3, ray_point));

    for (auto m : ents_to_check) {

      EntityHandle f = m.second;

      int num_nodes;
      const EntityHandle *conn;
      CHKERR get_conn(f, conn, num_nodes);
      VectorDouble9 coords(9);
      CHKERR moab.get_coords(conn, num_nodes, &coords[0]);

      Range adj_tets;
      CHKERR moab.get_adjacencies(conn, num_nodes, 3, false, adj_tets,
                                  moab::Interface::UNION);
      adj_tets = intersect(adj_tets, vOlume);

      map<EntityHandle, TreeData> vertices_on_cut_edges;
      for (int n = 0; n != num_nodes; ++n) {
        const EntityHandle node = conn[n];
        CHKERR project_node(node, vertices_on_cut_edges);
      }
      if (static_cast<int>(vertices_on_cut_edges.size()) != num_nodes) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "Data inconsistency");
      }

      double q = get_quality_change(adj_tets, vertices_on_cut_edges);
      if (q > 0.75) {
        bool check_q_again = false;
        for (auto &m : vertices_on_cut_edges) {
          EntityHandle node = m.first;
          if (tets_skin_verts.find(node) != tets_skin_verts.end()) {

            check_q_again = true;

            // check if node is at the corner
            bool zero_disp_node = false;
            if (corner_nodes) {
              if (corner_nodes->find(node) != corner_nodes->end()) {
                zero_disp_node = true;
              }
            }

            // check node is on the fixed edge
            Range adj_edges;
            CHKERR moab.get_adjacencies(&node, 1, 1, false, adj_edges);
            adj_edges = intersect(adj_edges, tets_skin_edges);
            if (fixed_edges) {
              Range e;
              // check if node is on fixed edge
              e = intersect(adj_edges, *fixed_edges);
              if (!e.empty()) {
                adj_edges.swap(e);
              }
              // check if split edge is fixed edge
              e = intersect(adj_edges, cutEdges);
              if (!e.empty()) {
                adj_edges.swap(e);
              } else {
                zero_disp_node = true;
              }
            }

            VectorDouble3 s0(3);
            CHKERR moab.get_coords(&node, 1, &s0[0]);

            if (zero_disp_node) {
              VectorDouble3 z(3);
              z.clear();
              m.second.dIst = 0;
              m.second.lEngth = 0;
              m.second.unitRayDir = z;
              m.second.rayPoint = s0;
            } else {
              if (adj_edges.empty()) {
                SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                        "Data inconsistency");
              }
              for (auto e : adj_edges) {
                if (edgesToCut.find(e) != edgesToCut.end()) {
                  auto d = edgesToCut.at(e);
                  VectorDouble3 new_pos = d.rayPoint + d.dIst * d.unitRayDir;
                  VectorDouble3 ray = new_pos - s0;
                  double dist0 = norm_2(ray);
                  m.second.dIst = dist0;
                  m.second.lEngth = dist0;
                  m.second.unitRayDir = dist0 > 0 ? ray / dist0 : ray;
                  m.second.rayPoint = s0;
                  break;
                } else {
                  SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                          "Data inconsistency");
                }
              }
            }
          }
        }

        if (check_q_again) {
          q = get_quality_change(adj_tets, vertices_on_cut_edges);
        }
        if (q > 0.75) {
          verticesOnCutEdges.insert(vertices_on_cut_edges.begin(),
                                    vertices_on_cut_edges.end());
          EntityHandle type = moab.type_from_handle(f);
          if (type == MBVERTEX) {
            zeroDistanceVerts.insert(f);
          } else {
            zeroDistanceEnts.insert(f);
          }
        }
      }
    }
  }

  for (auto f : unite(zeroDistanceEnts, zeroDistanceVerts)) {
    int num_nodes;
    const EntityHandle *conn;
    CHKERR get_conn(f, conn, num_nodes);
    Range adj_edges;
    CHKERR moab.get_adjacencies(conn, num_nodes, 1, false, adj_edges,
                                moab::Interface::UNION);
    for (auto e : adj_edges) {
      cutEdges.erase(e);
      edgesToCut.erase(e);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::cutEdgesInMiddle(const BitRefLevel bit,
                                           const bool debug) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MeshRefinement *refiner;
  const RefEntity_multiIndex *ref_ents_ptr;
  MoFEMFunctionBegin;
  if (cutEdges.size() != edgesToCut.size()) {
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Data inconsistency");
  }
  CHKERR m_field.getInterface(refiner);
  CHKERR m_field.get_ref_ents(&ref_ents_ptr);
  CHKERR refiner->add_verices_in_the_middel_of_edges(cutEdges, bit);
  CHKERR refiner->refine_TET(vOlume, bit, false, QUIET,
                             debug ? &cutEdges : NULL);
  cutNewVolumes.clear();
  CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      bit, BitRefLevel().set(), MBTET, cutNewVolumes);
  cutNewSurfaces.clear();
  CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      bit, BitRefLevel().set(), MBTRI, cutNewSurfaces);
  // Find new vertices on cut edges
  cutNewVertices.clear();
  CHKERR moab.get_connectivity(zeroDistanceEnts, cutNewVertices, true);
  cutNewVertices.merge(zeroDistanceVerts);
  for (map<EntityHandle, TreeData>::iterator mit = edgesToCut.begin();
       mit != edgesToCut.end(); ++mit) {
    RefEntity_multiIndex::index<
        Composite_ParentEnt_And_EntType_mi_tag>::type::iterator vit =
        ref_ents_ptr->get<Composite_ParentEnt_And_EntType_mi_tag>().find(
            boost::make_tuple(MBVERTEX, mit->first));
    if (vit ==
        ref_ents_ptr->get<Composite_ParentEnt_And_EntType_mi_tag>().end()) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "No vertex on cut edges, that make no sense");
    }
    const boost::shared_ptr<RefEntity> &ref_ent = *vit;
    if ((ref_ent->getBitRefLevel() & bit).any()) {
      EntityHandle vert = ref_ent->getRefEnt();
      cutNewVertices.insert(vert);
      verticesOnCutEdges[vert] = mit->second;
    } else {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "Vertex has wrong bit ref level");
    }
  }
  // Add zero distance entities faces
  Range tets_skin;
  Skinner skin(&moab);
  CHKERR skin.find_skin(0, cutNewVolumes, false, tets_skin);
  cutNewSurfaces.merge(zeroDistanceEnts.subset_by_type(MBTRI));
  // At that point cutNewSurfaces has all newly created faces, now take all
  // nodes on those faces and subtract nodes on catted edges. Faces adjacent to
  // nodes which left are not part of surface.
  Range diff_verts;
  CHKERR moab.get_connectivity(unite(cutNewSurfaces, zeroDistanceEnts),
                               diff_verts, true);
  diff_verts = subtract(diff_verts, cutNewVertices);
  Range subtract_faces;
  CHKERR moab.get_adjacencies(diff_verts, 2, false, subtract_faces,
                              moab::Interface::UNION);
  cutNewSurfaces = subtract(cutNewSurfaces, unite(subtract_faces, tets_skin));
  cutNewVertices.clear();
  CHKERR moab.get_connectivity(cutNewSurfaces, cutNewVertices, true);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::moveMidNodesOnCutEdges(Tag th) {
  MoFEMFunctionBeginHot;

  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;

  // Range out_side_vertices;
  for (map<EntityHandle, TreeData>::iterator mit = verticesOnCutEdges.begin();
       mit != verticesOnCutEdges.end(); mit++) {
    double dist = mit->second.dIst;
    VectorDouble3 new_coors =
        mit->second.rayPoint + dist * mit->second.unitRayDir;
    if (th) {
      CHKERR moab.tag_set_data(th, &mit->first, 1, &new_coors[0]);
    } else {
      CHKERR moab.set_coords(&mit->first, 1, &new_coors[0]);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::moveMidNodesOnTrimmedEdges(Tag th) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;
  // Range out_side_vertices;
  for (map<EntityHandle, TreeData>::iterator mit = verticesOnTrimEdges.begin();
       mit != verticesOnTrimEdges.end(); mit++) {
    double dist = mit->second.dIst;
    // cout << s << " " << mit->second.dIst << " " << mit->second.lEngth <<
    // endl;
    VectorDouble3 new_coors =
        mit->second.rayPoint + dist * mit->second.unitRayDir;
    if (th) {
      CHKERR moab.tag_set_data(th, &mit->first, 1, &new_coors[0]);
    } else {
      CHKERR moab.set_coords(&mit->first, 1, &new_coors[0]);
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::findEdgesToTrim(Range *fixed_edges,
                                          Range *corner_nodes, Tag th,
                                          const double tol, int verb) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;

  // takes edges on body skin
  Skinner skin(&moab);
  Range tets_skin;
  CHKERR skin.find_skin(0, cutNewVolumes, false, tets_skin);
  // vertives on the skin
  Range tets_skin_verts;
  CHKERR moab.get_connectivity(tets_skin, tets_skin_verts, true);
  // edges on the skin
  Range tets_skin_edges;
  CHKERR moab.get_adjacencies(tets_skin, 1, false, tets_skin_edges,
                              moab::Interface::UNION);
  // get edges on new surface
  Range edges;
  CHKERR moab.get_adjacencies(cutNewSurfaces, 1, false, edges,
                              moab::Interface::UNION);

  Range cut_surface_edges_on_fixed_edges;
  if (fixed_edges) {
    cut_surface_edges_on_fixed_edges = intersect(edges, *fixed_edges);
  }
  Range cut_surface_edges_on_fixed_edges_verts;
  CHKERR moab.get_connectivity(cut_surface_edges_on_fixed_edges,
                               cut_surface_edges_on_fixed_edges_verts, true);
  Range fixed_edges_verts_on_corners;
  if (fixed_edges) {
    CHKERR moab.get_connectivity(*fixed_edges, fixed_edges_verts_on_corners,
                                 true);
  }
  fixed_edges_verts_on_corners = subtract(
      fixed_edges_verts_on_corners, cut_surface_edges_on_fixed_edges_verts);
  if (corner_nodes) {
    fixed_edges_verts_on_corners.merge(*corner_nodes);
  }

  // clear data ranges
  trimEdges.clear();
  edgesToTrim.clear();
  verticesOnTrimEdges.clear();
  trimNewVertices.clear();

  // iterate over entities on new cut surface
  std::multimap<double, std::pair<EntityHandle, EntityHandle>> verts_map;
  for (auto e : edges) {
    // Get edge connectivity and coords
    int num_nodes;
    const EntityHandle *conn;
    CHKERR moab.get_connectivity(e, conn, num_nodes, true);
    double coords[3 * num_nodes];
    if (th) {
      CHKERR moab.tag_get_data(th, conn, num_nodes, coords);
    } else {
      CHKERR moab.get_coords(conn, num_nodes, coords);
    }
    // Put edges coords into boost vectors
    auto get_s_adaptor = [&coords](const int n) {
      return VectorAdaptor(3,
                           ublas::shallow_array_adaptor<double>(3, &coords[n]));
    };
    VectorAdaptor s0 = get_s_adaptor(0);
    VectorAdaptor s1 = get_s_adaptor(3);
    // get edge length
    double length = norm_2(s1 - s0);

    // Find point on surface closet to surface
    auto get_closets_delta = [this, &moab](const VectorAdaptor &s) {
      VectorDouble3 p(3);
      EntityHandle facets_out;
      // find closet point on the surface from first node
      CHKERR treeSurfPtr->closest_to_location(&s[0], rootSetSurf, &p[0],
                                              facets_out);
      VectorDouble3 n(3);
      Util::normal(&moab, facets_out, n[0], n[1], n[2]);
      VectorDouble3 w = p - s;
      VectorDouble3 normal = inner_prod(w, n) * n;
      w -= normal;
      return w;
    };

    // Calculate deltas, i.e. vectors from edges to closet point on surface
    VectorDouble3 delta0(3), delta1(3);
    noalias(delta0) = get_closets_delta(s0);
    noalias(delta1) = get_closets_delta(s1);

    // moab.tag_set_data(th,&conn[0],1,&delta0[0]);
    // moab.tag_set_data(th,&conn[1],1,&delta1[0]);
    // Calculate distances
    double dist0 = norm_2(delta0);
    double dist1 = norm_2(delta1);
    double min_dist = fmin(dist0, dist1);
    double max_dist = fmax(dist0, dist1);

    // add edge to trim
    double dist;
    VectorDouble3 ray;
    VectorDouble3 trimmed_end;
    VectorDouble3 itersection_point;

    if (min_dist < 1e-6 * aveLength && max_dist >= 1e-6 * aveLength) {
      if (max_dist == dist0) {
        // move mid node in reference to node 0
        trimmed_end = s0;
        ray = s1 - trimmed_end;
      } else {
        // move node in reference to node 1
        trimmed_end = s1;
        ray = s0 - trimmed_end;
      }

      // Solve nonlinera problem of finding point on surface front
      auto closest_point_projection =
          [this, &moab](VectorDouble3 ray, VectorDouble3 trimmed_end,
                        const int max_it, const double tol) {
            VectorDouble3 n(3), w(3), normal(3);
            double length = norm_2(ray);
            ray /= length;
            for (int ii = 0; ii != max_it; ii++) {
              EntityHandle facets_out;
              VectorDouble3 point_out(3);
              treeSurfPtr->closest_to_location(&trimmed_end[0], rootSetSurf,
                                               &point_out[0], facets_out);
              Util::normal(&moab, facets_out, n[0], n[1], n[2]);
              noalias(w) = point_out - trimmed_end;
              noalias(normal) = inner_prod(w, n) * n;
              double s = inner_prod(ray, w - normal);
              trimmed_end += s * ray;
              // cerr << "s " << ii << " " << s << " " << norm_2(w) << endl;
              if ((s / length) < tol)
                break;
            }
            return trimmed_end;
          };

      itersection_point = closest_point_projection(
          ray, trimmed_end, nbMaxTrimSearchIterations, 1e-12);

      ray = itersection_point - trimmed_end;
      dist = norm_2(ray);

      if ((1 - dist / length) > 0) {

        // check if edges should be trimmed, i.e. if edge is trimmed at very
        // end simply move closed node rather than trim
        edgesToTrim[e].dIst = dist;
        edgesToTrim[e].lEngth = dist;
        edgesToTrim[e].unitRayDir = ray / dist;
        edgesToTrim[e].rayPoint = trimmed_end;
        trimEdges.insert(e);

        auto add_vert = [&verts_map, e](EntityHandle v, double dist) {
          verts_map.insert(
              std::pair<double, std::pair<EntityHandle, EntityHandle>>(
                  dist, std::pair<EntityHandle, EntityHandle>(v, e)));
        };

        double dist0_to_intersection =
            norm_2(itersection_point - s0) / aveLength;
        double dist1_to_intersection =
            norm_2(itersection_point - s1) / aveLength;
        if (dist0_to_intersection < dist1_to_intersection) {
          add_vert(conn[0], dist0_to_intersection);
        } else {
          add_vert(conn[1], dist1_to_intersection);
        }
      }
    }
  }

  // EntityHandle meshset;
  // CHKERR moab.create_meshset(MESHSET_SET, meshset);
  // Tag th_aaa;
  // double def_val[] = {0,0,0};
  // CHKERR moab.tag_get_handle("AAAA", 3, MB_TYPE_DOUBLE, th_aaa,
  //                            MB_TAG_CREAT | MB_TAG_SPARSE, def_val);
  // Range all_nodes;
  // CHKERR moab.get_entities_by_type(0,MBVERTEX,all_nodes);
  // std::vector<double> aaa(all_nodes.size()*3);
  // CHKERR moab.tag_get_data(th,all_nodes,&aaa[0]);
  // CHKERR moab.tag_set_data(th_aaa,all_nodes,&aaa[0]);

  for (auto m : verts_map) {

    if (m.first < tol) {

      EntityHandle v = m.second.first;
      if (verticesOnTrimEdges.find(v) != verticesOnTrimEdges.end()) {
        continue;
      }

      VectorDouble3 ray_point(3);
      if (th) {
        CHKERR moab.tag_get_data(th, &v, 1, &ray_point[0]);
      } else {
        CHKERR moab.get_coords(&v, 1, &ray_point[0]);
      }

      Range adj_edges;
      CHKERR moab.get_adjacencies(&v, 1, 1, false, adj_edges);
      adj_edges = intersect(adj_edges, edges);
      Range w = intersect(adj_edges, tets_skin_edges);
      if (!w.empty()) {
        adj_edges.swap(w);
      }
      if (fixed_edges) {
        Range r = intersect(adj_edges, *fixed_edges);
        if (!r.empty()) {
          adj_edges.swap(r);
        }
      }
      if (adj_edges.empty()) {
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "Imposible case");
      }

      EntityHandle e = m.second.second;
      if (adj_edges.find(e) == adj_edges.end()) {
        continue;
      }

      bool corner_node = false;
      if (fixed_edges_verts_on_corners.find(v) !=
          fixed_edges_verts_on_corners.end()) {
        corner_node = true;
      }

      if (corner_node) {

        if (edgesToTrim.find(e) != edgesToTrim.end()) {
          auto &m = edgesToTrim.at(e);
          verticesOnTrimEdges[v] = m;
          verticesOnTrimEdges[v].dIst = 0;
          trimNewVertices.insert(v);
        }

      } else {

        VectorDouble3 new_pos(3);
        new_pos.clear();
        if (edgesToTrim.find(e) != edgesToTrim.end()) {
          auto &r = edgesToTrim.at(e);
          noalias(new_pos) = r.rayPoint + r.dIst * r.unitRayDir;
          VectorDouble3 unit_ray_dir = ray_point - new_pos;
          double dist = norm_2(unit_ray_dir);
          unit_ray_dir /= dist;

          auto get_quality_change = [this, &m_field, &moab, &new_pos, v,
                                     th](const Range &adj_tets) {
            double q0 = 1;
            CHKERR m_field.getInterface<Tools>()->minTetsQuality(adj_tets, q0);
            double q = 1;
            for (auto t : adj_tets) {
              int num_nodes;
              const EntityHandle *conn;
              CHKERR m_field.get_moab().get_connectivity(t, conn, num_nodes,
                                                         true);
              VectorDouble12 coords(12);
              if (th) {
                CHKERR moab.tag_get_data(th, conn, num_nodes, &coords[0]);
              } else {
                CHKERR moab.get_coords(conn, num_nodes, &coords[0]);
              }
              // cerr << coords << endl;
              for (int n = 0; n != 4; ++n) {
                auto n_coords = getVectorAdaptor(&coords[3 * n], 3);
                if (conn[n] == v) {
                  noalias(n_coords) = new_pos;
                } else {
                  auto m = verticesOnTrimEdges.find(conn[n]);
                  if (m != verticesOnTrimEdges.end()) {
                    auto r = m->second;
                    noalias(n_coords) = r.rayPoint + r.dIst * r.unitRayDir;
                  }
                }
              }
              q = std::min(q, Tools::volumeLengthQuality(&coords[0]));
            }
            return q / q0;
          };

          Range adj_tets;
          CHKERR moab.get_adjacencies(&v, 1, 3, false, adj_tets);
          adj_tets = intersect(adj_tets, cutNewVolumes);
          double q = get_quality_change(adj_tets);
          if (q > 0.75) {
            VectorDouble3 unit_ray_dir = new_pos - ray_point;
            double dist = norm_2(unit_ray_dir);
            unit_ray_dir /= dist;
            verticesOnTrimEdges[v].dIst = dist;
            verticesOnTrimEdges[v].lEngth = dist;
            verticesOnTrimEdges[v].unitRayDir = unit_ray_dir;
            verticesOnTrimEdges[v].rayPoint = ray_point;
            trimNewVertices.insert(v);
            // CHKERR moab.add_entities(meshset, &v, 1);
            // CHKERR moab.add_entities(meshset, &e, 1);
            // CHKERR moab.tag_set_data(th_aaa, &v, 1, &new_pos[0]);
          }
        }
      }
    }
  }

  // CHKERR moab.write_file("aaaa.vtk", "VTK", "", &meshset, 1);

  for (auto m : verticesOnTrimEdges) {
    EntityHandle v = m.first;
    Range adj_edges;
    CHKERR moab.get_adjacencies(&v, 1, 1, false, adj_edges);
    adj_edges = intersect(adj_edges, edges);
    for (auto e : adj_edges) {
      edgesToTrim.erase(e);
      trimEdges.erase(e);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::trimEdgesInTheMiddle(const BitRefLevel bit, Tag th,
                                               const double tol,
                                               const bool debug) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MeshRefinement *refiner;
  const RefEntity_multiIndex *ref_ents_ptr;
  MoFEMFunctionBegin;

  CHKERR m_field.getInterface(refiner);
  CHKERR m_field.get_ref_ents(&ref_ents_ptr);
  CHKERR refiner->add_verices_in_the_middel_of_edges(trimEdges, bit);
  CHKERR refiner->refine_TET(cutNewVolumes, bit, false, QUIET,
                             debug ? &trimEdges : NULL);

  trimNewVolumes.clear();
  CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      bit, BitRefLevel().set(), MBTET, trimNewVolumes);

  for (map<EntityHandle, TreeData>::iterator mit = edgesToTrim.begin();
       mit != edgesToTrim.end(); mit++) {
    auto vit = ref_ents_ptr->get<Composite_ParentEnt_And_EntType_mi_tag>().find(
        boost::make_tuple(MBVERTEX, mit->first));
    if (vit ==
        ref_ents_ptr->get<Composite_ParentEnt_And_EntType_mi_tag>().end()) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "No vertex on trim edges, that make no sense");
    }
    const boost::shared_ptr<RefEntity> &ref_ent = *vit;
    if ((ref_ent->getBitRefLevel() & bit).any()) {
      EntityHandle vert = ref_ent->getRefEnt();
      trimNewVertices.insert(vert);
      verticesOnTrimEdges[vert] = mit->second;
    }
  }

  // Get faces which are trimmed
  trimNewSurfaces.clear();
  CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      bit, BitRefLevel().set(), MBTRI, trimNewSurfaces);

  Range trim_new_surfaces_nodes;
  CHKERR moab.get_connectivity(trimNewSurfaces, trim_new_surfaces_nodes, true);
  trim_new_surfaces_nodes = subtract(trim_new_surfaces_nodes, trimNewVertices);
  trim_new_surfaces_nodes = subtract(trim_new_surfaces_nodes, cutNewVertices);
  Range faces_not_on_surface;
  CHKERR moab.get_adjacencies(trim_new_surfaces_nodes, 2, false,
                              faces_not_on_surface, moab::Interface::UNION);
  trimNewSurfaces = subtract(trimNewSurfaces, faces_not_on_surface);

  // Get surfaces which are not trimmed and add them to surface
  Range all_surfaces_on_bit_level;
  CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      bit, BitRefLevel().set(), MBTRI, all_surfaces_on_bit_level);
  all_surfaces_on_bit_level =
      intersect(all_surfaces_on_bit_level, cutNewSurfaces);
  trimNewSurfaces = unite(trimNewSurfaces, all_surfaces_on_bit_level);

  Range trim_surface_edges;
  CHKERR moab.get_adjacencies(trimNewSurfaces, 1, false, trim_surface_edges,
                              moab::Interface::UNION);

  // check of nodes are outside surface and if it are remove adjacent faces to
  // those nodes.
  Range check_verts;
  CHKERR moab.get_connectivity(trimNewSurfaces, check_verts, true);
  check_verts = subtract(check_verts, trimNewVertices);
  for (auto v : check_verts) {

    VectorDouble3 s(3);
    if (th) {
      CHKERR moab.tag_get_data(th, &v, 1, &s[0]);
    } else {
      CHKERR moab.get_coords(&v, 1, &s[0]);
    }

    VectorDouble3 p(3);
    EntityHandle facets_out;
    CHKERR treeSurfPtr->closest_to_location(&s[0], rootSetSurf, &p[0],
                                            facets_out);
    VectorDouble3 n(3);
    Util::normal(&moab, facets_out, n[0], n[1], n[2]);
    VectorDouble3 delta = s - p;
    VectorDouble3 normal = inner_prod(delta, n) * n;
    if (norm_2(delta - normal) > tol * aveLength) {
      Range adj;
      CHKERR moab.get_adjacencies(&v, 1, 2, false, adj);
      trimNewSurfaces = subtract(trimNewSurfaces, adj);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::getRayForEdge(const EntityHandle ent,
                                        VectorAdaptor &ray_point,
                                        VectorAdaptor &unit_ray_dir,
                                        double &ray_length) const {
  const CoreInterface &m_field = cOre;
  const moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;
  int num_nodes;
  const EntityHandle *conn;
  CHKERR moab.get_connectivity(ent, conn, num_nodes, true);
  double coords[6];
  CHKERR moab.get_coords(conn, num_nodes, coords);
  VectorAdaptor s0(3, ublas::shallow_array_adaptor<double>(3, &coords[0]));
  VectorAdaptor s1(3, ublas::shallow_array_adaptor<double>(3, &coords[3]));
  noalias(ray_point) = s0;
  noalias(unit_ray_dir) = s1 - s0;
  ray_length = norm_2(unit_ray_dir);
  unit_ray_dir /= ray_length;
  MoFEMFunctionReturn(0);
}

// int CutWithGA::segmentPlane(
//   VectorAdaptor s0,
//   VectorAdaptor s1,
//   VectorAdaptor x0,
//   VectorAdaptor n,
//   double &s
// ) const {
//   VectorDouble3 u = s1 - s0;
//   VectorDouble3 w = s0 - x0;
//   double nu = inner_prod(n,u);
//   double nw = -inner_prod(n,w);
//   const double tol = 1e-4;
//   if (fabs(nu) < tol) {           // segment is parallel to plane
//       if (nw == 0)                      // segment lies in plane
//           return 2;
//       else
//           return 0;                    // no intersection
//   }
//   // they are not parallel
//   // compute intersect param
//   s = nw / nu;
//   if (s < 0 || s > 1)
//       return 0;                        // no intersection
//   return 1;
// }

MoFEMErrorCode
CutWithGA::removePathologicalFrontTris(const BitRefLevel split_bit,
                                       Range &ents) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  PrismInterface *interface;
  MoFEMFunctionBegin;
  CHKERR m_field.getInterface(interface);
  // Remove tris on surface front
  {
    Range front_tris;
    EntityHandle meshset;
    CHKERR moab.create_meshset(MESHSET_SET, meshset);
    CHKERR moab.add_entities(meshset, ents);
    CHKERR interface->findIfTringleHasThreeNodesOnInternalSurfaceSkin(
        meshset, split_bit, true, front_tris);
    CHKERR moab.delete_entities(&meshset, 1);
    ents = subtract(ents, front_tris);
  }
  // Remove entities on skin
  Range tets;
  CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      split_bit, BitRefLevel().set(), MBTET, tets);
  // Remove entities on skin
  Skinner skin(&moab);
  Range tets_skin;
  rval = skin.find_skin(0, tets, false, tets_skin);
  ents = subtract(ents, tets_skin);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::splitSides(const BitRefLevel split_bit,
                                     const BitRefLevel bit, const Range &ents,
                                     Tag th) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  PrismInterface *interface;
  MoFEMFunctionBegin;
  CHKERR m_field.getInterface(interface);
  EntityHandle meshset_volume;
  CHKERR moab.create_meshset(MESHSET_SET, meshset_volume);
  CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      split_bit, BitRefLevel().set(), MBTET, meshset_volume);
  EntityHandle meshset_surface;
  CHKERR moab.create_meshset(MESHSET_SET, meshset_surface);
  CHKERR moab.add_entities(meshset_surface, ents);
  CHKERR interface->getSides(meshset_surface, split_bit, true);
  CHKERR interface->splitSides(meshset_volume, bit, meshset_surface, true,
                               true);
  CHKERR moab.delete_entities(&meshset_volume, 1);
  CHKERR moab.delete_entities(&meshset_surface, 1);
  if (th) {
    Range prisms;
    ierr = m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit, BitRefLevel().set(), MBPRISM, prisms);
    for (Range::iterator pit = prisms.begin(); pit != prisms.end(); pit++) {
      int num_nodes;
      const EntityHandle *conn;
      CHKERR moab.get_connectivity(*pit, conn, num_nodes, true);
      MatrixDouble data(3, 3);
      CHKERR moab.tag_get_data(th, conn, 3, &data(0, 0));
      // cerr << data << endl;
      CHKERR moab.tag_set_data(th, &conn[3], 3, &data(0, 0));
    }
  }
  MoFEMFunctionReturn(0);
}

struct LengthMapData {
  double lEngth;
  double qUality;
  EntityHandle eDge;
  bool skip;
  LengthMapData(const double l, double q, const EntityHandle e)
      : lEngth(l), qUality(q), eDge(e), skip(false) {}
};

typedef multi_index_container<
    LengthMapData,
    indexed_by<ordered_non_unique<
                   member<LengthMapData, double, &LengthMapData::lEngth>>,
               hashed_unique<
                   member<LengthMapData, EntityHandle, &LengthMapData::eDge>>>>
    LengthMapData_multi_index;

MoFEMErrorCode CutWithGA::mergeBadEdges(
    const int fraction_level, const Range &tets, const Range &surface,
    const Range &fixed_edges, const Range &corner_nodes, Range &edges_to_merge,
    Range &out_tets, Range &new_surf, Tag th, const bool update_meshsets,
    const BitRefLevel *bit_ptr, const bool debug) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;

  /**
   * \brief Merge nodes
   */
  struct MergeNodes {
    CoreInterface &mField;
    const bool onlyIfImproveQuality;
    const int lineSearch;
    Tag tH;
    bool updateMehsets;

    MergeNodes(CoreInterface &m_field, const bool only_if_improve_quality,
               const int line_search, Tag th, bool update_mehsets)
        : mField(m_field), onlyIfImproveQuality(only_if_improve_quality),
          lineSearch(line_search), tH(th), updateMehsets(update_mehsets) {
      mField.getInterface(nodeMergerPtr);
    }
    NodeMergerInterface *nodeMergerPtr;
    MoFEMErrorCode operator()(EntityHandle father, EntityHandle mother,
                              Range &proc_tets, Range &new_surf,
                              Range &edges_to_merge, Range &not_merged_edges,
                              bool add_child = true) const {
      moab::Interface &moab = mField.get_moab();
      MoFEMFunctionBegin;
      const EntityHandle conn[] = {father, mother};
      Range vert_tets;
      CHKERR moab.get_adjacencies(conn, 2, 3, false, vert_tets,
                                  moab::Interface::UNION);
      vert_tets = intersect(vert_tets, proc_tets);
      Range out_tets;
      CHKERR nodeMergerPtr->mergeNodes(father, mother, out_tets, &vert_tets,
                                       onlyIfImproveQuality, 0, lineSearch,
                                       tH); 
      out_tets.merge(subtract(proc_tets, vert_tets));
      proc_tets.swap(out_tets);

      if (add_child && nodeMergerPtr->getSucessMerge()) {

        NodeMergerInterface::ParentChildMap &parent_child_map =
            nodeMergerPtr->getParentChildMap();

        Range child_ents;
        NodeMergerInterface::ParentChildMap::iterator it;
        for (it = parent_child_map.begin(); it != parent_child_map.end();
             it++) {
          child_ents.insert(it->pArent);
        }

        Range new_surf_child_ents = intersect(new_surf, child_ents);
        new_surf = subtract(new_surf, new_surf_child_ents);
        Range child_surf_ents;
        CHKERR updateRangeByChilds(parent_child_map, new_surf_child_ents,
                                   child_surf_ents);
        new_surf.merge(child_surf_ents);

        Range edges_to_merge_child_ents = intersect(edges_to_merge, child_ents);
        edges_to_merge = subtract(edges_to_merge, edges_to_merge_child_ents);
        Range merged_child_edge_ents;
        CHKERR updateRangeByChilds(parent_child_map, edges_to_merge_child_ents,
                                   merged_child_edge_ents);

        Range not_merged_edges_child_ents =
            intersect(not_merged_edges, child_ents);
        not_merged_edges =
            subtract(not_merged_edges, not_merged_edges_child_ents);
        Range not_merged_child_edge_ents;
        CHKERR updateRangeByChilds(parent_child_map,
                                   not_merged_edges_child_ents,
                                   not_merged_child_edge_ents);

        merged_child_edge_ents =
            subtract(merged_child_edge_ents, not_merged_child_edge_ents);
        edges_to_merge.merge(merged_child_edge_ents);
        not_merged_edges.merge(not_merged_child_edge_ents);

        if (updateMehsets) {
          for (_IT_CUBITMESHSETS_FOR_LOOP_(
                   (*mField.getInterface<MeshsetsManager>()), cubit_it)) {
            EntityHandle cubit_meshset = cubit_it->meshset;
            Range parent_ents;
            CHKERR moab.get_entities_by_handle(cubit_meshset, parent_ents,
                                               true);
            Range child_ents;
            CHKERR updateRangeByChilds(parent_child_map, parent_ents,
                                       child_ents);
            CHKERR moab.add_entities(cubit_meshset, child_ents);
          }
        }
      }
      MoFEMFunctionReturn(0);
    }

  private:
    MoFEMErrorCode updateRangeByChilds(
        const NodeMergerInterface::ParentChildMap &parent_child_map,
        const Range &parents, Range &childs) const {
      MoFEMFunctionBeginHot;
      NodeMergerInterface::ParentChildMap::nth_index<0>::type::iterator it;
      for (Range::const_iterator eit = parents.begin(); eit != parents.end();
           eit++) {
        it = parent_child_map.get<0>().find(*eit);
        if (it == parent_child_map.get<0>().end())
          continue;
        childs.insert(it->cHild);
      }
      MoFEMFunctionReturnHot(0);
    }
  };

  /**
   * \brief Calculate edge length
   */
  struct LengthMap {
    Tag tH;
    CoreInterface &mField;
    moab::Interface &moab;
    const double maxLength;
    LengthMap(CoreInterface &m_field, Tag th, double max_length)
        : tH(th), mField(m_field), moab(m_field.get_moab()),
          maxLength(max_length) {
      gammaL = 1.;
      gammaQ = 3.;
      acceptedThrasholdMergeQuality = 0.5;
    }

    double
        gammaL; ///< Controls importance of length when ranking edges for merge
    double
        gammaQ; ///< Controls importance of quality when ranking edges for merge
    double acceptedThrasholdMergeQuality; ///< Do not merge quality if quality
                                          ///< above accepted thrashold

    MoFEMErrorCode operator()(const Range &tets, const Range &edges,
                              LengthMapData_multi_index &length_map,
                              double &ave) const {
      int num_nodes;
      const EntityHandle *conn;
      double coords[6];
      MoFEMFunctionBegin;
      VectorAdaptor s0(3, ublas::shallow_array_adaptor<double>(3, &coords[0]));
      VectorAdaptor s1(3, ublas::shallow_array_adaptor<double>(3, &coords[3]));
      VectorDouble3 delta(3);
      for (auto edge : edges) {
        CHKERR moab.get_connectivity(edge, conn, num_nodes, true);
        Range adj_tet;
        CHKERR moab.get_adjacencies(conn, num_nodes, 3, false, adj_tet);
        adj_tet = intersect(adj_tet, tets);
        if (tH) {
          CHKERR moab.tag_get_data(tH, conn, num_nodes, coords);
        } else {
          CHKERR moab.get_coords(conn, num_nodes, coords);
        }
        double q = 1;
        auto abs_min = [](double a, double b) {
          return std::min(fabs(a), fabs(b));
        };
        CHKERR mField.getInterface<Tools>()->minTetsQuality(adj_tet, q, tH,
                                                            abs_min);
        // if (q != q)
        //   SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
        //           "Quality not a number");
        if (fabs(q) > acceptedThrasholdMergeQuality)
          continue;
        noalias(delta) = (s0 - s1) / maxLength;
        double dot = inner_prod(delta, delta);
        double val = pow(q, gammaQ) * pow(dot, 0.5 * gammaL);
        length_map.insert(LengthMapData(val, q, edge));
      }
      ave = 0;
      for (LengthMapData_multi_index::nth_index<0>::type::iterator mit =
               length_map.get<0>().begin();
           mit != length_map.get<0>().end(); mit++) {
        ave += mit->qUality;
      }
      ave /= length_map.size();
      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief Topological relations
   */
  struct Toplogy {

    CoreInterface &mField;
    Tag tH;
    const double tOL;
    Toplogy(CoreInterface &m_field, Tag th, const double tol)
        : mField(m_field), tH(th), tOL(tol) {}

    enum TYPE {
      FREE = 0,
      SKIN = 1 << 0,
      SURFACE = 1 << 1,
      SURFACE_SKIN = 1 << 2,
      FRONT_ENDS = 1 << 3,
      FIX_EDGES = 1 << 4,
      FIX_CORNERS = 1 << 5
    };

    typedef map<int, Range> SetsMap;

    MoFEMErrorCode classifyVerts(const Range &surface, const Range &tets,
                                 const Range &fixed_edges,
                                 const Range &corner_nodes,
                                 SetsMap &sets_map) const {
      moab::Interface &moab(mField.get_moab());
      Skinner skin(&moab);
      MoFEMFunctionBegin;

      sets_map[FIX_CORNERS].merge(corner_nodes);
      Range fixed_verts;
      CHKERR moab.get_connectivity(fixed_edges, fixed_verts, true);
      sets_map[FIX_EDGES].swap(fixed_verts);

      Range tets_skin;
      CHKERR skin.find_skin(0, tets, false, tets_skin);
      Range tets_skin_edges;
      CHKERR moab.get_adjacencies(tets_skin, 1, false, tets_skin_edges,
                                  moab::Interface::UNION);

      // surface skin
      Range surface_skin;
      CHKERR skin.find_skin(0, surface, false, surface_skin);
      Range front_in_the_body;
      front_in_the_body = subtract(surface_skin, tets_skin_edges);
      Range front_ends;
      CHKERR skin.find_skin(0, front_in_the_body, false, front_ends);
      sets_map[FRONT_ENDS].swap(front_ends);

      Range surface_skin_verts;
      CHKERR moab.get_connectivity(surface_skin, surface_skin_verts, true);
      sets_map[SURFACE_SKIN].swap(surface_skin_verts);

      // surface
      Range surface_verts;
      CHKERR moab.get_connectivity(surface, surface_verts, true);
      sets_map[SURFACE].swap(surface_verts);

      // skin
      Range tets_skin_verts;
      CHKERR moab.get_connectivity(tets_skin, tets_skin_verts, true);
      sets_map[SKIN].swap(tets_skin_verts);

      Range tets_verts;
      CHKERR moab.get_connectivity(tets, tets_verts, true);
      sets_map[FREE].swap(tets_verts);

      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode getProcTets(const Range &tets, const Range &edges_to_merge,
                               Range &proc_tets) const {
      moab::Interface &moab(mField.get_moab());
      MoFEMFunctionBegin;
      Range edges_to_merge_verts;
      CHKERR moab.get_connectivity(edges_to_merge, edges_to_merge_verts, true);
      Range edges_to_merge_verts_tets;
      CHKERR moab.get_adjacencies(edges_to_merge_verts, 3, false,
                                  edges_to_merge_verts_tets,
                                  moab::Interface::UNION);
      edges_to_merge_verts_tets = intersect(edges_to_merge_verts_tets, tets);
      proc_tets.swap(edges_to_merge_verts_tets);
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode edgesToMerge(const Range &surface, const Range &tets,
                                Range &edges_to_merge) const {
      moab::Interface &moab(mField.get_moab());
      MoFEMFunctionBegin;

      Range surface_verts;
      CHKERR moab.get_connectivity(surface, surface_verts, true);
      Range surface_verts_edges;
      CHKERR moab.get_adjacencies(surface_verts, 1, false, surface_verts_edges,
                                  moab::Interface::UNION);
      edges_to_merge.merge(surface_verts_edges);
      Range tets_edges;
      CHKERR moab.get_adjacencies(tets, 1, false, tets_edges,
                                  moab::Interface::UNION);
      edges_to_merge = intersect(edges_to_merge, tets_edges);
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode removeBadEdges(const Range &surface, const Range &tets,
                                  const Range &fixed_edges,
                                  const Range &corner_nodes,
                                  Range &edges_to_merge,
                                  Range &not_merged_edges) {
      moab::Interface &moab(mField.get_moab());
      MoFEMFunctionBegin;

      // find skin
      Skinner skin(&moab);
      Range tets_skin;
      CHKERR skin.find_skin(0, tets, false, tets_skin);
      Range surface_skin;
      CHKERR skin.find_skin(0, surface, false, surface_skin);

      // end nodes
      Range tets_skin_edges;
      CHKERR moab.get_adjacencies(tets_skin, 1, false, tets_skin_edges,
                                  moab::Interface::UNION);
      Range surface_front;
      surface_front = subtract(surface_skin, tets_skin_edges);
      Range surface_front_nodes;
      CHKERR moab.get_connectivity(surface_front, surface_front_nodes, true);
      Range ends_nodes;
      CHKERR skin.find_skin(0, surface_front, false, ends_nodes);

      // remove bad merges

      // get surface and body skin verts
      Range surface_edges;
      CHKERR moab.get_adjacencies(surface, 1, false, surface_edges,
                                  moab::Interface::UNION);
      // get nodes on the surface
      Range surface_edges_verts;
      CHKERR moab.get_connectivity(surface_edges, surface_edges_verts, true);
      // get vertices on the body skin
      Range tets_skin_edges_verts;
      CHKERR moab.get_connectivity(tets_skin_edges, tets_skin_edges_verts,
                                   true);

      Range edges_to_remove;

      // remove edges self connected to body skin
      {
        Range ents_nodes_and_edges;
        ents_nodes_and_edges.merge(tets_skin_edges_verts);
        ents_nodes_and_edges.merge(tets_skin_edges);
        CHKERR removeSelfConectingEdges(ents_nodes_and_edges, edges_to_remove,
                                        0, false);
      }
      edges_to_merge = subtract(edges_to_merge, edges_to_remove);
      not_merged_edges.merge(edges_to_remove);

      // remove edges self connected to surface
      {
        Range ents_nodes_and_edges;
        ents_nodes_and_edges.merge(surface_edges_verts);
        ents_nodes_and_edges.merge(surface_edges);
        ents_nodes_and_edges.merge(tets_skin_edges_verts);
        ents_nodes_and_edges.merge(tets_skin_edges);
        CHKERR removeSelfConectingEdges(ents_nodes_and_edges, edges_to_remove,
                                        0, false);
      }
      edges_to_merge = subtract(edges_to_merge, edges_to_remove);
      not_merged_edges.merge(edges_to_remove);

      // remove edges adjacent corner_nodes execpt those on fixed edges
      Range fixed_edges_nodes;
      CHKERR moab.get_connectivity(fixed_edges, fixed_edges_nodes, true);
      {
        Range ents_nodes_and_edges;
        ents_nodes_and_edges.merge(fixed_edges_nodes);
        ents_nodes_and_edges.merge(ends_nodes);
        ents_nodes_and_edges.merge(corner_nodes);
        ents_nodes_and_edges.merge(fixed_edges);
        CHKERR removeSelfConectingEdges(ents_nodes_and_edges, edges_to_remove,
                                        0, false);
      }
      edges_to_merge = subtract(edges_to_merge, edges_to_remove);
      not_merged_edges.merge(edges_to_remove);

      // remove edges self connected to surface
      CHKERR removeSelfConectingEdges(surface_edges, edges_to_remove, 0, false);
      edges_to_merge = subtract(edges_to_merge, edges_to_remove);
      not_merged_edges.merge(edges_to_remove);

      // remove edges self contented on surface skin
      {
        Range ents_nodes_and_edges;
        ents_nodes_and_edges.merge(surface_skin);
        ents_nodes_and_edges.merge(fixed_edges_nodes);
        CHKERR removeSelfConectingEdges(ents_nodes_and_edges, edges_to_remove,
                                        0, false);
      }
      edges_to_merge = subtract(edges_to_merge, edges_to_remove);
      not_merged_edges.merge(edges_to_remove);

      // remove crack front nodes connected to the surface
      {
        Range ents_nodes_and_edges;
        ents_nodes_and_edges.merge(surface_front_nodes);
        ents_nodes_and_edges.merge(surface_front);
        ents_nodes_and_edges.merge(tets_skin_edges_verts);
        ents_nodes_and_edges.merge(tets_skin_edges);
        CHKERR removeSelfConectingEdges(ents_nodes_and_edges, edges_to_remove,
                                        0, false);
      }
      edges_to_merge = subtract(edges_to_merge, edges_to_remove);
      not_merged_edges.merge(edges_to_remove);

      // remove edges connecting crack front and fixed edges, except those
      {
        Range ents_nodes_and_edges;
        ents_nodes_and_edges.merge(surface_skin.subset_by_type(MBEDGE));
        ents_nodes_and_edges.merge(fixed_edges.subset_by_type(MBEDGE));
        CHKERR removeSelfConectingEdges(ents_nodes_and_edges, edges_to_remove,
                                        tOL, false);
      }
      edges_to_merge = subtract(edges_to_merge, edges_to_remove);
      not_merged_edges.merge(edges_to_remove);

      MoFEMFunctionReturn(0);
    }

  private:
    MoFEMErrorCode removeSelfConectingEdges(const Range &ents,
                                            Range &edges_to_remove,
                                            const bool length,
                                            bool debug) const {
      moab::Interface &moab(mField.get_moab());
      MoFEMFunctionBegin;
      // get nodes
      Range ents_nodes = ents.subset_by_type(MBVERTEX);
      if (ents_nodes.empty()) {
        CHKERR moab.get_connectivity(ents, ents_nodes, true);
      }
      // edges adj. to nodes
      Range ents_nodes_edges;
      CHKERR moab.get_adjacencies(ents_nodes, 1, false, ents_nodes_edges,
                                  moab::Interface::UNION);
      // nodes of adj. edges
      Range ents_nodes_edges_nodes;
      CHKERR moab.get_connectivity(ents_nodes_edges, ents_nodes_edges_nodes,
                                   true);
      // hanging nodes
      ents_nodes_edges_nodes = subtract(ents_nodes_edges_nodes, ents_nodes);
      Range ents_nodes_edges_nodes_edges;
      CHKERR moab.get_adjacencies(ents_nodes_edges_nodes, 1, false,
                                  ents_nodes_edges_nodes_edges,
                                  moab::Interface::UNION);
      // remove edges adj. to hanging edges
      ents_nodes_edges =
          subtract(ents_nodes_edges, ents_nodes_edges_nodes_edges);
      ents_nodes_edges =
          subtract(ents_nodes_edges, ents.subset_by_type(MBEDGE));
      if (length > 0) {
        Range::iterator eit = ents_nodes_edges.begin();
        for (; eit != ents_nodes_edges.end();) {

          int num_nodes;
          const EntityHandle *conn;
          rval = moab.get_connectivity(*eit, conn, num_nodes, true);
          double coords[6];
          if (tH) {
            CHKERR moab.tag_get_data(tH, conn, num_nodes, coords);
          } else {
            CHKERR moab.get_coords(conn, num_nodes, coords);
          }

          auto get_edge_length = [coords]() {
            FTensor::Tensor1<FTensor::PackPtr<const double *, 3>, 3> t_coords(
                &coords[0], &coords[1], &coords[2]);
            FTensor::Tensor1<double, 3> t_delta;
            FTensor::Index<'i', 3> i;
            t_delta(i) = t_coords(i);
            ++t_coords;
            t_delta(i) -= t_coords(i);
            return sqrt(t_delta(i) * t_delta(i));
          };

          if (get_edge_length() < tOL) {
            eit = ents_nodes_edges.erase(eit);
          } else {
            eit++;
          }
        }
      }
      edges_to_remove.swap(ents_nodes_edges);

      MoFEMFunctionReturn(0);
    }
  };

  Range not_merged_edges;
  const double tol = 0.05;
  CHKERR Toplogy(m_field, th, tol * aveLength)
      .edgesToMerge(surface, tets, edges_to_merge);
  CHKERR Toplogy(m_field, th, tol * aveLength)
      .removeBadEdges(surface, tets, fixed_edges, corner_nodes, edges_to_merge,
                      not_merged_edges);
  Toplogy::SetsMap sets_map;
  CHKERR Toplogy(m_field, th, tol * aveLength)
      .classifyVerts(surface, tets, fixed_edges, corner_nodes, sets_map);
  Range proc_tets;
  CHKERR Toplogy(m_field, th, tol * aveLength)
      .getProcTets(tets, edges_to_merge, proc_tets);
  out_tets = subtract(tets, proc_tets);
  if (bit_ptr) {
    for (int dd = 2; dd >= 0; dd--) {
      CHKERR moab.get_adjacencies(out_tets.subset_by_dimension(3), dd, false,
                                  out_tets, moab::Interface::UNION);
    }
    CHKERR m_field.getInterface<BitRefManager>()->addBitRefLevel(out_tets,
                                                                 *bit_ptr);
  }

  int nb_nodes_merged = 0;
  LengthMapData_multi_index length_map;
  new_surf = surface;

  double ave0 = 0, ave = 0, min = 0, min_p = 0, min_pp;
  for (int pp = 0; pp != nbMaxMergingCycles; ++pp) {

    int nb_nodes_merged_p = nb_nodes_merged;
    length_map.clear();
    min_pp = min_p;
    min_p = min;
    CHKERR LengthMap(m_field, th, aveLength)(proc_tets, edges_to_merge,
                                             length_map, ave);
    min = length_map.get<0>().begin()->qUality;
    if (pp == 0) {
      ave0 = ave;
    }

    int nn = 0;
    Range collapsed_edges;
    for (LengthMapData_multi_index::nth_index<0>::type::iterator mit =
             length_map.get<0>().begin();
         mit != length_map.get<0>().end(); mit++, nn++) {
      // cerr << mit->lEngth << endl; //" " << mit->eDge << endl;
      if (mit->skip)
        continue;
      int num_nodes;
      const EntityHandle *conn;
      CHKERR moab.get_connectivity(mit->eDge, conn, num_nodes, true);
      int conn_type[2] = {0, 0};
      for (int nn = 0; nn != 2; nn++) {
        conn_type[nn] = 0;
        for (Toplogy::SetsMap::reverse_iterator sit = sets_map.rbegin();
             sit != sets_map.rend(); sit++) {
          if (sit->second.find(conn[nn]) != sit->second.end()) {
            conn_type[nn] |= sit->first;
          }
        }
      }
      int type_father, type_mother;
      EntityHandle father, mother;
      if (conn_type[0] > conn_type[1]) {
        father = conn[0];
        mother = conn[1];
        type_father = conn_type[0];
        type_mother = conn_type[1];
      } else {
        father = conn[1];
        mother = conn[0];
        type_father = conn_type[1];
        type_mother = conn_type[0];
      }
      int line_search = 0;
      if (type_father == type_mother) {
        line_search = lineSearchSteps;
      }

      CHKERR MergeNodes(m_field, true, line_search, th,
                        update_meshsets)(father, mother, proc_tets, new_surf,
                                         edges_to_merge, not_merged_edges);
      if (m_field.getInterface<NodeMergerInterface>()->getSucessMerge()) {
        Range adj_mother_tets;
        CHKERR moab.get_adjacencies(&mother, 1, 3, false, adj_mother_tets);
        Range adj_mother_tets_nodes;
        CHKERR moab.get_connectivity(adj_mother_tets, adj_mother_tets_nodes,
                                     true);
        Range adj_edges;
        CHKERR moab.get_adjacencies(adj_mother_tets_nodes, 1, false, adj_edges,
                                    moab::Interface::UNION);
        CHKERR moab.get_adjacencies(&father, 1, 1, false, adj_edges,
                                    moab::Interface::UNION);
        for (Range::iterator ait = adj_edges.begin(); ait != adj_edges.end();
             ait++) {
          LengthMapData_multi_index::nth_index<1>::type::iterator miit =
              length_map.get<1>().find(*ait);
          if (miit != length_map.get<1>().end()) {
            (const_cast<LengthMapData &>(*miit)).skip = true;
          }
        }
        nb_nodes_merged++;
        collapsed_edges.insert(mit->eDge);
      }

      if (nn > static_cast<int>(length_map.size() / fraction_level))
        break;
      if (mit->qUality > ave)
        break;
    }

    Range adj_faces, adj_edges;
    CHKERR moab.get_adjacencies(proc_tets, 2, false, adj_faces,
                                moab::Interface::UNION);
    new_surf = intersect(new_surf, adj_faces);

    // PetscPrintf(m_field.get_comm(),
    //             "(%d) Number of nodes merged %d ave q %3.4e min q %3.4e\n",
    //             pp, nb_nodes_merged, ave, min);

    if (nb_nodes_merged == nb_nodes_merged_p)
      break;
    if (min > 1e-2 && min == min_pp)
      break;
    if (min > ave0)
      break;

    CHKERR moab.get_adjacencies(proc_tets, 1, false, adj_edges,
                                moab::Interface::UNION);
    edges_to_merge = intersect(edges_to_merge, adj_edges);
    CHKERR Toplogy(m_field, th, tol * aveLength)
        .removeBadEdges(new_surf, proc_tets, fixed_edges, corner_nodes,
                        edges_to_merge, not_merged_edges);
  }

  if (bit_ptr) {
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevel(proc_tets,
                                                                 *bit_ptr);
  }
  out_tets.merge(proc_tets);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
CutWithGA::mergeBadEdges(const int fraction_level, const BitRefLevel trim_bit,
                         const BitRefLevel cut_bit, const BitRefLevel bit,
                         const Range &surface, const Range &fixed_edges,
                         const Range &corner_nodes, Tag th,
                         const bool update_meshsets, const bool debug) {
  CoreInterface &m_field = cOre;
  MoFEMFunctionBeginHot;
  Range tets_level;
  ierr = m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      trim_bit, BitRefLevel().set(), MBTET, tets_level);
  CHKERRG(ierr);

  Range edges_to_merge;
  ierr = m_field.getInterface<BitRefManager>()->getEntitiesByParentType(
      trim_bit, trim_bit | cut_bit, MBEDGE, edges_to_merge);
  CHKERRG(ierr);
  edges_to_merge = edges_to_merge.subset_by_type(MBEDGE);

  // get all entities not in database
  Range all_ents_not_in_database_before;
  ierr = cOre.getInterface<BitRefManager>()->getAllEntitiesNotInDatabase(
      all_ents_not_in_database_before);
  CHKERRG(ierr);

  Range out_new_tets, out_new_surf;
  CHKERR mergeBadEdges(fraction_level, tets_level, surface, fixed_edges,
                       corner_nodes, edges_to_merge, out_new_tets, out_new_surf,
                       th, update_meshsets, &bit, debug);

  // get all entities not in database after merge
  Range all_ents_not_in_database_after;
  ierr = cOre.getInterface<BitRefManager>()->getAllEntitiesNotInDatabase(
      all_ents_not_in_database_after);
  CHKERRG(ierr);
  // delete hanging entities
  all_ents_not_in_database_after =
      subtract(all_ents_not_in_database_after, all_ents_not_in_database_before);
  Range meshsets;
  CHKERR m_field.get_moab().get_entities_by_type(0, MBENTITYSET, meshsets,
                                                 false);
  for (Range::iterator mit = meshsets.begin(); mit != meshsets.end(); mit++) {
    CHKERR m_field.get_moab().remove_entities(*mit,
                                              all_ents_not_in_database_after);
  }
  m_field.get_moab().delete_entities(all_ents_not_in_database_after);

  mergedVolumes.swap(out_new_tets);
  mergedSurfaces.swap(out_new_surf);
  MoFEMFunctionReturnHot(0);
}

#ifdef WITH_TETGEN

MoFEMErrorCode CutWithGA::rebuildMeshWithTetGen(
    vector<string> &switches, const BitRefLevel &mesh_bit,
    const BitRefLevel &bit, const Range &surface, const Range &fixed_edges,
    const Range &corner_nodes, Tag th, const bool debug) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  TetGenInterface *tetgen_iface;
  MoFEMFunctionBegin;
  CHKERR m_field.getInterface(tetgen_iface);

  tetGenData.clear();
  moabTetGenMap.clear();
  tetGenMoabMap.clear();

  if (tetGenData.size() < 1) {
    tetGenData.push_back(new tetgenio);
  }
  tetgenio &in = tetGenData.back();

  struct BitEnts {

    CoreInterface &mField;
    const BitRefLevel &bIt;
    BitEnts(CoreInterface &m_field, const BitRefLevel &bit)
        : mField(m_field), bIt(bit) {}

    Range mTets;
    Range mTris;
    Range mEdges;
    Range mNodes;

    MoFEMErrorCode getLevelEnts() {
      MoFEMFunctionBeginHot;
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
          bIt, BitRefLevel().set(), MBTET, mTets);
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
          bIt, BitRefLevel().set(), MBTRI, mTris);
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
          bIt, BitRefLevel().set(), MBEDGE, mEdges);
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
          bIt, BitRefLevel().set(), MBVERTEX, mNodes);
      MoFEMFunctionReturnHot(0);
    }

    Range mSkin;
    Range mSkinNodes;
    Range mSkinEdges;

    MoFEMErrorCode getSkin() {
      moab::Interface &moab = mField.get_moab();
      MoFEMFunctionBeginHot;
      Skinner skin(&moab);
      CHKERR skin.find_skin(0, mTets, false, mSkin);
      CHKERR mField.get_moab().get_connectivity(mSkin, mSkinNodes, true);
      CHKERR mField.get_moab().get_adjacencies(mSkin, 1, false, mSkinEdges,
                                               moab::Interface::UNION);
      MoFEMFunctionReturnHot(0);
    }
  };

  struct SurfaceEnts {

    CoreInterface &mField;
    SurfaceEnts(CoreInterface &m_field) : mField(m_field) {}

    Range sNodes;
    Range sEdges;
    Range sVols;
    Range vNodes;

    MoFEMErrorCode getVolume(const BitEnts &bit_ents, const Range &tris) {
      moab::Interface &moab = mField.get_moab();
      MoFEMFunctionBeginHot;
      CHKERR moab.get_connectivity(tris, sNodes, true);
      CHKERR moab.get_adjacencies(tris, 1, false, sEdges,
                                  moab::Interface::UNION);
      CHKERR moab.get_adjacencies(sNodes, 3, false, sVols,
                                  moab::Interface::UNION);
      sVols = intersect(sVols, bit_ents.mTets);
      CHKERR moab.get_connectivity(sVols, vNodes, true);
      MoFEMFunctionReturnHot(0);
    }

    Range sSkin;
    Range sSkinNodes;
    Range vSkin;
    Range vSkinNodes;
    Range vSkinWithoutBodySkin;
    Range vSkinNodesWithoutBodySkin;
    Range vSkinOnBodySkin;
    Range vSkinOnBodySkinNodes;

    MoFEMErrorCode getSkin(const BitEnts &bit_ents, const Range &tris,
                           const int levels = 3) {
      moab::Interface &moab = mField.get_moab();
      MoFEMFunctionBeginHot;
      Skinner skin(&moab);
      rval = skin.find_skin(0, sVols, false, vSkin);
      for (int ll = 0; ll != levels; ll++) {
        CHKERR moab.get_adjacencies(vSkin, 3, false, sVols,
                                    moab::Interface::UNION);
        sVols = intersect(sVols, bit_ents.mTets);
        vSkin.clear();
        CHKERR skin.find_skin(0, sVols, false, vSkin);
      }
      vSkinWithoutBodySkin = subtract(vSkin, bit_ents.mSkin);
      vSkinOnBodySkin = intersect(vSkin, bit_ents.mSkin);
      CHKERR moab.get_connectivity(vSkinOnBodySkin, vSkinOnBodySkinNodes, true);
      CHKERR moab.get_connectivity(sVols, vNodes, true);
      CHKERR moab.get_connectivity(vSkin, vSkinNodes, true);
      vSkinNodesWithoutBodySkin = subtract(vSkinNodes, bit_ents.mSkinNodes);
      CHKERR skin.find_skin(0, tris, false, sSkin);
      CHKERR moab.get_connectivity(sSkin, sSkinNodes, true);
      tVols = sVols;
      MoFEMFunctionReturnHot(0);
    }

    Range tVols;

    MoFEMErrorCode getTetsForRemesh(const BitEnts &bit_ents, Tag th = NULL) {
      moab::Interface &moab = mField.get_moab();
      MoFEMFunctionBeginHot;

      Range tets_with_four_nodes_on_skin;
      rval = moab.get_adjacencies(vSkinOnBodySkinNodes, 3, false,
                                  tets_with_four_nodes_on_skin,
                                  moab::Interface::UNION);
      Range tets_nodes;
      CHKERR moab.get_connectivity(tets_with_four_nodes_on_skin, tets_nodes,
                                   true);
      tets_nodes = subtract(tets_nodes, vSkinOnBodySkinNodes);
      Range other_tets;
      CHKERR moab.get_adjacencies(tets_nodes, 3, false, other_tets,
                                  moab::Interface::UNION);
      tets_with_four_nodes_on_skin =
          subtract(tets_with_four_nodes_on_skin, other_tets);
      Range to_remove;
      for (Range::iterator tit = tets_with_four_nodes_on_skin.begin();
           tit != tets_with_four_nodes_on_skin.end(); tit++) {
        int num_nodes;
        const EntityHandle *conn;
        CHKERR moab.get_connectivity(*tit, conn, num_nodes, true);
        double coords[12];
        if (th) {
          CHKERR moab.tag_get_data(th, conn, num_nodes, coords);
        } else {
          CHKERR moab.get_coords(conn, num_nodes, coords);
        }
        double quality = Tools::volumeLengthQuality(coords);
        if (quality < 1e-2) {
          to_remove.insert(*tit);
        }
      }

      sVols = subtract(sVols, to_remove);

      Skinner skin(&moab);
      vSkin.clear();
      CHKERR skin.find_skin(0, sVols, false, vSkin);
      Range m_skin;
      CHKERR
      skin.find_skin(0, subtract(bit_ents.mSkin, to_remove), false, m_skin);
      vSkinWithoutBodySkin = subtract(vSkin, m_skin);
      vSkinOnBodySkin = intersect(vSkin, m_skin);
      vNodes.clear();
      vSkinNodes.clear();
      vSkinOnBodySkinNodes.clear();
      CHKERR moab.get_connectivity(sVols, vNodes, true);
      CHKERR moab.get_connectivity(vSkinOnBodySkin, vSkinOnBodySkinNodes, true);
      CHKERR moab.get_connectivity(vSkin, vSkinNodes, true);
      MoFEMFunctionReturnHot(0);
    }
  };

  BitEnts bit_ents(m_field, mesh_bit);
  CHKERR bit_ents.getLevelEnts();
  CHKERR bit_ents.getSkin();
  SurfaceEnts surf_ents(m_field);
  CHKERR surf_ents.getVolume(bit_ents, surface);
  CHKERR surf_ents.getSkin(bit_ents, surface);
  CHKERR surf_ents.getTetsForRemesh(bit_ents);

  map<int, Range> types_ents;
  types_ents[TetGenInterface::RIDGEVERTEX].merge(
      surf_ents.vSkinNodesWithoutBodySkin);
  // FREESEGVERTEX
  types_ents[TetGenInterface::FREESEGVERTEX].merge(surf_ents.sSkinNodes);
  types_ents[TetGenInterface::FREESEGVERTEX] =
      subtract(types_ents[TetGenInterface::FREESEGVERTEX],
               types_ents[TetGenInterface::RIDGEVERTEX]);
  // FREEFACETVERTEX
  types_ents[TetGenInterface::FREEFACETVERTEX].merge(surf_ents.sNodes);
  types_ents[TetGenInterface::FREEFACETVERTEX] =
      subtract(types_ents[TetGenInterface::FREEFACETVERTEX],
               types_ents[TetGenInterface::RIDGEVERTEX]);
  types_ents[TetGenInterface::FREEFACETVERTEX] =
      subtract(types_ents[TetGenInterface::FREEFACETVERTEX],
               types_ents[TetGenInterface::FREESEGVERTEX]);
  // FREEVOLVERTEX
  types_ents[TetGenInterface::FREEVOLVERTEX].merge(surf_ents.vNodes);
  types_ents[TetGenInterface::FREEVOLVERTEX] =
      subtract(types_ents[TetGenInterface::FREEVOLVERTEX],
               types_ents[TetGenInterface::RIDGEVERTEX]);
  types_ents[TetGenInterface::FREEVOLVERTEX] =
      subtract(types_ents[TetGenInterface::FREEVOLVERTEX],
               types_ents[TetGenInterface::FREESEGVERTEX]);
  types_ents[TetGenInterface::FREEVOLVERTEX] =
      subtract(types_ents[TetGenInterface::FREEVOLVERTEX],
               types_ents[TetGenInterface::FREEFACETVERTEX]);

  Tag th_marker;
  // Clean markers
  rval = m_field.get_moab().tag_get_handle("TETGEN_MARKER", th_marker);
  if (rval == MB_SUCCESS) {
    CHKERR m_field.get_moab().tag_delete(th_marker);
    rval = MB_SUCCESS;
  }

  int def_marker = 0;
  CHKERR m_field.get_moab().tag_get_handle(
      "TETGEN_MARKER", 1, MB_TYPE_INTEGER, th_marker,
      MB_TAG_CREAT | MB_TAG_SPARSE, &def_marker);

  // Mark surface with id = 1
  vector<int> markers(surface.size(), 1);
  CHKERR m_field.get_moab().tag_set_data(th_marker, surface, &*markers.begin());
  // Mark all side sets
  int shift = 1;
  map<int, int> id_shift_map; // each meshset has set unique bit
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(
           (*cOre.getInterface<MeshsetsManager>()), SIDESET, it)) {
    int ms_id = it->getMeshsetId();
    id_shift_map[ms_id] = 1 << shift; // shift bit
    ++shift;
    Range sideset_faces;
    CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
        ms_id, SIDESET, 2, sideset_faces, true);
    markers.resize(sideset_faces.size());
    CHKERR m_field.get_moab().tag_get_data(th_marker, sideset_faces,
                                           &*markers.begin());
    for (unsigned int ii = 0; ii < markers.size(); ii++) {
      markers[ii] |= id_shift_map[ms_id]; // add bit to marker
    }
    CHKERR m_field.get_moab().tag_set_data(th_marker, sideset_faces,
                                           &*markers.begin());
  }
  Range nodes_to_remove; // none
  markers.resize(nodes_to_remove.size());
  fill(markers.begin(), markers.end(), -1);
  CHKERR m_field.get_moab().tag_set_data(th_marker, nodes_to_remove,
                                         &*markers.begin());

  // nodes
  if (tetGenData.size() == 1) {

    Range ents_to_tetgen = surf_ents.sVols;
    CHKERR m_field.get_moab().get_connectivity(surf_ents.sVols, ents_to_tetgen,
                                               true);
    for (int dd = 2; dd >= 1; dd--) {
      CHKERR m_field.get_moab().get_adjacencies(
          surf_ents.sVols, dd, false, ents_to_tetgen, moab::Interface::UNION);
    }

    // Load mesh to TetGen data structures
    CHKERR tetgen_iface->inData(ents_to_tetgen, in, moabTetGenMap,
                                tetGenMoabMap, th);
    CHKERR tetgen_iface->setGeomData(in, moabTetGenMap, tetGenMoabMap,
                                     types_ents);
    std::vector<pair<Range, int>> markers;
    for (Range::iterator tit = surface.begin(); tit != surface.end(); tit++) {
      Range facet;
      facet.insert(*tit);
      markers.push_back(pair<Range, int>(facet, 2));
    }
    for (Range::iterator tit = surf_ents.vSkinWithoutBodySkin.begin();
         tit != surf_ents.vSkinWithoutBodySkin.end(); tit++) {
      Range facet;
      facet.insert(*tit);
      markers.push_back(pair<Range, int>(facet, 1));
    }
    Range other_facets;
    other_facets = subtract(surf_ents.vSkin, surf_ents.vSkinWithoutBodySkin);
    for (Range::iterator tit = other_facets.begin(); tit != other_facets.end();
         tit++) {
      Range facet;
      facet.insert(*tit);
      markers.push_back(pair<Range, int>(facet, 0));
    }
    CHKERR tetgen_iface->setFaceData(markers, in, moabTetGenMap, tetGenMoabMap);
  }

  // generate new mesh
  {
    vector<string>::iterator sw = switches.begin();
    for (int ii = 0; sw != switches.end(); sw++, ii++) {
      tetgenio &_in_ = tetGenData.back();
      tetGenData.push_back(new tetgenio);
      tetgenio &_out_ = tetGenData.back();
      char *s = const_cast<char *>(sw->c_str());
      CHKERR tetgen_iface->tetRahedralize(s, _in_, _out_);
    }
  }
  tetgenio &out = tetGenData.back();
  // save elems

  CHKERR tetgen_iface->outData(in, out, moabTetGenMap, tetGenMoabMap, bit,
                               false, false);

  Range rest_of_ents = subtract(bit_ents.mTets, surf_ents.tVols);
  for (int dd = 2; dd >= 0; dd--) {
    CHKERR moab.get_adjacencies(rest_of_ents.subset_by_dimension(3), dd, false,
                                rest_of_ents, moab::Interface::UNION);
  }
  CHKERR m_field.getInterface<BitRefManager>()->addBitRefLevel(rest_of_ents,
                                                               bit);

  Range tetgen_faces;
  map<int, Range> face_markers_map;
  CHKERR tetgen_iface->getTriangleMarkers(tetGenMoabMap, out, &tetgen_faces,
                                          &face_markers_map);

  tetgenSurfaces = face_markers_map[1];
  for (map<int, Range>::iterator mit = face_markers_map.begin();
       mit != face_markers_map.end(); mit++) {
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(
             (*cOre.getInterface<MeshsetsManager>()), SIDESET, it)) {
      int msId = it->getMeshsetId();
      if (id_shift_map[msId] & mit->first) {
        EntityHandle meshset = it->getMeshset();
        CHKERR m_field.get_moab().add_entities(
            meshset, mit->second.subset_by_type(MBTRI));
      }
    }
  }

  MoFEMFunctionReturn(0);
}

#endif // WITH_TETGEN

MoFEMErrorCode CutWithGA::setTagData(Tag th, const BitRefLevel bit) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;
  Range nodes;
  if (bit.none()) {
    CHKERR moab.get_entities_by_type(0, MBVERTEX, nodes);
  } else {
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit, BitRefLevel().set(), MBVERTEX, nodes);
  }
  std::vector<double> coords(3 * nodes.size());
  CHKERR moab.get_coords(nodes, &coords[0]);
  CHKERR moab.tag_set_data(th, nodes, &coords[0]);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::setCoords(Tag th, const BitRefLevel bit,
                                    const BitRefLevel mask) {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;
  Range nodes;
  if (bit.none()) {
    CHKERR moab.get_entities_by_type(0, MBVERTEX, nodes);
  } else {
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit, mask, MBVERTEX, nodes);
  }
  std::vector<double> coords(3 * nodes.size());
  CHKERR moab.tag_get_data(th, nodes, &coords[0]);
  CHKERR moab.set_coords(nodes, &coords[0]);
  MoFEMFunctionReturn(0);
}

struct SaveData {
  moab::Interface &moab;
  SaveData(moab::Interface &moab) : moab(moab) {}
  MoFEMErrorCode operator()(const std::string name, const Range &ents) {
    MoFEMFunctionBeginHot;
    EntityHandle meshset;
    rval = moab.create_meshset(MESHSET_SET, meshset);
    CHKERRG(rval);
    rval = moab.add_entities(meshset, ents);
    CHKERRG(rval);
    rval = moab.write_file(name.c_str(), "VTK", "", &meshset, 1);
    CHKERRG(rval);
    rval = moab.delete_entities(&meshset, 1);
    CHKERRG(rval);
    MoFEMFunctionReturnHot(0);
  }
};

MoFEMErrorCode CutWithGA::saveCutEdges() {
  CoreInterface &m_field = cOre;
  moab::Interface &moab = m_field.get_moab();
  MoFEMFunctionBegin;
  CHKERR SaveData(moab)("out_vol.vtk", vOlume);
  CHKERR SaveData(moab)("out_surface.vtk", sUrface);
  CHKERR SaveData(moab)("out_cut_edges.vtk", cutEdges);
  CHKERR SaveData(moab)("out_cut_volumes.vtk", cutVolumes);
  CHKERR SaveData(moab)("out_cut_new_volumes.vtk", cutNewVolumes);
  CHKERR SaveData(moab)("out_cut_new_surfaces.vtk", cutNewSurfaces);
  CHKERR SaveData(moab)("out_cut_zero_distance_ents.vtk", zeroDistanceEnts);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CutWithGA::saveTrimEdges() {
  moab::Interface &moab = cOre.getInterface<CoreInterface>()->get_moab();
  MoFEMFunctionBegin;
  CHKERR SaveData(moab)("out_trim_new_volumes.vtk", trimNewVolumes);
  CHKERR SaveData(moab)("out_trim_new_surfaces.vtk", trimNewSurfaces);
  CHKERR SaveData(moab)("out_trim_edges.vtk", trimEdges);
  MoFEMFunctionReturn(0);
}




MoFEMErrorCode cutMeshRout(const double tolCut, const double tolCutClose,
                           const double tolTrim, const double tolTrimClose,
                           double &fitness) {
  MoFEMFunctionBegin;

  PetscBool flg_myfile = PETSC_TRUE;
  char mesh_file_name[255];
  int surface_side_set = 200;
  PetscBool flg_vol_block_set;
  int vol_block_set = 1;
  int edges_block_set = 2;
  int vertex_block_set = 3;
  PetscBool flg_shift;
  double shift[] = {0, 0, 0};
  int nmax = 3;
  int fraction_level = 2;
  PetscBool squash_bits = PETSC_TRUE;
  PetscBool set_coords = PETSC_TRUE;
  PetscBool output_vtk = PETSC_TRUE;
  int create_surface_side_set = 201;
  PetscBool flg_create_surface_side_set;

  // optimize params
  // double tolCut = 0.;
  // double tolCutClose = 0.;
  // double tolTrim = 0.;
  // double tolTrimClose = 0.;
  int popMax = 10;
  CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Mesh cut options", "none");
  CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                            mesh_file_name, 255, &flg_myfile);
  CHKERR PetscOptionsInt("-surface_side_set", "surface side set", "",
                         surface_side_set, &surface_side_set, PETSC_NULL);
  CHKERR PetscOptionsInt("-vol_block_set", "volume side set", "", vol_block_set,
                         &vol_block_set, &flg_vol_block_set);
  CHKERR PetscOptionsInt("-edges_block_set", "edges side set", "",
                         edges_block_set, &edges_block_set, PETSC_NULL);
  CHKERR PetscOptionsInt("-vertex_block_set", "vertex side set", "",
                         vertex_block_set, &vertex_block_set, PETSC_NULL);
  CHKERR PetscOptionsRealArray("-shift", "shift surface by vector", "", shift,
                               &nmax, &flg_shift);
  CHKERR PetscOptionsInt("-fraction_level", "fraction of merges merged", "",
                         fraction_level, &fraction_level, PETSC_NULL);
  CHKERR PetscOptionsBool("-squash_bits", "true to squash bits at the end", "",
                          squash_bits, &squash_bits, PETSC_NULL);
  CHKERR PetscOptionsBool("-set_coords", "true to set coords at the end", "",
                          set_coords, &set_coords, PETSC_NULL);
  CHKERR PetscOptionsBool("-output_vtk", "if true outout vtk file", "",
                          output_vtk, &output_vtk, PETSC_NULL);
  CHKERR PetscOptionsInt("-create_side_set", "crete side set", "",
                         create_surface_side_set, &create_surface_side_set,
                         &flg_create_surface_side_set);
  CHKERR PetscOptionsInt("-pop_max", "max size of the population", "", popMax,
                         &popMax, &flg_create_surface_side_set);
  ierr = PetscOptionsEnd();
  CHKERRG(ierr);

  if (flg_myfile != PETSC_TRUE) {
    SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
            "*** ERROR -my_file (MESH FILE NEEDED)");
  }
  if (flg_shift && nmax != 3) {
    SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID, "three values expected");
  }

  moab::Core mb_instance;
  moab::Interface &moab = mb_instance;
  ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
  if (pcomm == NULL)
    pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

  const char *option;
  option = ""; //"PARALLEL=BCAST";//;DEBUG_IO";
  
  std::cout.setstate(std::ios_base::failbit);
  CHKERR moab.load_file(mesh_file_name, 0, option);
  std::cout.clear();

  MoFEM::Core core(moab,PETSC_COMM_WORLD,0);
  MoFEM::CoreInterface &m_field = *(core.getInterface<MoFEM::CoreInterface>());

  // get cut mesh interface
  CutWithGA *cut_mesh = new CutWithGA(core);

  // get meshset manager interface
  MeshsetsManager *meshset_manager;
  CHKERR m_field.getInterface(meshset_manager);
  // get bit ref manager interface
  BitRefManager *bit_ref_manager;
  CHKERR m_field.getInterface(bit_ref_manager);

  BitRefLevel bit_level0;
  bit_level0.set(0);
  CHKERR bit_ref_manager->setBitRefLevelByType(0, MBTET, bit_level0);

  // get surface entities form side set
  Range surface;
  if (meshset_manager->checkMeshset(surface_side_set, SIDESET)) {
    CHKERR meshset_manager->getEntitiesByDimension(surface_side_set, SIDESET, 2,
                                                   surface, true);
  }
  if (surface.empty()) {
    SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "No surface to cut");
  }

  // Set surface entities. If surface entities are from existing side set,
  // copy those entities and do other geometrical transformations, like shift
  // scale or streach, rotate.
  if (meshset_manager->checkMeshset(surface_side_set, SIDESET)) {
    CHKERR cut_mesh->copySurface(surface, NULL, shift);
  } else {
    SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Side set not found %d",
             surface_side_set);
  }

  Range tets;
  if (flg_vol_block_set) {
    if (meshset_manager->checkMeshset(vol_block_set, BLOCKSET)) {
      CHKERR meshset_manager->getEntitiesByDimension(vol_block_set, BLOCKSET, 3,
                                                     tets, true);
    } else {
      SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
               "Block set %d not found", vol_block_set);
    }
  } else {
    CHKERR moab.get_entities_by_dimension(0, 3, tets, false);
  }
  CHKERR cut_mesh->setVolume(tets);

  // GET MINIMUM EDGE LENGTH
  Range all_edges;
  CHKERR moab.get_adjacencies(tets, 1, true, all_edges, moab::Interface::UNION);

  double minEdgeL = 1e16;
  FTensor::Index<'i', 3> i;
  // Store edge nodes coordinates in FTensor
  double edge_node_coords[6];
  FTensor::Tensor1<double *, 3> t_node_edge[2] = {
      FTensor::Tensor1<double *, 3>(edge_node_coords, &edge_node_coords[1],
                                    &edge_node_coords[2]),
      FTensor::Tensor1<double *, 3>(&edge_node_coords[3], &edge_node_coords[4],
                                    &edge_node_coords[5])};
  for (auto edge : all_edges) {
    int num_nodes;
    const EntityHandle *conn;
    CHKERR moab.get_connectivity(edge, conn, num_nodes, true);
    CHKERR moab.get_coords(conn, num_nodes, edge_node_coords);
    t_node_edge[0](i) -= t_node_edge[1](i);
    double l = sqrt(t_node_edge[0](i) * t_node_edge[0](i));
    minEdgeL = (minEdgeL > l) ? l : minEdgeL;
  }

  // PetscPrintf(PETSC_COMM_WORLD, "Min edge length = %6.4g\n", minEdgeL);

  // Build tree, it is used to ask geometrical queries, i.e. to find edges
  // to cut or trim.
  CHKERR cut_mesh->buildTree();

  BitRefLevel bit_level1; // Cut level
  bit_level1.set(1);
  BitRefLevel bit_level2; // Trim level
  bit_level2.set(2);
  BitRefLevel bit_level3; // Merge level
  bit_level3.set(3);
  BitRefLevel bit_level4; // TetGen level
  bit_level4.set(4);

  // Create tag storing nodal positions
  double def_position[] = {0, 0, 0};
  Tag th;
  CHKERR moab.tag_get_handle("POSITION", 3, MB_TYPE_DOUBLE, th,
                             MB_TAG_CREAT | MB_TAG_SPARSE, def_position);
  // Set tag values with coordinates of nodes
  CHKERR cut_mesh->setTagData(th);

  // Get geometric corner nodes and corner edges
  Range fixed_edges, corner_nodes;
  if (meshset_manager->checkMeshset(edges_block_set, BLOCKSET)) {
    CHKERR meshset_manager->getEntitiesByDimension(edges_block_set, BLOCKSET, 1,
                                                   fixed_edges, true);
  }
  if (meshset_manager->checkMeshset(vertex_block_set, BLOCKSET)) {
    CHKERR meshset_manager->getEntitiesByDimension(vertex_block_set, BLOCKSET,
                                                   0, corner_nodes, true);
  }
  
      // GENETIC ALGRORITM LOOP
      // Cut mesh, trim surface and merge bad edges
      CHKERR cut_mesh->cutTrimAndMerge(
          fraction_level, bit_level1, bit_level2, bit_level3, th, tolCut, tolCutClose,
                          tolTrim, tolTrimClose, fitness, fixed_edges, corner_nodes, true, true);


      MoFEMFunctionReturn(0);
    }




} // namespace MoFEM
#endif