/** \file h_adaptive_transport.cpp
\brief Example implementation of transport problem using mixed formulation

\todo Should be implemented and tested problem from this article
Demkowicz, Leszek, and Jayadeep Gopalakrishnan. "Analysis of the DPG method for
the Poisson equation." SIAM Journal on Numerical Analysis 49.5 (2011):
1788-1809.

\ingroup mofem_mix_transport_elem
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

#include <BasicFiniteElements.hpp>
#include <MixTransportElement.hpp>

using namespace MoFEM;
using namespace MixTransport;

static char help[] = "-my_file input file"
                     "-my_order order of approximation"
                     "-nb_levels number of refinements levels"
                     "\n\n";

/**
 * Data structure to pass information between function evaluating boundary
 * values and fluxes and generic data structures for boundary conditions on
 * meshsets.
 */
struct BcFluxData {
  Range eNts;
  double fLux;
};
typedef map<int, BcFluxData> BcFluxMap;

/** \brief Application of mix transport data structure
  *
  * MixTransportElement is a class collecting functions, operators and
  * data for mix implementation of transport element. See there to
  * learn how elements are created or how operators look like.
  *
  * Some methods in MixTransportElement are abstract, f.e. user need to
  * implement own source therm.

  * \ingroup mofem_mix_transport_elem
  */
struct MyTransport : public MixTransportElement {

  BcFluxMap &bcFluxMap;
  EntityHandle lastEnt;
  double lastFlux;

  MyTransport(MoFEM::Interface &m_field, BcFluxMap &bc_flux_map)
      : MixTransportElement(m_field), bcFluxMap(bc_flux_map), lastEnt(0),
        lastFlux(0) {}

  /**
   * \brief set source term
   * @param  ent  handle to entity on which function is evaluated
   * @param  x    coord
   * @param  y    coord
   * @param  z    coord
   * @param  flux reference to source term set by function
   * @return      error code
   */
  MoFEMErrorCode getSource(EntityHandle ent, const double x, const double y,
                           const double z, double &flux) {
    MoFEMFunctionBeginHot;
    flux = 0;
    MoFEMFunctionReturnHot(0);
  }

  /**
   * \brief natural (Dirihlet) boundary conditions (set values)
   * @param  ent   handle to finite element entity
   * @param  x     coord
   * @param  y     coord
   * @param  z     coord
   * @param  value reference to value set by function
   * @return       error code
   */
  MoFEMErrorCode getBcOnValues(const EntityHandle ent, const double x,
                               const double y, const double z, double &value) {
    MoFEMFunctionBeginHot;
    value = 0;
    MoFEMFunctionReturnHot(0);
  }

  /**
   * \brief essential (Neumann) boundary condition (set fluxes)
   * @param  ent  handle to finite element entity
   * @param  x    coord
   * @param  y    coord
   * @param  z    coord
   * @param  flux reference to flux which is set by function
   * @return      [description]
   */
  MoFEMErrorCode getBcOnFluxes(const EntityHandle ent, const double x,
                               const double y, const double z, double &flux) {
    MoFEMFunctionBeginHot;
    if (lastEnt == ent) {
      flux = lastFlux;
    } else {
      flux = 0;
      for (BcFluxMap::iterator mit = bcFluxMap.begin(); mit != bcFluxMap.end();
           mit++) {
        Range &tris = mit->second.eNts;
        if (tris.find(ent) != tris.end()) {
          flux = mit->second.fLux;
        }
      }
      lastEnt = ent;
      lastFlux = flux;
    }
    MoFEMFunctionReturnHot(0);
  }

  /**
   * \brief set-up boundary conditions
   * @param  ref_level mesh refinement level

   \note It is assumed that user would like to something non-standard with
   boundary conditions, have a own type of data structures to pass to functions
   calculating values and fluxes on boundary. For example BcFluxMap. That way
   this function is implemented here not in generic class MixTransportElement.

   * @return           error code
   */
  MoFEMErrorCode addBoundaryElements(BitRefLevel &ref_level) {
    MoFEMFunctionBegin;
    Range tets;
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        ref_level, BitRefLevel().set(), MBTET, tets);
    Skinner skin(&mField.get_moab());
    Range skin_faces; // skin faces from 3d ents
    CHKERR skin.find_skin(0, tets, false, skin_faces);
    // note: what is essential (dirichlet) is natural (neumann) for mix-FE
    // compared to classical FE
    Range natural_bc;
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, NODESET | TEMPERATURESET, it)) {
      Range tris;
      CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 2, tris,
                                                 true);
      natural_bc.insert(tris.begin(), tris.end());
    }
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, SIDESET | HEATFLUXSET, it)) {
      HeatFluxCubitBcData mydata;
      CHKERR it->getBcDataStructure(mydata);
      if (mydata.data.flag1 == 1) {
        Range tris;
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 2, tris,
                                                   true);
        bcFluxMap[it->getMeshsetId()].eNts = tris;
        bcFluxMap[it->getMeshsetId()].fLux = mydata.data.value1;
        // cerr << bcFluxMap[it->getMeshsetId()].eNts << endl;
        // cerr << bcFluxMap[it->getMeshsetId()].fLux << endl;
      }
    }
    Range essential_bc = subtract(skin_faces, natural_bc);
    Range bit_tris;
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        ref_level, BitRefLevel().set(), MBTRI, bit_tris);
    essential_bc = intersect(bit_tris, essential_bc);
    natural_bc = intersect(bit_tris, natural_bc);
    CHKERR mField.add_ents_to_finite_element_by_type(essential_bc, MBTRI,
                                                     "MIX_BCFLUX");
    CHKERR mField.add_ents_to_finite_element_by_type(natural_bc, MBTRI,
                                                     "MIX_BCVALUE");
    // CHKERR
    // mField.add_ents_to_finite_element_by_type(skin_faces,MBTRI,"MIX_BCVALUE");
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Refine mesh
   * @param  ufe       general data structure
   * @param  nb_levels number of refinement levels
   * @param  order     set order of approximation
   * @return           errpr code

   Refinement of could result in distorted mesh, for example, imagine when you
   have two levels of non-uniform refinement. Some tetrahedra on the mesh at
   first refinement instance are only refined by splitting subset of edges on
   it. Then refined child tetrahedra usually will have worse quality than
   quality of parent element. Refining such element in subsequent mesh
   refinement, potentially will deteriorate elements quality even worse. To
   prevent that adding new refinement level, recreate whole hierarchy of meshes.

   Note on subsequent improvement could include refinement of
   tetrahedra from different levels, including initial mesh. So refinement two
   could split elements created during refinement one and also split elements
   from an initial mesh.

   That adding the new refinement level creates refinement hierarchy of meshes
   from a scratch,  not adding to existing one.

   Entities from previous hierarchy are used in that process, but bit levels on
   those entities are squashed.

   */
  MoFEMErrorCode refineMesh(MixTransportElement &ufe, const int nb_levels,
                            const int order) {
    MeshRefinement *refine_ptr;
    MoFEMFunctionBegin;
    // get refined edges having child vertex
    auto ref_ents_ptr = mField.get_ref_ents();
    typedef RefEntity_multiIndex::index<
        Composite_EntType_and_ParentEntType_mi_tag>::type RefEntsByComposite;
    const RefEntsByComposite &ref_ents =
        ref_ents_ptr->get<Composite_EntType_and_ParentEntType_mi_tag>();
    RefEntsByComposite::iterator rit, hi_rit;
    rit = ref_ents.lower_bound(boost::make_tuple(MBVERTEX, MBEDGE));
    hi_rit = ref_ents.upper_bound(boost::make_tuple(MBVERTEX, MBEDGE));
    Range refined_edges;
    // thist loop is over vertices which parent is edge
    for (; rit != hi_rit; rit++) {
      refined_edges.insert((*rit)->getParentEnt()); // get parent edge
    }
    // get tets which has large error
    Range tets_to_refine;
    const double max_error = ufe.errorMap.rbegin()->first;
    // int size = ((double)5/6)*ufe.errorMap.size();
    for (map<double, EntityHandle>::iterator mit = ufe.errorMap.begin();
         mit != ufe.errorMap.end(); mit++) {
      // cerr << mit->first << " " << mit->second << endl;
      // if((size--)>0) continue;
      if (mit->first < 0.25 * max_error)
        continue;
      tets_to_refine.insert(mit->second);
    }
    Range tets_to_refine_edges;
    CHKERR mField.get_moab().get_adjacencies(
        tets_to_refine, 1, false, tets_to_refine_edges, moab::Interface::UNION);
    refined_edges.merge(tets_to_refine_edges);
    CHKERR mField.getInterface(refine_ptr);
    for (int ll = 0; ll != nb_levels; ll++) {
      Range edges;
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
          BitRefLevel().set(ll), BitRefLevel().set(), MBEDGE, edges);
      edges = intersect(edges, refined_edges);
      // add edges to refine at current level edges (some of the where refined
      // before)
      CHKERR refine_ptr->addVerticesInTheMiddleOfEdges(
          edges, BitRefLevel().set(ll + 1));
      //  get tets at current level
      Range tets;
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
          BitRefLevel().set(ll), BitRefLevel().set(), MBTET, tets);
      CHKERR refine_ptr->refineTets(tets, BitRefLevel().set(ll + 1));
      CHKERR updateMeshsetsFieldsAndElements(ll + 1);
    }

    // update fields and elements
    EntityHandle ref_meshset;
    CHKERR mField.get_moab().create_meshset(MESHSET_SET, ref_meshset);
    {
      // cerr << BitRefLevel().set(nb_levels) << endl;
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
          BitRefLevel().set(nb_levels), BitRefLevel().set(), MBTET,
          ref_meshset);

      Range ref_tets;
      CHKERR mField.get_moab().get_entities_by_type(ref_meshset, MBTET,
                                                    ref_tets);

      // add entities to field
      CHKERR mField.add_ents_to_field_by_type(ref_meshset, MBTET, "FLUXES");
      CHKERR mField.add_ents_to_field_by_type(ref_meshset, MBTET, "VALUES");
      CHKERR mField.set_field_order(0, MBTET, "FLUXES", order + 1);
      CHKERR mField.set_field_order(0, MBTRI, "FLUXES", order + 1);
      CHKERR mField.set_field_order(0, MBTET, "VALUES", order);

      // add entities to skeleton
      Range ref_tris;
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
          BitRefLevel().set(nb_levels), BitRefLevel().set(), MBTRI, ref_tris);
      CHKERR mField.add_ents_to_finite_element_by_type(ref_tris, MBTRI,
                                                       "MIX_SKELETON");

      // add entities to finite elements
      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
               mField, BLOCKSET | MAT_THERMALSET, it)) {
        Mat_Thermal temp_data;
        CHKERR it->getAttributeDataStructure(temp_data);
        setOfBlocks[it->getMeshsetId()].cOnductivity =
            temp_data.data.Conductivity;
        setOfBlocks[it->getMeshsetId()].cApacity = temp_data.data.HeatCapacity;
        CHKERR mField.get_moab().get_entities_by_type(
            it->meshset, MBTET, setOfBlocks[it->getMeshsetId()].tEts, true);
        setOfBlocks[it->getMeshsetId()].tEts =
            intersect(ref_tets, setOfBlocks[it->getMeshsetId()].tEts);
        CHKERR mField.add_ents_to_finite_element_by_type(
            setOfBlocks[it->getMeshsetId()].tEts, MBTET, "MIX");
      }
    }
    CHKERR mField.get_moab().delete_entities(&ref_meshset, 1);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Squash bits of entities

   Information about hierarchy of meshsets is lost, but entities are not deleted
   from the mesh. After squash entities bits, new hierarchy can be created.

   * @return error code
   */
  MoFEMErrorCode squashBits() {
    MoFEMFunctionBegin;
    BitRefLevel all_but_0;
    all_but_0.set(0);
    all_but_0.flip();
    BitRefLevel garbage_bit;
    garbage_bit.set(BITREFLEVEL_SIZE - 1); // Garbage level
    auto ref_ents_ptr = mField.get_ref_ents();
    RefEntity_multiIndex::iterator mit = ref_ents_ptr->begin();
    for (; mit != ref_ents_ptr->end(); mit++) {
      if (mit->get()->getEntType() == MBENTITYSET)
        continue;
      BitRefLevel bit = mit->get()->getBitRefLevel();
      if ((all_but_0 & bit) == bit) {
        *(const_cast<RefEntity *>(mit->get())->getBitRefLevelPtr()) =
            garbage_bit;
      } else {
        *(const_cast<RefEntity *>(mit->get())->getBitRefLevelPtr()) =
            BitRefLevel().set(0);
      }
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief update meshsets with new entities after mesh refinement
   * @param  nb_levels nb_levels
   * @param  order     appropriate order
   * @return           error code
   */
  MoFEMErrorCode updateMeshsetsFieldsAndElements(const int nb_levels) {
    BitRefLevel ref_level;
    MoFEMFunctionBegin;
    ref_level.set(nb_levels);
    for (_IT_CUBITMESHSETS_FOR_LOOP_(mField, it)) {
      EntityHandle meshset = it->meshset;
      CHKERR mField.getInterface<BitRefManager>()
          ->updateMeshsetByEntitiesChildren(meshset, ref_level, meshset,
                                            MBVERTEX, true);
      CHKERR mField.getInterface<BitRefManager>()
          ->updateMeshsetByEntitiesChildren(meshset, ref_level, meshset, MBEDGE,
                                            true);
      CHKERR mField.getInterface<BitRefManager>()
          ->updateMeshsetByEntitiesChildren(meshset, ref_level, meshset, MBTRI,
                                            true);
      CHKERR mField.getInterface<BitRefManager>()
          ->updateMeshsetByEntitiesChildren(meshset, ref_level, meshset, MBTET,
                                            true);
    }
    MoFEMFunctionReturn(0);
  }
};

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-ksp_monitor\n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // get file name form command line
    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                                 mesh_file_name, 255, &flg);
    if (flg != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create mofem interface
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // Add meshsets with material and boundary conditions
    MeshsetsManager *meshsets_manager_ptr;
    CHKERR m_field.getInterface(meshsets_manager_ptr);
    CHKERR meshsets_manager_ptr->setMeshsetFromFile();

    PetscPrintf(PETSC_COMM_WORLD,
                "Read meshsets add added meshsets for bc.cfg\n");
    for (_IT_CUBITMESHSETS_FOR_LOOP_(m_field, it)) {
      PetscPrintf(PETSC_COMM_WORLD, "%s",
                  static_cast<std::ostringstream &>(
                      std::ostringstream().seekp(0) << *it << endl)
                      .str()
                      .c_str());
      cerr << *it << endl;
    }

    // set entities bit level
    BitRefLevel ref_level;
    ref_level.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, ref_level);

    // set app. order
    // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
    // (Mark Ainsworth & Joe Coyle)
    PetscInt order;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_order", &order,
                              &flg);
    if (flg != PETSC_TRUE) {
      order = 0;
    }

    // finite elements

    BcFluxMap bc_flux_map;
    MyTransport ufe(m_field, bc_flux_map);

    // Initially calculate problem on coarse mesh

    CHKERR ufe.addFields("VALUES", "FLUXES", order);
    CHKERR ufe.addFiniteElements("FLUXES", "VALUES");
    // Set boundary conditions
    CHKERR ufe.addBoundaryElements(ref_level);
    CHKERR ufe.buildProblem(ref_level);
    CHKERR ufe.createMatrices();
    CHKERR ufe.solveLinearProblem();
    CHKERR ufe.calculateResidual();
    CHKERR ufe.evaluateError();
    CHKERR ufe.destroyMatrices();
    CHKERR ufe.postProc("out_0.h5m");

    int nb_levels = 5; // default number of refinement levels
    // get number of refinement levels form command line
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-nb_levels", &nb_levels,
                              PETSC_NULL);

    // refine mesh, solve problem and do it again until number of refinement
    // levels are exceeded.
    for (int ll = 1; ll != nb_levels; ll++) {
      const int nb_levels = ll;
      CHKERR ufe.squashBits();
      CHKERR ufe.refineMesh(ufe, nb_levels, order);
      ref_level = BitRefLevel().set(nb_levels);
      bc_flux_map.clear();
      CHKERR ufe.addBoundaryElements(ref_level);
      CHKERR ufe.buildProblem(ref_level);
      CHKERR ufe.createMatrices();
      CHKERR ufe.solveLinearProblem();
      CHKERR ufe.calculateResidual();
      CHKERR ufe.evaluateError();
      CHKERR ufe.destroyMatrices();
      CHKERR ufe.postProc(
          static_cast<std::ostringstream &>(std::ostringstream().seekp(0)
                                            << "out_" << nb_levels << ".h5m")
              .str());
    }
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
