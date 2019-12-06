/**
 * \file main_snippet.cpp
 * \example main_snippet.cpp
 *
 * Using Basic interface calculate the divergence of base functions, and
 * integral of flux on the boundary. Since the h-div space is used, volume
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

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    // Create MoAB
    moab::Core mb_instance;              ///< database
    moab::Interface &moab = mb_instance; ///< interface

    // Create MoFEM
    MoFEM::Core core(moab); ///< database
    MoFEM::Interface &m_field = core; ///< interface

    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();
    CHKERR simple->loadFile("");

    CHKERR simple->addDomainField("FIELD", H1,
                                             AINSWORTH_LEGENDRE_BASE, 1);

    constexpr int order = 2;
    CHKERR simple->setFieldOrder("FIELD", order);
    CHKERR simple->setUp();

    Basic *basic = m_field.getInterface<Basic>();

    basic->getOpDomainRhsPipeline().push_back(
        new OpCalculateScalarFieldValues(
            "FIELD", common_data_ptr->rho_at_integration_points));

    auto integration_rule = [](int, int, int p_data) { return p_data + 2; };
    CHKERR basic->setDomainRhsIntegrationRule(integration_rule);
    CHKERR basic->loopFiniteElements();

  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}

