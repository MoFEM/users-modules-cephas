#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <RDOperators.hpp>

using namespace MoFEM;
using namespace ReactionDiffusion;

const double B = 1e-3;
const double B0 = 1e-3;
const double B_epsilon = 0.0;
const double r = 1;


static char help[] = "...\n\n";


int main(int argc, char *argv[])
{
  cout << "Conductivity (B) : " << B << endl;
  return 0;
}