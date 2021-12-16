#include <mlsm/internal/Shape.h>

using namespace mls::internal;

int main()
{
  std::shared_ptr<ShapeBase> e = std::make_shared<EmptyShape>(3, 5);
  std::shared_ptr<ShapeBase> b = std::make_shared<BandShape>(5, 5, 1, 1);
  std::shared_ptr<ShapeBase> d = std::make_shared<DenseShape>(5, 3);

  auto eb = mult(*e, *b);
  auto bs = mult(*b, *d);

  return 0;
}