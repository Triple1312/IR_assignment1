

abstract class Tensor<T extends num> {

  List<int> get shape;

  T get(List<int> indices);

  void set(List<int> indices, T value);

  Tensor();

  num norm(int p);

  num infinityNorm();

}
