
import 'dart:math';


import 'Tensor.dart';


class Vector<T extends num> implements Tensor<T> , List<T> {

  List<T> _base = [];

  Vector.empty() : _base = [];

  Vector.zeroes(int length) : _base = List<T>.filled(length, 0 as T);

  // factory Vector.sparse_zeroes(int length) => SparceVector.zeroes(length);
  //
  // factory Vector.sparce(List<T> base) => SparceVector(base);

  Vector(List<T> values) : _base = values;

  List<T> get flat => _base;



  @override
  T get(List<int> indices) {
    if (indices.length != 1) {
      throw Exception("Invalid number of indices");
    }
    return _base[indices[0]];
  }

  @override
  void set(List<int> indices, T value) {
    // TODO: implement set
  }

  void extend(int size) {
    while (size > length) {
      _base.add(0 as T);
    }
  }

  void addExtend(int index, T value) {
    if (index >= length) {
      extend(index + 1);
    }
    _base[index] = value;
  }

  ///////////////////////// operators /////////////////////////

  Vector<num> operator *(num scalar) {
    return Vector<num>(flat.map((e) => e * scalar).toList());
  }

  Vector<num> operator /(num scalar) {
    return Vector<num>(flat.map((e) => e / scalar).toList());
  }

  bool operator ==(Object other) {
    if (other is Vector) {
      return flat == other.flat;
    }
    else if(other is List) {
      return other == flat;
    }
    return false;
  }


  ///////////////////////// Math /////////////////////////

  Vector<num> normalize() {
    return this / norm(2);
  }

  num dot(Vector v) {
    if (shape[0] != v.shape[0]) {
      throw Exception("Vectors must be same size");
    }
    num x = List.generate(shape[0], (i) => this[i] * v[i]).fold(0, (a, b) => a + b);
    return List.generate(shape[0], (i) => this[i] * v[i]).fold(0, (a, b) => a + b);
  }

  Vector proj(Vector v) {
    return v * (v.dot(this) / v.dot(v));
  }

  T get sum => flat.fold(0 as T, (a, b) => (a + b) as T);

  T get product => this.fold(1 as T, (a, b) => (a * b) as T);

  num get mean => sum / length;

  num get median => {
    if (length % 2 == 0) {
      (flat[length ~/ 2] + flat[(length ~/ 2) + 1]) / 2
    } else {
      flat[length ~/ 2]
    }
  } as num;

  Vector<num> minusVector(Vector<num> other) {
    if (length != other.length) {
      throw Exception("Vectors must be same size");
    }
    return Vector<num>(List.generate(length, (i) => this[i] - other[i] ));
  }

  Vector<num> plusVector(Vector<num> other) {
    if (length != other.length) {
      throw Exception("Vectors must be same size");
    }
    return Vector<num>(List.generate(length, (i) => this[i] + other[i]));
  }

  @override
  num infinityNorm() {
    // TODO: implement infinityNorm
    throw UnimplementedError();
  }

  @override
  num norm(int p) {
    return pow(this.fold(0, (a,b) => a+ pow(b.abs(),p)), 1/p);
  }

  num norm2() {
    return sqrt(flat.fold(0, (a, b) => a + b*b));
  }


  ///////////////////////// List overrides /////////////////////////

  @override
  bool any(bool Function(T) test) => flat.any(test);

  @override
  List<T> cast<T>() => flat.cast<T>();

  @override
  bool contains(Object? element) => flat.contains(element);

  @override
  T elementAt(int index) => flat.elementAt(index);

  @override
  bool every(bool Function(T) test) => flat.every(test);

  @override
  Iterable<K> expand<K>(Iterable<K> Function(T) f) => flat.expand(f);

  @override
  T get first => flat.first;

  @override
  T firstWhere(bool Function(T) test, {T Function()? orElse}) =>
      flat.firstWhere(test, orElse: orElse);

  @override
  E fold<E>(E initialValue, E Function(E previousValue, T element) combine) =>
      flat.fold(initialValue, combine);

  @override
  Iterable<T> followedBy(Iterable<T> other) => flat.followedBy(other);

  @override
  void forEach(void Function(T) f) => flat.forEach(f);

  @override
  bool get isEmpty => flat.isEmpty;

  @override
  bool get isNotEmpty => flat.isNotEmpty;

  @override
  Iterator<T> get iterator => flat.iterator;

  @override
  String join([String separator = '']) => flat.join(separator);

  @override
  T get last => flat.last;

  @override
  T lastWhere(bool Function(T) test, {T Function()? orElse}) =>
      flat.lastWhere(test, orElse: orElse);

  @override
  int get length => flat.length;

  @override
  Iterable<E> map<E>(E Function(T) f) => flat.map(f);

  @override
  T reduce(T Function(T value, T element) combine) => flat.reduce(combine);

  @override
  T get single => flat.single;

  @override
  T singleWhere(bool Function(T) test, {T Function()? orElse}) {
    return flat.singleWhere(test, orElse: orElse);
  }

  @override
  Iterable<T> skip(int n) => flat.skip(n);

  @override
  Iterable<T> skipWhile(bool Function(T) test) => flat.skipWhile(test);

  @override
  Iterable<T> take(int n) => flat.take(n);

  @override
  Iterable<T> takeWhile(bool Function(T) test) => flat.takeWhile(test);

  @override
  List<T> toList({bool growable = true}) => flat.toList(growable: growable);

  @override
  Set<T> toSet() => flat.toSet();

  @override
  Iterable<T> where(bool Function(T) test) => flat.where(test);

  @override
  Iterable<T> whereType<T>() => flat.whereType<T>();

  @override
  String toString() => flat.toString();

  @override
  T operator [](int index) => flat[index];

  @override
  void operator []=(int index, T value) {
    flat[index] = value;
  }

  @override
  List<T> operator +(List<T> other) => flat + other;

  @override
  void add(T value) {
    flat.add(value);
  }

  @override
  void addAll(Iterable<T> iterable) {
    flat.addAll(iterable);
  }

  @override
  Map<int, T> asMap() => flat.asMap();

  @override
  void clear() {
    flat.clear();
  }

  @override
  void fillRange(int start, int end, [T? fillValue]) {
    flat.fillRange(start, end, fillValue);
  }

  @override
  set first(T value) {
    if (isEmpty) throw RangeError.index(0, this);
    this[0] = value;
  }

  @override
  Iterable<T> getRange(int start, int end) => flat.getRange(start, end);

  @override
  int indexOf(T element, [int start = 0]) => flat.indexOf(element, start);

  @override
  int indexWhere(bool Function(T) test, [int start = 0]) =>
      flat.indexWhere(test, start);

  @override
  void insert(int index, T element) {
    flat.insert(index, element);
  }

  @override
  void insertAll(int index, Iterable<T> iterable) {
    flat.insertAll(index, iterable);
  }

  @override
  set last(T value) {
    if (isEmpty) throw RangeError.index(0, this);
    this[length - 1] = value;
  }

  @override
  int lastIndexOf(T element, [int? start]) => flat.lastIndexOf(element, start);

  @override
  int lastIndexWhere(bool Function(T) test, [int? start]) =>
      flat.lastIndexWhere(test, start);

  @override
  set length(int newLength) {
    flat.length = newLength;
  }

  @override
  bool remove(Object? value) => flat.remove(value);

  @override
  T removeAt(int index) => flat.removeAt(index);

  @override
  T removeLast() => flat.removeLast();

  @override
  void removeRange(int start, int end) {
    flat.removeRange(start, end);
  }

  @override
  void removeWhere(bool Function(T) test) {
    flat.removeWhere(test);
  }

  @override
  void replaceRange(int start, int end, Iterable<T> iterable) {
    flat.replaceRange(start, end, iterable);
  }

  @override
  void retainWhere(bool Function(T) test) {
    flat.retainWhere(test);
  }

  @override
  Iterable<T> get reversed => flat.reversed;

  @override
  void setAll(int index, Iterable<T> iterable) {
    flat.setAll(index, iterable);
  }

  @override
  void setRange(int start, int end, Iterable<T> iterable, [int skipCount = 0]) {
    flat.setRange(start, end, iterable, skipCount);
  }

  @override
  void shuffle([Random? random]) {
    flat.shuffle(random);
  }

  @override
  void sort([int Function(T, T)? compare]) {
    flat.sort(compare);
  }

  @override
  List<T> sublist(int start, [int? end]) => flat.sublist(start, end);

  @override
  List<int> get shape => [flat.length];

}


class SparceVector<T extends num> extends Vector<T> {

  // this is not sorted rn
  List<int> _indices = [];

  int length;

  SparceVector.zeroes(this.length) : super([]);

  SparceVector.empty() : this.length = 0, super.empty();

  SparceVector.map(Map<int, T> map, int length) : this.length = length, super.empty() {
    for (int i in map.keys) {
      if (map[i] != 0) {
        _base.add(map[i] as T);
        _indices.add(i);
      }
    }
  }

  num dot(Vector<num> v) {
    num sum = 0;
    for (int i = 0; i < _indices.length; i++) {
      sum += v[_indices[i]] * _base[i];
    }
    return sum;
  }

  @override
  num norm(int p) {
    num ret = 0;
    for (int i = 0; i < _base.length; i++) {
      ret += pow(_base[i].abs(), p);
    }
    return pow(ret, 1/p);
  }

  get flat {
    List<T> values = List<T>.filled(length, 0 as T);
    for (int i = 0; i < _indices.length; i++) {
      values[_indices[i]] = _base[i];
    }
    return values;
  }

  @override
  Vector<num> operator /(num scalar) {
    SparceVector<num> v = SparceVector<num>.zeroes(length);
    v._indices = _indices;
    for (int i = 0; i < _base.length; i++) {
      v._base.add(_base[i] / scalar);
    }
    return v;
  }


  SparceVector(List<T> base) : this.length = base.length, super.empty() {
    for (int i = 0; i < base.length; i++) {
      if (base[i] != 0) {
        _base.add(base[i]);
        _indices.add(i);
      }
    }
  }

  SparceSparceVectorIterable<T> get sparceIterator => new SparceSparceVectorIterable(this);

  SparceVector.sparce(List<T> values, this._indices, this.length) : super(values);


  @override
  void extend(int size) {
    this.length = size;
  }

  operator [](int index) {
    if (_indices.contains(index)) {
      return _base[_indices.indexOf(index)];
    }
    return 0.0 as T;
  }

  // not sorted, but doesnt matter
  operator []=(int index, T value) {
    if (value == 0) {
      if (_indices.contains(index)) {
        _base.removeAt(_indices.indexOf(index));
        _indices.remove(index);
      }
      return;
    }
    if (_indices.contains(index)) {
      _base[_indices.indexOf(index)] = value;
    } else {
      _base.add(value);
      _indices.add(index);
    }
  }

  void add(T value) {
    if (value == 0) {
      length++;
      return;
    }
    _base.add(value);
    _indices.add(length);
    length++;
  }

  Iterator<T> get iterator => new SparceVectorIterator(this);

  void addExtend(int index, T value) {
    if (index > length) length = index;
    this[index] = value;
  }

  num norm2() {
    num ret = 0;
    for (int i = 0; i < _base.length; i++) {
      T tmp = _base[i];
      ret += tmp*tmp;
    }
    return sqrt(ret);
  }
}


class MapVector<T extends num> extends Vector<T> {
  Map<int, T> data;

  int length;

  MapVector() : data = {}, length = 0, super.empty();

  MapVector.map(this.data, this.length) : super.empty();

  num dot(Vector<num> v) {
    num sum = 0;
    for (var x in data.keys) {
      sum += v[x] * data[x]!;
    }
    return sum;
  }
  num norm(int p) {
    num ret = 0;
    for (var x in data.keys) {
      ret += pow(data[x]!.abs(), p);
    }
    return pow(ret, 1/p);
  }

  num norm2() {
    num ret = 0;
    for (var x in data.keys) {
      T tmp = data[x]!;
      ret += tmp*tmp;
    }
    return sqrt(ret);
  }

  Vector<num> operator /(num scalar) {
    Map<int, num> new_data = {};
    for (var x in data.keys) {
      new_data[x] = data[x]! / scalar;
    }
    return MapVector<num>.map(new_data, length);
  }

  Vector<num> diveq(num scalar) {
    data.map((key, value) => MapEntry(key, value / scalar));
    return this;
  }

  operator [](int index) {
    return data[index]?? 0.0 as T;
  }

}

class SparceVectorIterator<T extends num> implements Iterator<T> {

  int _currentIndex = -1;

  SparceVector<T> _vector;

  SparceVectorIterator(this._vector);

  @override
  bool moveNext() {
    if (_currentIndex < _vector.length - 1) {
      _currentIndex++;
      return true;
    }
    return false;
  }


  @override
  get current => _vector[_currentIndex];
}

class SparceSparceVectorIterable<T extends num> extends Iterable<(int, T)> {

  SparceVector<T> vector;

  SparceSparceVectorIterable(this.vector);

  @override
  Iterator<(int, T)> get iterator => SparceSparceVectorIterator(vector);

}


class SparceSparceVectorIterator<T extends num> implements Iterator<(int, T)> {

  SparceVector<T> _vector;

  int _index = -1;

  SparceSparceVectorIterator(this._vector);

  @override
  (int, T) get current => (_vector._indices[_index], _vector._base[_index]);

  @override
  bool moveNext() {
    if (_index < _vector._indices.length - 1) {
      _index++;
      return true;
    }
    return false;
  }

}