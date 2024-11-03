

import 'dart:math';

class DFColumn<T> implements List<T?> {

  String columnName;

  List<T?> _base = [];

  Type get type => DFColumn<T>;

  Type get baseType => T;

  List<T> get unique => (_base.toSet().where((e) => e != null) as Set<T>).toList();

  List<T> get vocabulary => unique;

  DFColumn(this.columnName, [this._base = const []]);

  DFColumn<T> copy() => DFColumn(columnName, toList(growable: true));


  /// Returns if all elements are not null
  bool all() {
    for (var element in _base) {
      if (element == null) return false;
    }
    return true;
  }

  bool anyNotNull() {
    return any((e) => e != null);
  }

  bool operator==(Object other) {
    if (other is DFColumn) {
      return _base == other._base;
    }
    else if(other is List) {
      for (int i = 0; i < length; i++) {
        if (this[i] != other[i]) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
  
  
  void fillna(T value) {
    if (value == null) throw Exception('Value cannot be null in fillna');
    for (int i = 0; i < length; i++) {
      if (this[i] == null) {
        this[i] = value;
      }
    }
  }
  
  List<int> naLocations() {
    List<int> locations = [];
    for (int i = 0; i < length; i++) {
      if (this[i] == null) {
        locations.add(i);
      }
    }
    return locations;
  }

  Map<int, T?> to_dict() {
    Map<int, T?> res = {};
    for (int i = 0; i < length; i++) {
      res[i] = _base[i];
    }
    return res;
  }

  int count() {
    int count = 0;
    for (var element in _base) {
      if (element != null) count++;
    }
    return count;
  }
  
  List<String> asString() => _base.map((e) => e.toString()).toList();

  List<int?> asInt() => _base.map((e) => e != null? baseType == String? int.tryParse(e as String) : (e as double).floor() : null).toList();

  List<double?> asDouble() => _base.map((e) => e != null? baseType == String? double.tryParse(e as String) : e as double : null).toList();

  List<bool> asBool() => _base.map((e) => e != 'false' && e != '0' && e != '' && e != 'null' && e != 'False' && e != 0 && e != 0.0).toList();



  //////////////////// list methods ////////////////////

  @override
  bool any(bool Function(T?) test) => _base.any(test);

  @override
  List<T> cast<T>() => _base.cast<T>();

  @override
  bool contains(Object? element) => _base.contains(element);

  @override
  T? elementAt(int index) => _base.elementAt(index);

  @override
  bool every(bool Function(T?) test) => _base.every(test);

  @override
  Iterable<K> expand<K>(Iterable<K> Function(T?) f) => _base.expand(f);

  @override
  T? get first => _base.first;

  @override
  T? firstWhere(bool Function(T?) test, {T? Function()? orElse}) =>
      _base.firstWhere(test, orElse: orElse);

  @override
  E fold<E>(E initialValue, E Function(E previousValue, T? element) combine) =>
      _base.fold(initialValue, combine);

  @override
  Iterable<T?> followedBy(Iterable<T?> other) => _base.followedBy(other);

  @override
  void forEach(void Function(T?) f) => _base.forEach(f);

  @override
  bool get isEmpty => _base.isEmpty;

  @override
  bool get isNotEmpty => _base.isNotEmpty;

  @override
  Iterator<T?> get iterator => _base.iterator;

  @override
  String join([String separator = '']) => _base.join(separator);

  @override
  T? get last => _base.last;

  @override
  T? lastWhere(bool Function(T?) test, {T? Function()? orElse}) =>
      _base.lastWhere(test, orElse: orElse);

  @override
  int get length => _base.length;

  @override
  Iterable<E> map<E>(E Function(T?) f) => _base.map(f);

  @override
  T? reduce(T? Function(T? value, T? element) combine) => _base.reduce(combine);

  @override
  T? get single => _base.single;

  @override
  T? singleWhere(bool Function(T?) test, {T? Function()? orElse}) {
    return _base.singleWhere(test, orElse: orElse);
  }

  @override
  Iterable<T?> skip(int n) => _base.skip(n);

  @override
  Iterable<T?> skipWhile(bool Function(T?) test) => _base.skipWhile(test);

  @override
  Iterable<T?> take(int n) => _base.take(n);

  @override
  Iterable<T?> takeWhile(bool Function(T?) test) => _base.takeWhile(test);

  @override
  List<T?> toList({bool growable = true}) => _base.toList(growable: growable);

  @override
  Set<T?> toSet() => _base.toSet();

  @override
  Iterable<T?> where(bool Function(T?) test) => _base.where(test);

  @override
  Iterable<T> whereType<T>() => _base.whereType<T>();

  @override
  String toString() => _base.toString();

  @override
  T? operator [](int index) => _base[index];

  @override
  void operator []=(int index, T? value) {
    _base[index] = value;
  }

  @override
  List<T?> operator +(List<T?> other) => _base + other;

  @override
  void add(T? value) {
    _base.add(value);
  }

  @override
  void addAll(Iterable<T?> iterable) {
    _base.addAll(iterable);
  }

  @override
  Map<int, T?> asMap() => _base.asMap();

  @override
  void clear() {
    _base.clear();
  }

  @override
  void fillRange(int start, int end, [T? fillValue]) {
    _base.fillRange(start, end, fillValue);
  }

  @override
  set first(T? value) {
    if (isEmpty) throw RangeError.index(0, this);
    this[0] = value;
  }

  @override
  Iterable<T?> getRange(int start, int end) => _base.getRange(start, end);

  @override
  int indexOf(T? element, [int start = 0]) => _base.indexOf(element, start);

  @override
  int indexWhere(bool Function(T?) test, [int start = 0]) =>
      _base.indexWhere(test, start);

  @override
  void insert(int index, T? element) {
    _base.insert(index, element);
  }

  @override
  void insertAll(int index, Iterable<T?> iterable) {
    _base.insertAll(index, iterable);
  }

  @override
  set last(T? value) {
    if (isEmpty) throw RangeError.index(0, this);
    this[length - 1] = value;
  }

  @override
  int lastIndexOf(T? element, [int? start]) => _base.lastIndexOf(element, start);

  @override
  int lastIndexWhere(bool Function(T?) test, [int? start]) =>
      _base.lastIndexWhere(test, start);

  @override
  set length(int newLength) {
    _base.length = newLength;
  }

  @override
  bool remove(Object? value) => _base.remove(value);

  @override
  T? removeAt(int index) => _base.removeAt(index);

  @override
  T? removeLast() => _base.removeLast();

  @override
  void removeRange(int start, int end) {
    _base.removeRange(start, end);
  }

  @override
  void removeWhere(bool Function(T?) test) {
    _base.removeWhere(test);
  }

  @override
  void replaceRange(int start, int end, Iterable<T?> iterable) {
    _base.replaceRange(start, end, iterable);
  }

  @override
  void retainWhere(bool Function(T?) test) {
    _base.retainWhere(test);
  }

  @override
  Iterable<T?> get reversed => _base.reversed;

  @override
  void setAll(int index, Iterable<T?> iterable) {
    _base.setAll(index, iterable);
  }

  @override
  void setRange(int start, int end, Iterable<T?> iterable, [int skipCount = 0]) {
    _base.setRange(start, end, iterable, skipCount);
  }

  @override
  void shuffle([Random? random]) {
    _base.shuffle(random);
  }

  @override
  void sort([int Function(T?, T?)? compare]) {
    _base.sort(compare);
  }

  @override
  List<T?> sublist(int start, [int? end]) => _base.sublist(start, end);

  @override
  List<int> get shape => [_base.length];
  
}