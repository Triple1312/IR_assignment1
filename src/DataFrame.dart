

import 'dart:io';

import 'DFColumn.dart';


class DataFrame extends Iterable<DFColumn> {

  List<DFColumn> _base = [];

  DataFrame.from_csv(String filepath, {bool hascolumnNames = true, seperator = ',', List<String> columnNames = const [], List<Type> columnTypes = const []}) {
    List<String> lines = File(filepath).readAsLinesSync();
    List<List<String>> words = lines.map((e) => e.split(seperator)).toList();
    List<String> columnNamesnew = [];
    columnNamesnew.addAll(columnNames);
    if (columnNamesnew.isEmpty) {
      columnNamesnew = [];
      columnNamesnew = words[0];
      words = words.sublist(1);
    }
    List<Type> columnTypesnew = [];
    columnTypesnew.addAll(columnTypes);
    if (columnTypes.length < words[0].length) {
      for (int i = columnTypesnew.length; i < words[0].length; i++) {
        columnTypesnew.add(dynamic);
      }
    }
    _base = DataFrame.build(words, columnNamesnew, columnTypesnew)._base; // todo this is so bad , pl fix
  }

  DataFrame(this._base);

  DataFrame.build(List<List<String>> words, List<String> columnNames, List<Type> columnTypes) {
    for (int i = 0; i < columnNames.length; i++) {
      Type curretnType = String; // default type
      if (columnTypes.length > i) { // if columntype is defined for this element
        curretnType = columnTypes[i]; // set given columntype
      }
      if (curretnType == String) {
        _base.add(DFColumn<String>(columnNames[i], words.fold([], (prev, e) => prev..add(e[i] == ''? null: e[i]))));
      }
      else if (curretnType == int) {
        _base.add(DFColumn<int>(columnNames[i], words.fold([], (prev, e) => prev..add(e[i] == ''? null: int.parse(e[i])))));
      }
      else if (curretnType == double) {
        _base.add(DFColumn<double>(columnNames[i], words.fold([], (prev, e) => prev..add(e[i] == ''? null: double.parse(e[i])))));
      }
      else if (curretnType == bool) {
        _base.add(DFColumn<bool>(columnNames[i], words.fold([], (prev, e) => prev..add(e[i] == ''? null: e[i] == 'true'))));
      }
      else if(curretnType == dynamic) {
        DFColumn column = new DFColumn(columnNames[i], []);
        for (var wordl in words) {
          if (wordl[i] == '') {
            column.add(null);
            continue;
          }
          try {
            column.add(int.parse(wordl[i]));
          }
          catch (e) {
            try {
              column.add(double.parse(wordl[i]));
            }
            catch (e) {
              try {
                column.add(wordl[i] == 'true' ? true : wordl[i] == 'false'
                    ? false
                    : throw Exception('Invalid boolean value'));
              }
              catch (e) {
                column.add(wordl[i]);
              }
            }
          }
        }
        _base.add(column);
      }
    }
  }



  get shape => [_base.isEmpty? 0: _base[0].length, _base.length];

  get size => _base.isEmpty? 0: _base[0].length * _base.length;

  get columns => _base.map((e) => e.columnName).toList();

  DFColumn operator [](String columnName) {
    for (var column in _base) {
      if (column.columnName == columnName) {
        return column;
      }
    }
    throw Exception('Column not found');
  }

  List row(int index) {
    return _base.map((e) => e[index]).toList();
  }

  List<List> rows([int start = 0, int end = -1]) {
    if (end < 0) {
      end = _base[0].length - end + 1;
    }
    return _base.map((e) => e.sublist(start, end)).toList();
  }

  List<List<int?>> asInt() => _base.map((e) => e.asInt()).toList();

  List<List<double?>> asDouble() => _base.map((e) => e.asDouble()).toList();

  List<List<bool?>> asBool() => _base.map((e) => e.asBool()).toList();

  List<List<String>> asString() => _base.map((e) => e.asString()).toList();

  List<List> asList() => _base.map((e) => e).toList();

  @override
  Iterator<DFColumn> get iterator => new DFColumnIterator(this);

  Map<String, bool> all() {
    Map<String, bool> res = {};
    for (var column in _base) {
      res[column.columnName] = column.all();
    }
    return res;
  }

  Map<String, bool> anyNotNull() {
    Map<String, bool> res = {};
    for (var column in _base) {
      res[column.columnName] = column.anyNotNull();
    }
    return res;
  }

  Map<String, int> count() {
    Map<String, int> res = {};
    for (var column in _base) {
      res[column.columnName] = column.count();
    }
    return res;
  }

  void addColumn(DFColumn column) {
    _base.add(column);
  }

  void removeColumn(String columnName) {
    _base.removeWhere((element) => element.columnName == columnName);
  }

}

class DFColumnIterator implements Iterator<DFColumn>{
  DataFrame df;
  int _index = 0;

  DFColumnIterator(this.df);

  @override
  DFColumn get current => df._base[_index];

  @override
  bool moveNext() {
    if ( _index < df._base.length - 1) {
      _index++;
      return true;
    }
    return false;
  }
}