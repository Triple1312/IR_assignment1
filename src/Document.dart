

import 'dart:io';

import 'TextModifier.dart';

// todo Add a stream method ?

class Document {

  File? file;

  String _contents = "";

  String _filename = "";

  List<TextModifier> modifiers = []; // only used if text is not loaded

  String get contents {
    if (_contents != "") return _contents;
    String tmp = file!.readAsStringSync();
    for (TextModifier modifier in modifiers) {
      tmp = modifier.transform(tmp);
    }
    return tmp;
  }

  String get filename {
    if (_filename != "") return _filename;
    _filename = file!.path.split("/").last;
    return _filename;
  }

  Document(this._filename, this._contents) : file = null;

  Document.file(File file) : file = file;

  Document.path(String path) : file = new File(path);

  Document.load_file(String filepath) : file = new File(filepath) {
    load();
  }

  int size() {
    if (contents == "") {
      return file!.lengthSync();
    }
    else {
      return _contents.length;
    }
  }

  set contents(String _contents) {
    file!.writeAsStringSync(_contents);
  }

  set filename(String _filename) {
    file!.rename(_filename);
  }

  void load() {
    _contents = file!.readAsStringSync();
    for (TextModifier modifier in modifiers) {
      _contents = modifier.transform(_contents);
    }
    modifiers.clear();
  }

  Future<void> loadAsync() async {
    _contents = await file!.readAsString();
    for (TextModifier modifier in modifiers) {
      _contents = modifier.transform(_contents);
    }
    modifiers.clear();
  }

  void save() {
    file!.writeAsStringSync(_contents);
  }

  void rename(String newname) {
    file!.rename(newname);
  }

  void saveToFile(String filepath) {
    File curr_file = new File(filepath);
    curr_file.writeAsStringSync(_contents);
  }

  void addModifier(TextModifier modifier) {
    if (_contents != "") {
      _contents = modifier.transform(_contents);
    }
    else {
      modifiers.add(modifier);
    }
  }

}




// class CSVDocument extends Document {
//
//   String contents;
//
//   CSVDocument(String filepath, [String seperator = ',']) {
//     File file = new File(filepath);
//     file.open();
//     List<String> lines = file.readAsLinesSync();
//   }
//   @override
//   String contents;
//
//   @override
//   String filename;
//
// }