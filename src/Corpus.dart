
import 'dart:convert';
import 'dart:io';

import 'TextModifier.dart';
import 'Document.dart';



// todo make it implement list

class Corpus {

  List<Document> docs = new List<Document>.empty(growable: true);

  List<TextModifier> modifiers = []; // keeps all modifiers and will apply them when a new document is added

  String folderpath = "";

  Corpus(this.docs);

  Corpus.folder(String folderpath) { // todo add a filter for filetypes
    this.folderpath = folderpath;
    List<String> files = Directory(folderpath).listSync().map((e) => e.path).toList();

    for (int i = 0; i < files.length; i++) {
      String file = files[i];
      docs.add(Document.path(file));
    }

  }

  int get length => docs.length;

  List<String> get filenames => docs.map((e) => e.filename).toList();

  Document operator[](int index) => docs[index];

  Document? getDoc(String name) {
    for (Document doc in docs) {
      if (doc.filename == name) {
        return doc;
      }
    }
    return null;
  }

  removeDoc(String filename) {
    for (int i = 0; i < docs.length; i++) {
      if (docs[i].filename == filename) {
        docs.removeAt(i);
        return;
      }
    }
  }

  void addModifier(TextModifier modifier) {
    modifiers.add(modifier);
    docs.forEach((doc) {
      doc.addModifier(modifier);
    });
  }


  Future<void> loadAllFiles([int batchcount = 5000]) async {
    for (int i = 0; i < docs.length; i += batchcount) {
      int batchend = i + batchcount;
      if (batchend > docs.length) batchend = docs.length;
      List<Document> batch = docs.sublist(i,batchend);
      await Future.wait(batch.map((doc) => doc.loadAsync()));
      print(batchend);
    }

    // for (int i = 0; i < docs.length; i++) {
    //   Document doc = docs[i];
    //   doc.load();
    //   if (i % 1000 == 0) print(i);
    // }
  }

  void loadAllFilesAsync() async {
    for (int i = 0; i < docs.length; i++) {
      Document doc = docs[i];
      await doc.loadAsync();
      if (i % 1000 == 0) print(i);
    }
  }

}