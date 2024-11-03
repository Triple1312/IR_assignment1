
import 'dart:io';
import 'dart:convert';

import 'Corpus.dart';
import 'Document.dart';
import 'InvertedIndexer.dart';

class SPIMI {

  void fit_corpus(Corpus corpus, int maxByteSize) {
    int currentSize = 0;
    List<Document> currentDocs = [];
    int index = 0;
    for (Document doc in corpus.docs) {
      int docSize = doc.size();
      if (docSize > maxByteSize) {
        throw Exception("Document ${doc.filename} is too large for the maxByteSize");
      }
      if (currentSize + doc.size() > maxByteSize ) {
        InvertedIndexer indexer = new InvertedIndexer()..fit(currentDocs);
        writeTemp(indexer, "tmpSPIMI$index.json");
        index++;
      }
      else {
        currentDocs.add(doc);
        currentSize += doc.size();
      }
    }
    if (currentDocs.isNotEmpty) {
      InvertedIndexer indexer = new InvertedIndexer()..fit(currentDocs);
      writeTemp(indexer, "tmpSPIMI$index.json");
    }
    mergeAll();
  }

  void mergeAll([String filename = "SPIMImerged.json"]) {
    Directory dir = Directory.current;
    List<FileSystemEntity> files = dir.listSync();
    files.removeWhere((f) => !f.path.startsWith("tmpSPIMI"));
    Map<String, List<int>> mergedIndex = jsonDecode((files[0] as File).readAsStringSync());
    for (int i = 1; i < files.length; i++) {
      Map<String, List<int>> index = jsonDecode((files[i] as File).readAsStringSync());
      for (String term in index.keys) {
        if (mergedIndex.containsKey(term)) {
          mergedIndex[term] = (mergedIndex[term]! + index[term]!);
        }
        else {
          mergedIndex[term] = index[term]!;
        }
      }
    }
    File file = File(filename);
    file.writeAsStringSync(jsonEncode(mergedIndex));
  }


  void writeTemp(InvertedIndexer indexer, String filename) {
    File file = new File(filename);
    file.writeAsStringSync(jsonEncode(indexer.invertedIndex));
  }

}