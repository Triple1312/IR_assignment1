


import 'Document.dart';

class PositionalIndexer {

  Map<String, Map<int, int>> indeces = {};

  List<Document> documents = [];

  void fit(List<Document> documents) {
    this.documents = documents;
    for (int i = 0; i < documents.length; i++) {
      List<String> terms = documents[i].contents.split(" ");
      for (int j = 0; j < terms.length; j++) {
        if (!indeces.containsKey(terms[j])) {
          indeces[terms[j]] = {};
        }
        if (!indeces[terms[j]]!.containsKey(i)) {
          indeces[terms[j]]![i] = j;
        }
      }
    }
  }

  Map<Document,Map<String, List<int>>> query(String query) =>
      query_list(query.split(" "));

  Map<Document,Map<String, List<int>>> query_list(List<String> query) {

    Map<Document,Map<String, List<int>>> result = {};

    for (String term in query) {
      if (indeces.containsKey(term)) {
        for (int i in indeces[term]!.keys) {
          if (!result.containsKey(documents[i])) {
            result[documents[i]] = {};
          }
          if (!result[documents[i]]!.containsKey(term)) {
            result[documents[i]]![term] = [];
          }
          result[documents[i]]![term]!.add(indeces[term]![i]!);
        }
      }
    }

    return result;
  }

  // String word;
  //
  // int frequency;
  //
  // List<List<int>> positions;
  //
  // PositionalIndexer(this.word): frequency = 0, positions = [];
  //
  // void addDocumentPositions(List<int> docPositions) {
  //   frequency+=docPositions.length;
  //   positions.add(docPositions);
  // }
  //
  // void addDocumentPosition(int documentIndex, int position) {
  //   frequency++;
  //   positions[documentIndex].add(position);
  // }
  //
  // void removeDocument(int documentIndex) {
  //   frequency-=positions[documentIndex].length;
  //   positions.removeAt(documentIndex);
  // }




}