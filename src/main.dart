
import 'dart:io';
import 'DataFrame.dart';
import 'Corpus.dart';
import 'TfidfVectorizer.dart';
import 'PorterStemmer.dart';

void main() async {
  DataFrame large_queries = DataFrame.from_csv("data/dev_queries.tsv", seperator: "\t", columnTypes: [int, String]);
  DataFrame large_qres = DataFrame.from_csv("data/dev_query_results.csv", columnTypes:  [int, int]);
  DataFrame test_queries = DataFrame.from_csv("data/queries.csv", seperator: "\t", columnTypes: [int, String]);
  print("all dataframes loaded");
  Corpus documents = Corpus.folder("data\\full_docs"); // this is the directory where the documents are stored
  List<String> realDocNames = documents.docs.map((doc) => doc.filename.split("_").last.split('.').first).toList();
  documents.addModifier(new PorterStemmer());
  PorterStemmer stemmer = new PorterStemmer();

  await documents.loadAllFiles(); // load all files at once
  TfIdfVectorizer vectorizer = TfIdfVectorizer();
  vectorizer.fit_corpus(documents);
  String filecontents = "queryId\tdoc1\tdoc2\tdoc3\tdoc4\tdoc5\tdoc6\tdoc7\tdoc8\tdoc9\tdoc10\n";
  for (int i = 0; i < large_queries["Query number"].length ; i++) {
    int queryId = large_queries["Query number"][i];
    String queryString = stemmer.stem_document(large_queries["Query"][i]);
    List<(int, num)> query = vectorizer.query(queryString, 10);
    filecontents += "${queryId}";
    for (int j = 0; j < query.length; j++) {
      filecontents += "\t${realDocNames[query[j].$1]}";
    }
    filecontents += "\n";
    if (i % 10 == 0) print("written query $queryId");
  }
  File file = new File("tmp_scores.txt");
  // file.openWrite();
  file.openWrite();
  await file.writeAsString(filecontents);

  print("writen everything");

}