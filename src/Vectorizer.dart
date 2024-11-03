
import 'dart:collection';

import 'Matrix.dart';


abstract class Vectorizer {

  Map<String, int> vocab = {};



  // @protected
  // void dataFrameConstructor(DataFrame data); // todo maybe could implement here // no idea what I should do with it

  bool contains(String key) => vocab[key] != null;

  String? translate(int index) => vocab.keys.elementAt(index);

  Matrix<num> fit_transform(List<String> documents, {bool sorted = false});

  void fit(List<String> documents, {bool sorted = false});

  Matrix<num> transform(List<String> documents);

  // void fit_df(DataFrame data, String columnName) => fit(data.getColumn(columnName)!.asString());

  // void fit_column(DFColumn data) => fit(data.asString());

  // List<List<num>> transform_df(DataFrame data, String columnName) => transform(data.getColumn(columnName)!.asString());

  // List<List<num>> transform_column(DFColumn data) => transform(data.asString());

  bool add_to_vocab(String word) {
    if (vocab[word] == null) {
      vocab[word] = vocab.length;
      return true;
    }
    return false;
  }

}