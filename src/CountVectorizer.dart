

import 'dart:collection';

import 'Matrix.dart';
import 'Vector.dart';
import 'Corpus.dart';
import 'Vectorizer.dart';

class Countvectorizer extends Vectorizer {

  Matrix<num> data = Matrix<num>.empty();

  Map<String, Map<int, int>> word_doc_count = {};

  @override
  Matrix<num> transform(List<String> documents) {
    // // Precompute the indices in a map for quick lookup.
    // Map<String, int> vocabIndices = {};
    // for (int i = 0; i < vocab.length; i++) {
    //   vocabIndices[vocab[i]] = i;
    // }
    //
    // return documents.map((document) {
    //   List<String> words = document.toLowerCase().split(" ");
    //   List<double> counts = List<double>.filled(vocab.length, 0.0);
    //
    //   words.forEach((word) {
    //     if (vocabIndices.containsKey(word)) {
    //       counts[vocabIndices[word]!]++;
    //     }
    //   });
    //
    //   return counts;
    // }).toList();
    return new Matrix<num>.empty();
  }

  @override
  void fit(List<String> documents, {bool sorted = false}) {
    for (String document in documents) {
      for (String word in document.toLowerCase().split(" ")) {
        if (vocab[word] == null) {
          vocab[word] = vocab.length;
        }
      }
    }
  }

  @override
  Matrix<num> fit_transform(List<String> documents, {bool sorted = false}) {
    fit(documents, sorted: sorted);
    return transform(documents);
  }


  Matrix<num> fit_new_transform(Corpus corpus) {
    word_doc_count = {};
    for (int i = 0; i < corpus.length; i++) {
      for (String word in corpus[i].contents.toLowerCase().split(" ")) {
        if (!word_doc_count.containsKey(word)) {
          word_doc_count[word] = {};
        }
        if (!word_doc_count[word]!.containsKey(i)) {
          word_doc_count[word]![i] = 1;
        }
        else {
          word_doc_count[word]![i] = word_doc_count[word]![i]! + 1;
        }
      }
      if(i % 1000 == 0) {
        print(i);
      }
    }


    // vocab = word_doc_count.keys.toList();
    // Map<String, int> vocabIndices = {};
    vocab.clear();
    int index = 0;
    for (var key in word_doc_count.keys) {
      vocab[key] = index;
      index++;
    }
    print("vocab made");

    data = Matrix<num>.map(word_doc_count.values.toList(), corpus.length, vocab.length);
    return data;
  }




  Matrix<num> fit_transform_corpus(Corpus corpus, {bool sorted = false}) {
    List<String> vocab = [];
    List<Vector<num>> docmatrix = [];

    for (int i = 0; i < corpus.length; i++) {
      Map<String, int> tmp_words = new Map(); // because words that are already found in a document have a higher chance of comming back
      for (String word in corpus[i].contents.toLowerCase().split(" ")) {
        tmp_words[word] = tmp_words.containsKey(word) ? tmp_words[word]! + 1 : 1; // dont know if fast enough
      }
      docmatrix.add(Vector.empty());
      tmp_words.forEach((key, value) {
        int index = vocab.indexOf(key);
        if (index == -1) {
          vocab.add(key);
          index = vocab.length - 1;
          docmatrix[i].addExtend(index, value);
        }
        else {
          docmatrix[i].addExtend(index, value);
        }
      });
      if(i % 1000 == 0) {
        print(i);
      }
    }
    this.data = Matrix<num>.csr(values: docmatrix);
    return data;
  }

  // List<List<int>> fit_transform_int(List<String> documents, {bool sorted = false}) {
  //   fit(documents, sorted: sorted);
  //   return transform_int(documents);
  // }
  //
  // List<List<int>> transform_int(List<String> documents) {
  //   // Precompute the indices in a map for quick lookup.
  //   Map<String, int> vocabIndices = {};
  //   for (int i = 0; i < vocab.length; i++) {
  //     vocabIndices[vocab[i]] = i;
  //   }
  //
  //   return documents.map((document) {
  //     List<String> words = document.toLowerCase().split(" ");
  //     List<int> counts = List<int>.filled(vocab.length, 0);
  //
  //     words.forEach((word) {
  //       if (vocabIndices.containsKey(word)) {
  //         counts[vocabIndices[word]!]++;
  //       }
  //     });
  //
  //     return counts;
  //   }).toList();
  // }

}