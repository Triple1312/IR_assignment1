
import 'dart:collection';
import 'dart:math';
import 'Matrix.dart';
import 'Vector.dart';
import 'Corpus.dart';
import 'Document.dart';
import 'Vectorizer.dart';
import 'stopwords.dart';

class TfIdfVectorizer extends Vectorizer {
  String _tfType;
  String _idfType;
  String _normType;
  int docCount = 0;
  bool _stopwords;

  // Matrix fitMatrix = Matrix.csr();
  // HashMap<String, double> idfValues = new HashMap<String, double>();
  Map<int, Set<int>> documentFrequency = {}; // defines in what documents a word appears
  List<Map<int, int>> doc_word_count = [];

  // tested tftype: n, t // idftype: t // normtype: n, c // stopwords: language
  TfIdfVectorizer({String tfType = 'n', String idfType = 't', String normType = 'c', String stopwords = ''}) : _normType = normType, _idfType = idfType, _tfType = tfType, _stopwords = stopwords == 'english';

  double _calculateTF(double rawTF, double maxTF, int docLength) {
    switch (_tfType) {
      case 'n': return rawTF;
      case 'l': return 1 + log(rawTF);
      case 'a': return 0.5 + (0.5 * rawTF / maxTF);
      case 'b': return rawTF > 0 ? 1.0 : 0.0;
      case 'L': return (1 + log(rawTF)) / (1 + log(docLength.toDouble()));
      default: return rawTF;
    }
  }

  double _calculateIDF(int docFreq) {
    switch (_idfType) {
      case 'n': return 1.0;
      case 't': return log((docCount+1) / (docFreq + 1)) +1;
      case 'tl': return log(docCount / docFreq.toDouble()) + 1;
      case 'p': return log((docCount - docFreq) / docFreq.toDouble()) + 1;
      default: return log(docCount / docFreq.toDouble()) + 1;
    }
  }

  Map<String, double> _normalizeVector(Map<String, double> vector) {
    if (_normType == 'n') return vector;  // No normalization
    if (_normType == 'l2' || _normType == 'c') {
      double norm = sqrt(vector.values.map((x) => x * x).reduce((a, b) => a + b));
      return vector.map((term, value) => MapEntry(term, value / norm));
    }
    return vector;
  }


  // different than other fit methods, resets vocab and doc count
  @override
  void fit(List<String> documents, {bool sorted = false} ) { // should not be called directly but possible if necessary // todo
    docCount = documents.length;
    for (int i = 0; i < documents.length; i++) {
      Map<int, int> doc_map = {};
      for (String word in documents[i].toLowerCase().split(" ")) {
        if (!vocab.containsKey(word)) {
          vocab[word] = vocab.length;
          this.documentFrequency[vocab[word]!] = {};
        }
        if (!doc_map.containsKey(vocab[word])) {
          doc_map[vocab[word]!] = 1;
        }
        else {
          doc_map[vocab[word]!] = doc_map[vocab[word]]! + 1;
        }
        this.documentFrequency[vocab[word]!]!.add(i);   // = docFreq[vocab[word]!]! + 1;

      }
      doc_word_count.add(doc_map);
      if (i % 10000 == 0) {
        print(i);
      }
    }
  }

  void fit_corpus(Corpus corpus) {
    docCount = corpus.length;
    for (int i = 0; i < corpus.length; i++) {
      Map<int, int> doc_map = {};
      for (String word in corpus[i].contents.toLowerCase().split(" ")) {
        if (!vocab.containsKey(word)) {
          vocab[word] = vocab.length;
          this.documentFrequency[vocab[word]!] = {};
        }
        if (!doc_map.containsKey(vocab[word])) {
          doc_map[vocab[word]!] = 1;
        }
        else {
          doc_map[vocab[word]!] = doc_map[vocab[word]]! + 1;
        }
        this.documentFrequency[vocab[word]!]!.add(i);   // = docFreq[vocab[word]!]! + 1;

      }
      doc_word_count.add(doc_map);
      if (i % 10000 == 0) {
        print(i);
      }
    }
  }

  Matrix<num> transform_corpus(Corpus corpus) {
    List<Map<int, int>> doc_word_count = [];

    for (int i = 0; i < corpus.length; i++) {
      Map<int, int> doc_map = {};
      for (String word in corpus[i].contents.toLowerCase().split(" ")) {
        if (!vocab.containsKey(word)) {
          vocab[word] = vocab.length;
          // this.documentFrequency[vocab[word]!] = {};
        }
        if (!doc_map.containsKey(vocab[word])) {
          doc_map[vocab[word]!] = 1;
        }
        else {
          doc_map[vocab[word]!] = doc_map[vocab[word]]! + 1;
        }
        // this.documentFrequency[vocab[word]!]!.add(i);   // = docFreq[vocab[word]!]! + 1;

      }
      doc_word_count.add(doc_map);
    }


    List<Map<int, double>> final_ret_mat = [];

    for (int j = 0; j < corpus.length; j++) {
      int maxTF = doc_word_count[j].values.reduce(max);

      Map<int, double> tfidfVector = {};
      for (var term in doc_word_count[j].keys) {
        double tf = _calculateTF(doc_word_count[j][term]!.toDouble(), doc_word_count[j].length.toDouble(), maxTF);
        double tmp_num = _calculateIDF(this.documentFrequency[term]!=null? this.documentFrequency[term]!.length : 0);
        tfidfVector[term] = tf * tmp_num;
      }
      double norm = sqrt(tfidfVector.values.map((x) => x * x).reduce((a, b) => a + b));
      for (var term in tfidfVector.keys) {
        tfidfVector[term] = tfidfVector[term]! / norm;
      }
      final_ret_mat.add(tfidfVector);

    }
    return Matrix<num>.map(final_ret_mat, corpus.length, vocab.length);
  }


  @override
  Matrix<num> transform(List<String> documents) {
    List<Map<int, int>> doc_word_count = [];

    for (int i = 0; i < documents.length; i++) {
      Map<int, int> doc_map = {};
      for (String word in documents[i].toLowerCase().split(" ")) {
        if (!vocab.containsKey(word)) {
          vocab[word] = vocab.length;
          // this.documentFrequency[vocab[word]!] = {};
        }
        if (!doc_map.containsKey(vocab[word])) {
          doc_map[vocab[word]!] = 1;
        }
        else {
          doc_map[vocab[word]!] = doc_map[vocab[word]]! + 1;
        }
        // this.documentFrequency[vocab[word]!]!.add(i);   // = docFreq[vocab[word]!]! + 1;

      }
      doc_word_count.add(doc_map);
    }


    List<Map<int, double>> final_ret_mat = [];

    for (int j = 0; j < documents.length; j++) {
      int maxTF = doc_word_count[j].values.reduce(max);

      Map<int, double> tfidfVector = {};
      for (var term in doc_word_count[j].keys) {
        double tf = _calculateTF(doc_word_count[j][term]!.toDouble(), doc_word_count[j].length.toDouble(), maxTF);
        double tmp_num = _calculateIDF(this.documentFrequency[term]!=null? this.documentFrequency[term]!.length : 0);
        tfidfVector[term] = tf * tmp_num;
      }
      double norm = sqrt(tfidfVector.values.map((x) => x * x).reduce((a, b) => a + b));
      for (var term in tfidfVector.keys) {
        tfidfVector[term] = tfidfVector[term]! / norm;
      }
      final_ret_mat.add(tfidfVector);

    }
    return Matrix<num>.map(final_ret_mat, documents.length, vocab.length);

  }


  Matrix<num> ffff(List<String> corpus) {
    docCount = corpus.length;
    for (int i = 0; i < corpus.length; i++) {
      Map<int, int> doc_map = {};
      for (String word in corpus[i].toLowerCase().split(" ")) {
        if (!vocab.containsKey(word)) {
          vocab[word] = vocab.length;
          this.documentFrequency[vocab[word]!] = {};
        }
        if (!doc_map.containsKey(vocab[word])) {
          doc_map[vocab[word]!] = 1;
        }
        else {
          doc_map[vocab[word]!] = doc_map[vocab[word]]! + 1;
        }
        this.documentFrequency[vocab[word]!]!.add(i);   // = docFreq[vocab[word]!]! + 1;

      }
      doc_word_count.add(doc_map);
      if (i % 10000 == 0) {
        print(i);
      }
    }

    List<Map<int, double>> final_ret_mat = [];

    for (int j = 0; j < corpus.length; j++) {
      int maxTF = doc_word_count[j].values.reduce(max);

      Map<int, double> tfidfVector = {};
      for (var term in doc_word_count[j].keys) {
        double tf = _calculateTF(doc_word_count[j][term]!.toDouble(), doc_word_count[j].length.toDouble(), maxTF);
        double tmp_num = _calculateIDF(this.documentFrequency[term]!.length);
        tfidfVector[term] = tf * tmp_num;
      }
      double norm = sqrt(tfidfVector.values.map((x) => x * x).reduce((a, b) => a + b));
      for (var term in tfidfVector.keys) {
        tfidfVector[term] = tfidfVector[term]! / norm;
      }
      final_ret_mat.add(tfidfVector);

    }
    return Matrix<num>.map(final_ret_mat, corpus.length, vocab.length);
  }


  // Matrix<num> fit_new_transform(Corpus corpus) {
  //   List<int> docLengths = List<int>.filled(corpus.length, 0);
  //
  //   Map<String, Map<int, int>> word_doc_count = {};
  //   for (int i = 0; i < corpus.length; i++) {
  //     List<String> terms = corpus[i].contents.toLowerCase().split(" "); // todo test if this takes more space
  //     docLengths[i] = terms.length;
  //     for (String word in terms) {
  //       if (!word_doc_count.containsKey(word)) {
  //         word_doc_count[word] = {};
  //       }
  //       if (!word_doc_count[word]!.containsKey(i)) {
  //         word_doc_count[word]![i] = 1;
  //       }
  //       else {
  //         word_doc_count[word]![i] = word_doc_count[word]![i]! + 1;
  //       }
  //     }
  //     if(i % 1000 == 0) {
  //       print(i);
  //     }
  //   }
  //
  //   for (int j = 0; j < corpus.length; j++) {
  //     // Calculate term frequencies in one pass.
  //     List<String> terms = corpus[j].contents.toLowerCase().split(" "); // todo test if this takes more space
  //     Map<String, int> termFreqs = {};
  //     for (var term in terms) {
  //       termFreqs[term] = (termFreqs[term] ?? 0) + 1;
  //     }
  //   }
  //
  //
  //   // vocab = word_doc_count.keys.toList();
  //   // Map<String, int> vocabIndices = {};
  //   vocab.clear();
  //   int index = 0;
  //   for (var key in word_doc_count.keys) {
  //     vocab[key] = index;
  //     index++;
  //   }
  //   print("vocab made");
  //
  //   // data = Matrix<num>.map(word_doc_count.values.toList(), corpus.length, vocab.length);
  //
  //
  //   for ( var voc in vocab.keys) {
  //     idfValues[voc] = _calculateIDF(countvectorizer.word_doc_count[voc]!.length);
  //   }
  //
  //
  // }

  @override
  Matrix<num> fit_transform(List<String> documents, {bool sorted = false}) {
    // docCount = documents.length;
    // var termDocs = HashMap<String, Set<int>>();
    // // HashMap<String, double> idfScores = HashMap<String, double>();
    //
    //
    // var uniqueTerms = Set<String>(); // needed ??
    //
    // for (var i = 0; i < documents.length; i++) {
    //   List<String> terms = documents[i].split(" ");
    //   terms.removeWhere((term) => term == '' || term == ' ' || term == "." || term == "?" || term == "!" || term == "," || term == ":" || term == ";");
    //   if (_stopwords) {
    //     terms = terms.where((term) => !ENGLISH_STOP_WORDS.contains(term)).toList();
    //   }
    //   uniqueTerms = Set<String>.from(terms);
    //   uniqueTerms.forEach((term) {
    //     (termDocs[term] ??= {}).add(i);
    //   });
    // }
    // termDocs.forEach((term, docs) {
    //   idfValues[term] = _calculateIDF(docs.length);
    // });
    //
    // for (var i = 0; i < documents.length; i++) {
    //   List<String> terms = documents[i].split(" ");
    //   if (_stopwords) {
    //     terms = terms.where((term) => !ENGLISH_STOP_WORDS.contains(term)).toList();
    //   }
    //   uniqueTerms.addAll(terms);
    // }
    //
    // // vocab = uniqueTerms.toList(); // this is the order
    //
    // Matrix tmp_m = Matrix.csr();
    // for (var doc in documents) {
    //   // Split the document into terms and optionally filter out stop words.
    //   List<String> rawTerms = doc.split(" ");
    //   List<String> terms = _stopwords ? rawTerms.where((term) => !ENGLISH_STOP_WORDS.contains(term)).toList() : rawTerms;
    //
    //   // Calculate term frequencies in one pass.
    //   Map<String, int> termFreqs = {};
    //   for (var term in terms) {
    //     termFreqs[term] = (termFreqs[term] ?? 0) + 1;
    //   }
    //
    //   // Determine the maximum term frequency for normalization.
    //   int maxTF = termFreqs.values.reduce(max);
    //
    //   // Create the TF-IDF vector for the document.
    //   Map<String, double> tfidfVector = termFreqs.map((term, count) {
    //     double tf = _calculateTF(count.toDouble(), terms.length.toDouble(), maxTF);
    //     return MapEntry(term, tf * (idfValues[term] ?? 0.0));
    //   });
    //
    //   // Normalize the TF-IDF vector and convert it into a list using the vocabulary.
    //   double norm = sqrt(tfidfVector.values.map((x) => x * x).reduce((a, b) => a + b));
    //   SparceVector<double> tmp_vec = SparceVector.empty();
    //   for (var word in vocab.keys) {
    //     tmp_vec.add((tfidfVector[word] ?? 0.0) / norm);
    //   }
    //   // vocab.forEach((word) => tmp_vec.add(tfidfVector[word] ?? 0.0) / norm));
    //   tmp_m.addRow(tmp_vec);
    // }
    // fitMatrix = tmp_m;
    // return tmp_m;
    return ffff(documents);
  }


  // Matrix<num> firt_tr(Corpus corpus, {bool sorted = false}) {
  //
  // }

  Matrix<num> fit_transform_corpus(Corpus corpus, {bool sorted = false}) {
    return ffff(corpus.docs.map((e) => e.contents).toList());
  }


  List<(int, num)> query(String query, [int k = 10]) {
    List<num> queryVector = transform([query])[0];
    Vector<num> vec = SparceVector(queryVector);

    List<(int, num)> scores = [];
    num min = 0;

    List<Map<int, double>> final_ret_mat = [];


    for (int j = 0; j < doc_word_count.length; j++) {
      Map<int, int> doc_map = doc_word_count[j];
      int maxTF = doc_map.values.reduce(max);

      Map<int, double> tfidfVector = {};
      for (var term in doc_map.keys) {
        double tf = _calculateTF(doc_map[term]!.toDouble(),
            doc_map.length.toDouble(), maxTF);
        double tmp_num = _calculateIDF(this.documentFrequency[term]!.length);
        tfidfVector[term] = tf * tmp_num;
      }

      // double norm = sqrt(
      //     tfidfVector.values.map((x) => x * x).reduce((a, b) => a + b));
      // for (var term in tfidfVector.keys) {
      //   tfidfVector[term] = tfidfVector[term]! / norm;
      // }

      Vector<num> vecvec = SparceVector.map(tfidfVector, vocab.length);
      vecvec = vecvec.normalize();
      num sim = vecvec.dot(vec) / (vecvec.norm(2) * vec.norm(2));
      if (sim < min) continue;
      if (scores.length < k) {
        scores.add((j, sim));
        continue;
      };
      scores.add((j, sim));
      scores.sort((a, b) => b.$2.compareTo(a.$2));
      scores.removeLast();
      min = scores.last.$2;
    }
    scores.sort((a, b) => b.$2.compareTo(a.$2));
    return scores;
  }


  // List<(int, num)> query(String query, [int k = 10]) {
  //   List<num> queryVector = transform([query])[0];
  //   List<(int, num)> scores = [];
  //   num min = 0;
  //   Vector vec = Vector(queryVector);
  //   for (int i = 0; i < fitMatrix.length; i++) {
  //     Vector tmp_vec = Vector(fitMatrix[i]);
  //     /// cosine similarity
  //     num sim = tmp_vec.dot(vec) / (tmp_vec.norm(2) * vec.norm(2));
  //     if (sim < min) continue;
  //     if (scores.length < k) {scores.add((i, sim)); continue;};
  //
  //     for (int t = 0;  t <  scores.length; t++) {
  //       if (scores[t].$2 <= min) {
  //         scores[t] = (i, sim);
  //         min = 999999999;
  //         for (var y in scores) {
  //           if (y.$2 < min) {
  //             min = y.$2;
  //           }
  //         }
  //       }
  //     }
  //
  //   }
  //   scores.sort( (a, b) => a.$2.compareTo(b.$2));
  //   return scores;
  // }

  translate(int index) {
    // return vocab[index];
  }

  set tfType(String value) {
    if (value != 'n' && value != 'l' && value != 'a' && value != 'b' && value != 'L') {
      throw Exception('Invalid value for tfType: $value');
    }
    else {
      _tfType = value;
    }
  }

  set idfType(String value) {
    if (value != 'n' && value != 't' && value != 'p') {
      throw Exception('Invalid value for idfType: $value');
    }
    else {
      _idfType = value;
    }
  }

  set normType(String value) {
    if (value != 'n' && value != 'l1' && value != 'l2') {
      throw Exception('Invalid value for normType: $value');
    }
    else {
      _normType = value;
    }
  }

  String get tfType => _tfType;
  String get idfType => _idfType;
  String get normType => _normType;



}