
import 'dart:math';
import 'Matrix.dart';
import 'Vector.dart';
import 'Corpus.dart';
import 'Vectorizer.dart';

class TfIdfVectorizer extends Vectorizer {
  String _tfType;
  String _idfType;
  String _normType;
  int docCount = 0;
  bool _stopwords;


  Map<int, Set<int>> documentFrequency = {}; // defines in what documents a word appears
  Map<int, double> idfValues = {};
  List<Map<int, int>> doc_word_count = [];
  Matrix data = Matrix.empty();

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
    fit_transform(documents);
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

  List<(int, num)> new_query(String query, [int k = 10]) {

    SparceVector<num> doc_vec = SparceVector.empty();
    doc_vec.extend(vocab.length);
    for (String word in query.split(" ")) {
      int? v = vocab[word];
      if (v != null) {
        doc_vec[v] = doc_vec[v] + 1;
      }
    }
    SparceVector<num> query_tfidf = SparceVector.empty();
    for ((int, num) p in doc_vec.sparceIterator) {
      query_tfidf[p.$1] = p.$2 * idfValues[p.$1]!;
    }

    /// transform done

    List<(int, num)> scores = [];
    num min = 0;

    for (int j = 0; j < data.data.rowLenght; j++) {
      Vector vecvec = data.row(j);
      num sim = vecvec.dot(query_tfidf) / (vecvec.norm2() * query_tfidf.norm2());
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


  Matrix<num> fit_transform(List<String> corpus, {bool sorted = false}) {
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

    for (int voc in vocab.values) {
      idfValues[voc] = _calculateIDF(this.documentFrequency[voc]!.length);
    }
    this.documentFrequency.clear();

    data = Matrix<num>.map(final_ret_mat, corpus.length, vocab.length);
    return data;
  }




  Matrix<num> fit_transform_corpus(Corpus corpus, {bool sorted = false}) {
    return fit_transform(corpus.docs.map((e) => e.contents).toList());
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