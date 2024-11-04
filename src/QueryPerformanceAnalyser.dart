

class QueryPerformanceAnalyser {

  static double mean_average_precision<T>(List<List<T>> relevant, List<List<T>> predictions, int k) {
    double precision = 0;
    for (int i = 0; i < relevant.length; i++) {
      precision += QueryPerformanceAnalyser.precision(relevant[i], predictions[i], k);
    }
    return precision / relevant.length;
  }

  static double mean_average_recall<T>(List<List<T>> relevant, List<List<T>> predictions, int k) {
    double recall = 0;
    for (int i = 0; i < relevant.length; i++) {
      recall += QueryPerformanceAnalyser.precision(relevant[i], predictions[i], k);
    }
    return recall / relevant.length;
  }

  static double precision<T>(List<T> relevant, List<T> predictions, int k) {
    int tp = 0;
    for (int i = 0; i < k; i++) {
      if (relevant.contains(predictions[i])) tp++;
    }
    return tp / k;
  }

  static double recall<T>(List<T> relevant, List<T> predictions, int k) {
    int tp = 0;
    for (int i = 0; i < k; i++) {
      if (relevant.contains(predictions[i])) tp++;
    }
    return tp / relevant.length;
  }

}