// based on Porter, M. "An algorithm for suffix stripping."
// includes the Martin extentions

import 'TextModifier.dart';

import 'stopwords.dart';

class PorterStemmer implements TextModifier{

  static String vowels = "aeiou";

  static String consonants = "bcdfghjklmnpqrstvwxyz";

  bool _stopwords = false;

  PorterStemmer({bool stopwords = false}) {
    _stopwords = stopwords;
  }

  bool _vowel(String v) => vowels.contains(v); // pl just only send a char

  bool _consonant(String c) => consonants.contains(c); // pl just only send a char

  String _cvTranslate(String string) =>
      string.split("").fold("", (a, String b) =>
          _vowel(b)? a+= 'v' : _consonant(b)? a+= 'c' : a+= ' '
      );
  
  int _vcCount(String word) => RegExp(r'[aeiou][^aeiou\s\W\d]').allMatches(word).length;

  bool _cvcEnd(String word) => _cvTranslate(word).endsWith("cvc") && !word.endsWith("w") && !word.endsWith("x") && !word.endsWith("y");

  int _vowelCount(String word) => word.split("").where((c) => _vowel(c)).length;

  String _strip1a(String word) =>
      word.endsWith("sses") ? word.substring(0, word.length - 2) :
      word.endsWith("ies") ? word.substring(0, word.length - 2) :
      word.endsWith("ss") ? word :
      word.endsWith("s")? word.substring(0, word.length - 1) :
      word;

  String _strip1b(String word) {
    String newWord = word;

    if (newWord.endsWith("eed")) return _vcCount(newWord.substring(0, newWord.length - 3)) > 0 ? newWord.substring(0, newWord.length - 1) : newWord;

    // these could be shorter I think
    if (newWord.endsWith("ed")) { // same as containsvowel of stem
      newWord = newWord.substring(0, newWord.length - 2);
      if (_vowelCount(newWord) == 0) {
        return word;
      }
    }
    else if(newWord.endsWith("ing")) {
      newWord = newWord.substring(0, newWord.length - 3);
      if (_vowelCount(newWord) == 0) {
        return word;
      }
    }
    else {
      return word;
    }

    if (newWord.endsWith("at") || newWord.endsWith("bl") || newWord.endsWith("iz")) {
      return newWord + "e";
    }
    if (newWord[newWord.length - 1] == newWord[newWord.length - 1] &&
        newWord[newWord.length - 1] != 'l' &&
        newWord[newWord.length - 1] != 's' &&
        newWord[newWord.length - 1] != 'z') {
      return newWord.substring(0, newWord.length - 1);
    }
    if (_vcCount(newWord) == 1 && _cvcEnd(newWord)) {
      return newWord + "e";
    }

    return newWord;
  }

  String _strip1c(String word) {
    if (word.endsWith("y") && _vowelCount(word.substring(0, word.length - 1)) > 0) {
      return word.substring(0, word.length - 1) + "i";
    }
    else {
      return word;
    }
  }

  String _strip2(String word) {
    if (_vcCount(word) == 0) {
      return word;
    }

    if (word.endsWith("ational")) return _stemvcCount(word, "ational") > 0 ? word.substring(0, word.length - 5) + "e" : word;
    if (word.endsWith("tional")) return _stemvcCount(word, "tional") > 0 ? word.substring(0, word.length - 2) : word;
    if (word.endsWith("enci")) return _stemvcCount(word, "enci") > 0 ? word.substring(0, word.length - 1)  + "e": word;
    if (word.endsWith("anci")) return _stemvcCount(word, "anci") > 0 ? word.substring(0, word.length - 1) + "e" : word;
    if (word.endsWith("izer")) return _stemvcCount(word, "izer") > 0 ? word.substring(0, word.length - 1) : word;
    if (word.endsWith("abli")) return _stemvcCount(word, "abli") > 0 ? word.substring(0, word.length - 1) + "e" : word;
    if (word.endsWith("alli")) return _stemvcCount(word, "alli") > 0 ? word.substring(0, word.length - 2) : word;
    if (word.endsWith("entli")) return _stemvcCount(word, "entli") > 0 ? word.substring(0, word.length - 2) : word;
    if (word.endsWith("eli")) return _stemvcCount(word, "eli") > 0 ? word.substring(0, word.length - 2) : word;
    if (word.endsWith("ousli")) return _stemvcCount(word, "ousli") > 0 ? word.substring(0, word.length - 2) : word;
    if (word.endsWith("ization")) return _stemvcCount(word, "ization") > 0 ? word.substring(0, word.length - 5) + 'e' : word;
    if (word.endsWith("ation")) return _stemvcCount(word, "ation") > 0 ? word.substring(0, word.length - 3) + 'e' : word;
    if (word.endsWith("ator")) return _stemvcCount(word, "ator") > 0 ? word.substring(0, word.length - 2) + 'e' : word;
    if (word.endsWith("alism")) return _stemvcCount(word, "alism") > 0 ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("iveness")) return _stemvcCount(word, "iveness") > 0 ? word.substring(0, word.length - 4) : word;
    if (word.endsWith("fulness")) return _stemvcCount(word, "fulness") > 0 ? word.substring(0, word.length - 4) : word;
    if (word.endsWith("ousness")) return _stemvcCount(word, "ousness") > 0 ? word.substring(0, word.length - 4) : word;
    if (word.endsWith("aliti")) return _stemvcCount(word, "aliti") > 0 ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("iviti")) return _stemvcCount(word, "iviti") > 0 ? word.substring(0, word.length - 3) + "e" : word;
    if (word.endsWith("biliti")) return _stemvcCount(word, "biliti") > 0 ? word.substring(0, word.length - 5) + "le" : word;
    if (word.endsWith("logi")) return _stemvcCount(word, "logi") > 0 ? word.substring(0, word.length - 1) : word;
    return word;
  }
  
  String _strip3(String word) {
    if (word.endsWith("icate")) return _stemvcCount(word, "icate") > 0 ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("ative")) return _stemvcCount(word, "ative") > 0 ? word.substring(0, word.length - 5) : word;
    if (word.endsWith("alize")) return _stemvcCount(word, "alize") > 0 ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("iciti")) return _stemvcCount(word, "iciti") > 0 ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("ical")) return _stemvcCount(word, "ical") > 0 ? word.substring(0, word.length - 2) : word;
    if (word.endsWith("ful")) return _stemvcCount(word, "ful") > 0 ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("ness")) return _stemvcCount(word, "ness") > 0 ? word.substring(0, word.length - 4) : word;
    return word;
  }
  
  int _stemvcCount(String word, String suffix) {
    if (!word.endsWith(suffix)) return -1;
    return _vcCount(word.substring(0, word.length - suffix.length));
  }
  
  String _strip4(String word) {
    // if (_vcCount(word) < 2) {
    //   return word;
    // }
    if ( word.endsWith("al")) return (_stemvcCount(word, "al") > 1) ? word.substring(0, word.length - 2) : word;
    if (word.endsWith("ance")) return (_stemvcCount(word, "ance") > 1) ? word.substring(0, word.length - 4) : word;
    if (word.endsWith("ence")) return (_stemvcCount(word, "ence") > 1) ? word.substring(0, word.length - 4) : word;
    if (word.endsWith("er")) return (_stemvcCount(word, "er") > 1) ? word.substring(0, word.length - 2) : word;
    if (word.endsWith("ic")) return (_stemvcCount(word, "ic") > 1) ? word.substring(0, word.length - 2) : word;
    if (word.endsWith("able")) return (_stemvcCount(word, "able") > 1) ? word.substring(0, word.length - 4) : word;
    if (word.endsWith("ible")) return (_stemvcCount(word, "ible") > 1) ? word.substring(0, word.length - 4) : word;
    if (word.endsWith("ant")) return (_stemvcCount(word, "ant") > 1) ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("ement")) return (_stemvcCount(word, "ement") > 1) ? word.substring(0, word.length - 5) : word;
    if (word.endsWith("ment")) return (_stemvcCount(word, "ment") > 1) ? word.substring(0, word.length - 4) : word;
    if (word.endsWith("ent")) return (_stemvcCount(word, "ent") > 1) ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("sion")) return (_stemvcCount(word, "ion") > 1) ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("tion")) return (_stemvcCount(word, "ion") > 1) ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("ou")) return (_stemvcCount(word, "ou") > 1) ? word.substring(0, word.length - 2) : word;
    if (word.endsWith("ism")) return (_stemvcCount(word, "ism") > 1) ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("ate")) return (_stemvcCount(word, "ate") > 1) ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("iti")) return (_stemvcCount(word, "iti") > 1) ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("ous")) return (_stemvcCount(word, "ous") > 1) ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("ive")) return (_stemvcCount(word, "ive") > 1) ? word.substring(0, word.length - 3) : word;
    if (word.endsWith("ize")) return (_stemvcCount(word, "ize") > 1) ? word.substring(0, word.length - 3) : word;
    return word;
  }

  String _strip5a(String word) {
    if (word.endsWith('e')) {
      if (_stemvcCount(word, "e") > 1) {
        return word.substring(0, word.length - 1);
      }
      if (_stemvcCount(word, "e") == 1 && !_cvcEnd(word.substring(0, word.length - 1))) {
        return word.substring(0, word.length - 1);
      }
    }
    return word;
  }

  String _strip5b(String word) {
    if (_stemvcCount(word, "ll") > 1) {
      return word.substring(0, word.length - 1);
    }
    return word;
  }

  String stem(String word) {
    String newword = word.toLowerCase();
    if (newword.length < 3) {
      return newword;
    }
    return _strip5b(
      _strip5a(
        _strip4(
          _strip3(
            _strip2(
              _strip1c(
                _strip1b(
                  _strip1a(newword)
                )
              )
            )
          )
        )
      )
    );
  }

  String stem_document(String document) {
    List<String> words = (RegExp(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b|[.,!?;:"()&$]')).allMatches(document).map((match) => match.group(0)!).toList();
    if (_stopwords) {
      words = removeStopWords(words, 'english');
    }
    return words.map((word) => stem(word)).join(" ");
  }

  @override
  String transform(String text) => stem_document(text);

}