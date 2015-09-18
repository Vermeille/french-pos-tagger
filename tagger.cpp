#include <vector>
#include <fstream>
#include <string>
#include <utility>
#include <map>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

static const int kVocabSize = 100000;
static const int kNbKinds = 31;
static const double kLearningRate = 0.01;
static const unsigned int kNotFound = -1;

enum Kind {
    ADJ,
    ADJWH,
    ADV,
    ADVWH,
    CC,
    CLO,
    CLR,
    CLS,
    CS,
    DET,
    DETWH,
    ET,
    I,
    ID,
    NC,
    NPP,
    P,
    P_D,
    PONCT,
    PREF,
    PRO,
    PROREL,
    PROWH,
    U,
    V,
    VIMP,
    VINF,
    VPP,
    VPR,
    VS,
    NBR,
};

Kind TextToPOS(const std::string& str) {
    if (str =="ADJ") return ADJ;
    if (str =="ADJWH") return ADJWH;
    if (str =="ADV") return ADV;
    if (str =="ADVWH") return ADVWH;
    if (str =="CC") return CC;
    if (str =="CLO") return CLO;
    if (str =="CLR") return CLR;
    if (str =="CLS") return CLS;
    if (str =="CS") return CS;
    if (str =="DET") return DET;
    if (str =="DETWH") return DETWH;
    if (str =="ET") return ET;
    if (str =="I") return I;
    if (str =="ID") return ID;
    if (str =="NC") return NC;
    if (str =="NPP") return NPP;
    if (str =="P") return P;
    if (str =="P+D") return P_D;
    if (str =="PONCT") return PONCT;
    if (str =="PREF") return PREF;
    if (str =="PRO") return PRO;
    if (str =="PROREL") return PROREL;
    if (str =="PROWH") return PROWH;
    if (str =="U") return U;
    if (str =="V") return V;
    if (str =="VIMP") return VIMP;
    if (str =="VINF") return VINF;
    if (str =="VPP") return VPP;
    if (str =="VPR") return VPR;
    if (str =="VS") return VS;
    if (str =="NBR") return NBR;
    return (Kind)kNotFound;
};

std::string POSToText(Kind pos) {
    if (pos == ADJ) return "ADJ";
    if (pos == ADJWH) return "ADJWH";
    if (pos == ADV) return "ADV";
    if (pos == ADVWH) return "ADVWH";
    if (pos == CC) return "CC";
    if (pos == CLO) return "CLO";
    if (pos == CLR) return "CLR";
    if (pos == CLS) return "CLS";
    if (pos == CS) return "CS";
    if (pos == DET) return "DET";
    if (pos == DETWH) return "DETWH";
    if (pos == ET) return "ET";
    if (pos == I) return "I";
    if (pos == ID) return "ID";
    if (pos == NC) return "NC";
    if (pos == NPP) return "NPP";
    if (pos == P) return "P";
    if (pos == P_D) return "P+D";
    if (pos == PONCT) return "PONCT";
    if (pos == PREF) return "PREF";
    if (pos == PRO) return "PRO";
    if (pos == PROREL) return "PROREL";
    if (pos == PROWH) return "PROWH";
    if (pos == U) return "U";
    if (pos == V) return "V";
    if (pos == VIMP) return "VIMP";
    if (pos == VINF) return "VINF";
    if (pos == VPP) return "VPP";
    if (pos == VPR) return "VPR";
    if (pos == VS) return "VS";
    if (pos == NBR) return "NBR";
    return "ERROR";
};

typedef std::pair<unsigned int, Kind> TaggedWord;
typedef std::vector<TaggedWord> Document;


static const int kNbSuffixes = 35;
enum Suffix {
    /* ADV */
    MENT,
    MANT,
    ION,
    IONS,
    /* VERB */
    ISSONS,
    ISSIONS,
    ISSEZ,
    ISSIEZ,
    AIS,
    AIT,
    AIENT,
    ONT,
    ONS,
    ENT,
    ER,
    IR,
    INDRE,
    OIRE,
    EZ,
    ES,
    S,
    E,
    T,
    /* ADJ */
    IQUE,
    E_ACCENTED,
    E_ACCENTED_S,
    EES,
    AUX,
    AL,
    ALE,
    ELLE,
    ELLES,
    ALES,
    ETTE,
    ETTES,
};

const char* suffixes_str[] = {
    /* ADV */
    "ment",
    "mant",
    "ion",
    "ions",
    /* VERB */
    "issons",
    "issez",
    "issions",
    "issiez",
    "ais",
    "ait",
    "aient",
    "ont",
    "ons",
    "ent",
    "er",
    "ir",
    "indre",
    "oire",
    "ez",
    "es",
    "s",
    "e",
    "t",
    /* ADJ */
    "ique",
    "é",
    "és",
    "ées",
    "aux",
    "al",
    "ale",
    "elle",
    "elles",
    "ales",
    "ette",
    "ettes",
};

static const int kNbPrefixes = 7;
enum Prefix {
    RE,
    RE_ACCENTED,
    MIS,
    MAL,
    AN,
    A,
    DES,
};

const char* prefixes_str[] = {
    "re",
    "ré",
    "mis",
    "mal",
    "an",
    "a",
    "dés",
};

enum Caps {
    NO_CAPS,
    FIRST_LETTER_CAPS,
    ALL_CAPS,
};

struct WordFeatures {
    std::string as_string;
    Suffix suffix;
    Prefix prefix;
    unsigned int idx;
    Caps caps;
    bool has_numbers;

    WordFeatures() : suffix((Suffix)kNotFound), idx(kNotFound), caps(NO_CAPS) {}
    WordFeatures(const std::string& str) : as_string(str), idx(kNotFound), caps(NO_CAPS) {}
};

std::vector<WordFeatures> word_features(kVocabSize, WordFeatures());

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline bool starts_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.begin(), ending.end(), value.begin());
}
/* WORD WEIGHT */

std::vector<double> word_weight[kNbKinds];

double WordF(Kind target, const WordFeatures& w) {
    if (w.idx == kNotFound)
        return 0;

    return word_weight[target][w.idx];
}

void WordF_Backprop(const TaggedWord& w, const double* probabilities) {
    if (w.first == kNotFound)
        return;

    for (int k = 0; k < kNbKinds; ++k) {
        double target = (w.second == k) ? 1 : 0;
        word_weight[k][w.first] += kLearningRate * (target - probabilities[k]);
    }
}

/* SUFFIXES */

double suffixes[kNbKinds][kNbSuffixes];

double SuffixF(Kind target, const WordFeatures& w) {
    unsigned int word_suffix = w.suffix;
    if (word_suffix != kNotFound)
        return suffixes[target][word_suffix];
    else
        return 0;
}

void SuffixF_Backprop(const TaggedWord& w, const double* probabilities) {
    unsigned int word_suffix = word_features[w.first].suffix;
    if (word_suffix == kNotFound)
        return;

    for (int k = 0; k < kNbKinds; ++k) {
        double target = k == w.second ? 1 : 0;
        suffixes[k][word_suffix] += kLearningRate * (target - probabilities[k]);
    }
}

/* PREFIXES */

double prefixes[kNbKinds][kNbPrefixes];

double PrefixF(Kind target, const WordFeatures& w) {
    unsigned int word_prefix = w.prefix;
    if (word_prefix != kNotFound)
        return prefixes[target][word_prefix];
    else
        return 0;
}

void PrefixF_Backprop(const TaggedWord& w, const double* probabilities) {
    unsigned int word_prefix = word_features[w.first].prefix;
    if (word_prefix == kNotFound)
        return;

    for (int k = 0; k < kNbKinds; ++k) {
        double target = k == w.second ? 1 : 0;
        prefixes[k][word_prefix] += kLearningRate * (target - probabilities[k]);
    }
}

/* CAPITALIZATION */

double capitalization[kNbKinds][3];

double CapsF(Kind target, const WordFeatures& w) {
    return capitalization[target][w.caps];
}

void CapsF_Backprop(const TaggedWord& w, const WordFeatures& wf, const double* probabilities) {
    for (int k = 0; k < kNbKinds; ++k) {
        double target = k == w.second ? 1 : 0;
        capitalization[k][wf.caps] += kLearningRate * (target - probabilities[k]);
    }
}

/* TRANSITION */

double transitions[kNbKinds][kNbKinds];

double TransitionF(Kind target, const TaggedWord& prev) {
    if ((unsigned int)prev.second == kNotFound) {
        return 0;
    }

    return transitions[target][prev.second];
}

void TransitionF_Backprop(const TaggedWord& w, const TaggedWord& prev, const double* probabilities) {
    if ((unsigned int)prev.second == kNotFound) {
        return;
    }

    for (int k = 0; k < kNbKinds; ++k) {
        double target = k == w.second ? 1 : 0;
        transitions[k][prev.second] += kLearningRate * (target - probabilities[k]);
    }
}

/* FIRST WORD */

double first_word[kNbKinds][2];

double FirstWordF(Kind target, const TaggedWord& prev) {
        return first_word[target][(prev.first == 0 /* dot */) ? 0 : 1];
}

void FirstWordF_Backprop(const TaggedWord& w, const TaggedWord& prev, const double* probabilities) {
    int idx = (prev.first == 0 /* dot */) ? 0 : 1;
    for (int k = 0; k < kNbKinds; ++k) {
        double target = k == w.second ? 1 : 0;
        first_word[k][idx] += kLearningRate * (target - probabilities[k]);
    }
}

/* NUMBERS */

double numbers[kNbKinds][2];

double HasNumbersF(Kind target, const WordFeatures& w) {
        return numbers[target][w.has_numbers ? 0 : 1];
}

void HasNumbersF_Backprop(const TaggedWord& w, const WordFeatures& wf, const double* probabilities) {
    int idx = (wf.has_numbers) ? 0 : 1;
    for (int k = 0; k < kNbKinds; ++k) {
        double target = k == w.second ? 1 : 0;
        numbers[k][idx] += kLearningRate * (target - probabilities[k]);
    }
}

double RunAllFeatures(Kind k, const WordFeatures& w, const TaggedWord& prev) {
    double sum = 0;
    sum += WordF(k, w);
    sum += SuffixF(k, w);
    sum += CapsF(k, w);
    sum += TransitionF(k, prev);
    sum += FirstWordF(k, prev);
    sum += HasNumbersF(k, w);
    return sum;
}

Kind ComputeClass(const WordFeatures& w, const TaggedWord& prev, double* probabilities) {
    double total = 0;
    for (int k = 0; k < kNbKinds; ++k) {
        probabilities[k] = std::exp(RunAllFeatures((Kind) k, w, prev));
        total += probabilities[k];
    }

    int max = 0;
    for (int k = 0; k < kNbKinds; ++k) {
        probabilities[k] /= total;
        if (probabilities[k] > probabilities[max]) {
            max = k;
        }
    }
    return (Kind)max;
}

void Backprop(const TaggedWord& tw, const WordFeatures& wf, const TaggedWord& prev,
        const double* probabilities) {
    WordF_Backprop(tw, probabilities);
    SuffixF_Backprop(tw, probabilities);
    CapsF_Backprop(tw, wf, probabilities);
    TransitionF_Backprop(tw, prev, probabilities);
    FirstWordF_Backprop(tw, prev, probabilities);
    HasNumbersF_Backprop(tw, wf, probabilities);
}

double ComputeNLL(double* probas) {
    double nll = 0;
    for (int i = 0; i < kNbKinds; ++i) {
        nll += std::log(probas[i]);
    }
    return -nll;
}

std::map<std::string, int> dict;

WordFeatures BuildFeatures(const std::string& w) {
    WordFeatures features(w);

    features.suffix = (Suffix)kNotFound;
    for (int i = 0; i < kNbSuffixes; ++i) {
        if (ends_with(w, suffixes_str[i])) {
            features.suffix = (Suffix) i;
            break;
        }
    }

    features.prefix = (Prefix)kNotFound;
    for (int i = 0; i < kNbPrefixes; ++i) {
        if (starts_with(w, prefixes_str[i])) {
            features.prefix = (Prefix) i;
            break;
        }
    }

    if (std::all_of(w.begin(), w.end(), ::isupper))
        features.caps = ALL_CAPS;
    else if (isupper(w[0]))
        features.caps = FIRST_LETTER_CAPS;

    features.has_numbers = std::any_of(w.begin(), w.end(), ::isdigit);

    std::transform(features.as_string.begin(), features.as_string.end(), features.as_string.begin(),
            ::tolower);
    auto res = dict.find(features.as_string);
    features.idx = res == dict.end() ? kNotFound : res->second;
    return features;
}

void Init() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0, 1);

    for (int i = 0; i < kNbKinds; ++i) {
        for (int j = 0; j < kNbSuffixes; ++j) {
            suffixes[i][j] = d(gen);
        }

        for (int j = 0; j < 3; ++j) {
            capitalization[i][j] = d(gen);
        }

        word_weight[i].resize(kVocabSize);
        for (int j = 0; j < kVocabSize; ++j) {
            word_weight[i][j] = d(gen);
        }

        for (int j = 0; j < 2; ++j) {
            numbers[i][j] = d(gen);
        }

        for (int j = 0; j < 2; ++j) {
            first_word[i][j] = d(gen);
        }
    }
}

Document BuildDocument(char* filename) {
    Document doc;

    std::ifstream input(filename);
    std::string w;
    std::string pos;
    unsigned int max_word_id = 1;

    dict["."] = 0;
    word_features[0] = BuildFeatures(".");
    while (input) {
        unsigned word_id = max_word_id;
        input >> w;
        std::string case_i_w = w;
        std::transform(case_i_w.begin(), case_i_w.end(), case_i_w.begin(), ::tolower);
        auto res = dict.insert(std::make_pair(case_i_w, max_word_id));
        if (!res.second) {
            word_id = res.first->second;
        } else {
            word_features[word_id] = BuildFeatures(w);
            ++max_word_id;
        }
        input >> pos;

        doc.push_back(std::make_pair(word_id, TextToPOS(pos)));
    }
    return doc;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: ./" << argv[0] << " <training set>\n";
        return 1;
    }
    Init();
    Document doc = BuildDocument(argv[1]);

    for (int epoch = 0; epoch < 150; ++epoch) {
        double nll = 0;
        double probas[kNbKinds];
        int nb_correct = 0;
        int nb_tokens = 0;
        TaggedWord prev = std::make_pair(0, PONCT);
        for (size_t i = 0; i < doc.size(); ++i) {
            const WordFeatures& wf = word_features[doc[i].first];
            Kind predicted = ComputeClass(wf, prev, probas);
            nb_correct += predicted == doc[i].second ? 1 : 0;
            ++nb_tokens;

            nll += ComputeNLL(probas);

            Backprop(doc[i], wf, prev, probas);

            prev = doc[i];

            if (i % 10000 == 0) {
                std::cout << nb_correct << " / " << nb_tokens << " (" << ((double) nb_correct *100 / nb_tokens) << "%)" << std::endl;
            }
        }
        std::cout << nll << "\n" << nb_correct << " / " << nb_tokens << "\n=======\n";
    }

    std::cout << "==== TESTING ====\n";
    if (argc == 2) {
        TaggedWord prev = std::make_pair(0, PONCT);
        while (std::cin) {
            std::string w;
            std::cin >> w;

            std::cout << w << ":\n";

            double probas[kNbKinds];
            WordFeatures wf = BuildFeatures(w);

            if (wf.idx != kNotFound)
                std::cout << "  idx: " << wf.idx << "\n";

            Kind k = ComputeClass(wf, prev, probas);
            prev = std::make_pair(wf.idx, k);
            std::cout << "  POS: " << POSToText(k) << " (confidence: " << probas[k] *100 << " %)\n";
        }
    } else {
        double nll = 0;
        double probas[kNbKinds];
        int nb_correct = 0;
        int nb_tokens = 0;
        std::string w;
        std::string pos_str;
        std::ifstream test(argv[2]);
        TaggedWord prev = std::make_pair(0, PONCT);
        while (test) {
            test >> w;
            test >> pos_str;
            const WordFeatures& wf = BuildFeatures(w);
            Kind predicted = ComputeClass(wf, prev, probas);
            if (predicted == TextToPOS(pos_str)) {
                ++nb_correct;
            } else {
                std::cout << w << "(" << wf.idx << ") said " << POSToText(predicted) << " but " << pos_str << std::endl;
            }
            ++nb_tokens;

            nll += ComputeNLL(probas);

            prev = std::make_pair(wf.idx, predicted);

            if (nb_tokens % 10000 == 0) {
                std::cout << nb_correct << " / " << nb_tokens << " (" << ((double) nb_correct *100 / nb_tokens) << "%)" << std::endl;
            }
        }
        std::cout << nll << "\n" << nb_correct << " / " << nb_tokens << "\n=======\n";
    }

    return 0;
}
