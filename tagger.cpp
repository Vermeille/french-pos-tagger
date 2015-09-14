#include <vector>
#include <fstream>
#include <string>
#include <utility>
#include <map>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

static const int kNbKinds = 30;
static const double kLearningRate = 0.1;

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
    return (Kind)-1;
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
    return "ERROR";
};

typedef std::pair<unsigned int, Kind> TaggedWord;
typedef std::vector<TaggedWord> Document;


static const int kNbSuffixes = 26;
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
    EZ,
    S,
    T,
    /* ADJ */
    IQUE,
    E,
    ES,
    EES,
    AUX,
    AL,
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
    "ez",
    "s",
    "t",
    /* ADJ */
    "ique",
    "é",
    "és",
    "ées",
    "aux",
    "al",
};

enum Caps {
    NO_CAPS,
    FIRST_LETTER_CAPS,
    ALL_CAPS,
};

struct WordFeatures {
    std::string as_string;
    Suffix suffix;
    unsigned int idx;
    Caps caps;

    WordFeatures() : suffix((Suffix)-1), idx(-1), caps(NO_CAPS) {}
    WordFeatures(const std::string& str) : as_string(str), idx(-1), caps(NO_CAPS) {}
};

std::map<unsigned int, WordFeatures> word_features;

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

/* WORD WEIGHT */

std::map<unsigned int, double> word_weight[kNbKinds];

double WordF(Kind target, const WordFeatures& w) {
    return word_weight[target][w.idx];
}

void WordF_Backprop(const TaggedWord& w, const double* probabilities) {
    for (int k = 0; k < kNbKinds; ++k) {
        double target = (w.second == k) ? 1 : 0;
        word_weight[k][w.first] += kLearningRate * (target - probabilities[k]);
    }
}

/* SUFFIXES */

double suffixes[kNbKinds][kNbSuffixes];

double SuffixF(Kind target, const WordFeatures& w) {
    int word_suffix = w.suffix;
    if (word_suffix != -1)
        return suffixes[target][word_suffix];
    else
        return 0;
}

void SuffixF_Backprop(const TaggedWord& w, const double* probabilities) {
    int word_suffix = word_features[w.first].suffix;
    if (word_suffix == -1)
        return;

    for (int k = 0; k < kNbKinds; ++k) {
        double target = k == w.second ? 1 : 0;
        suffixes[k][word_suffix] += kLearningRate * (target - probabilities[k]);
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

double RunAllFeatures(Kind k, const WordFeatures& w) {
    double sum = 0;
    sum += WordF(k, w);
    sum += SuffixF(k, w);
    sum += CapsF(k, w);
    return sum;
}

Kind ComputeClass(const WordFeatures& w, double* probabilities) {
    double total = 0;
    for (int k = 0; k < kNbKinds; ++k) {
        probabilities[k] = std::exp(RunAllFeatures((Kind) k, w));
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

void Backprop(const TaggedWord& tw, const WordFeatures& wf, const double* probabilities) {
    WordF_Backprop(tw, probabilities);
    SuffixF_Backprop(tw, probabilities);
    CapsF_Backprop(tw, wf, probabilities);
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

    features.suffix = (Suffix)-1;
    for (int i = 0; i < kNbSuffixes; ++i) {
        if (ends_with(w, suffixes_str[i])) {
            features.suffix = (Suffix) i;
            break;
        }
    }

    if (std::all_of(w.begin(), w.end(), ::isupper))
        features.caps = ALL_CAPS;
    else if (std::any_of(w.begin(), w.end(), ::isupper))
        features.caps = FIRST_LETTER_CAPS;

    std::transform(features.as_string.begin(), features.as_string.end(), features.as_string.begin(),
            ::tolower);
    auto res = dict.find(features.as_string);
    features.idx = res == dict.end() ? -1 : res->second;
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
    }
}

Document BuildDocument(char* filename) {
    Document doc;

    std::ifstream input(filename);
    std::string w;
    std::string pos;
    unsigned int max_word_id = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0, 1);

    while (input) {
        unsigned word_id = max_word_id;
        input >> w;
        std::string case_i_w = w;
        std::transform(case_i_w.begin(), case_i_w.end(), case_i_w.begin(), ::tolower);
        auto res = dict.insert(std::make_pair(case_i_w, max_word_id));
        if (!res.second) {
            word_id = res.first->second;
        } else {
            for (int k = 0; k < kNbKinds; ++k) {
                word_weight[k][word_id] = d(gen);
            }
            word_features[word_id] = BuildFeatures(w);
            ++max_word_id;
        }
        input >> pos;

        doc.push_back(std::make_pair(word_id, TextToPOS(pos)));
    }
    return doc;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./" << argv[0] << " <training set>\n";
        return 1;
    }
    Init();
    Document doc = BuildDocument(argv[1]);

    for (int epoch = 0; epoch < 5; ++epoch) {
        double nll = 0;
        double probas[kNbKinds];
        int nb_correct = 0;
        int nb_tokens = 0;
        for (size_t i = 0; i < doc.size(); ++i) {
            const WordFeatures& wf = word_features[doc[i].first];
            Kind predicted = ComputeClass(wf, probas);
            nb_correct += predicted == doc[i].second ? 1 : 0;
            ++nb_tokens;

            nll += ComputeNLL(probas);

            Backprop(doc[i], wf, probas);
            if (i % 10000 == 0) {
                std::cout << nb_correct << " / " << nb_tokens << " (" << ((double) nb_correct *100 / nb_tokens) << "%)" << std::endl;
            }
        }
        std::cout << nll << "\n" << nb_correct << " / " << nb_tokens << "\n=======\n";
    }

    while (std::cin) {
        std::string w;
        std::cin >> w;

        std::cout << w << ":\n";

        double probas[kNbKinds];
        WordFeatures wf = BuildFeatures(w);

        if (wf.idx != (unsigned int) -1)
            std::cout << "  idx: " << wf.idx << "\n";

        Kind k = ComputeClass(wf, probas);
        std::cout << "  POS: " << POSToText(k) << " (confidence: " << probas[k] *100 << " %)\n";
    }

    return 0;
}