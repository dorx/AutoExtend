#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocab_word {
    long long cn;
    char *word;
};

struct pair_word {
    int u, v;
};

char data_file[MAX_STRING], seed_file[MAX_STRING], out_file[MAX_STRING], cnt_file[MAX_STRING], label[MAX_STRING];
struct vocab_word *vocab;
int window = 5, max_sample = -1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, doc_size = 0, seed_size = 0, train_words = 0;

std::vector< std::vector<int> > doc;
std::vector<pair_word> seeds;

void swap(int &a, int &b)
{
    int t;
    t = a;
    a = b;
    b = t;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            }
            else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    int a, size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    size = vocab_size;
    for (a = 0; a < size; a++) {
        // Hash will be re-computed, as after the sorting it is not actual
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
    }
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

void LearnVocabFromTrainFile() {
    char word[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    fin = fopen(data_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    AddWordToVocab((char *)"</s>");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        train_words++;
        if (train_words % 100000 == 0) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        i = SearchVocab(word);
        if (i == -1) {
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        }
        else vocab[i].cn++;
    }
    SortVocab();
    printf("Words in train file: %lld\n", train_words);
    printf("Vocab size: %lld\n", vocab_size);
    fclose(fin);
}

void ReadSeed()
{
    FILE *fi;
    char su[MAX_STRING], sv[MAX_STRING];
    int u, v;
    pair_word pair;
    
    fi = fopen(seed_file, "rb");
    seeds.clear();
    seed_size = 0;
    while (fscanf(fi, "%s %s", su, sv) == 2)
    {
        u = SearchVocab(su);
        v = SearchVocab(sv);
        if (u == -1 || v == -1) continue;
        
        pair.u = u;
        pair.v = v;
        
        seeds.push_back(pair);
        seed_size++;
    }
    printf("Seed size: %lld\n", seed_size);
    fclose(fi);
}

void ReadDoc()
{
    FILE *fi;
    int curword;
    char word[MAX_STRING];
    std::vector<int> vt;
    
    fi = fopen(data_file, "rb");
    while (1)
    {
        vt.clear();
        while (1)
        {
            ReadWord(word, fi);
            if (feof(fi)) break;
            curword = SearchVocab(word);
            if (curword == -1) continue;
            if (curword == 0) break;
            vt.push_back(curword);
        }
        if (feof(fi)) break;
        
        doc_size++;
        doc.push_back(vt);
    }
    fclose(fi);
    printf("Doc size: %lld\n", doc_size);
}

void Output()
{
    FILE *foc, *fod;
    int u, v, p, q, len, cnt;
    
    foc = fopen(cnt_file, "wb");
    fod = fopen(out_file, "wb");
    for (int s = 0; s != seed_size; s++)
    {
        u = seeds[s].u;
        v = seeds[s].v;
        
        cnt = 0;
        for (int d = 0; d != doc_size; d++)
        {
            p = -1;
            q = -1;
            
            len = (int)(doc[d].size());
            for (int k = 0; k != len; k++)
            {
                if (doc[d][k] == u) p = k;
                if (doc[d][k] == v) q = k;
            }
            
            if (p == -1 || q == -1) continue;
            
            //if (abs(p - q) < 10) continue;
            if (p > q) swap(p, q);
            if (q - p > 10) continue;
            
            fprintf(fod, "%s\n", label);
            for (int k = p - window; k != p; k++)
            {
                if (k < 0 || k >= len) fprintf(fod, "NAN ");
                else if (doc[d][k] == -1) fprintf(fod, "NAN ");
                else fprintf(fod, "%s ", vocab[doc[d][k]].word);
            }
            fprintf(fod, "\n");
            for (int k = p + 1; k != q; k++)
            {
                if (doc[d][k] == -1) fprintf(fod, "NAN ");
                else fprintf(fod, "%s ", vocab[doc[d][k]].word);
            }
            fprintf(fod, "\n");
            for (int k = q + 1; k <= q + window; k++)
            {
                if (k < 0 || k >= len) fprintf(fod, "NAN ");
                else if (doc[d][k] == -1) fprintf(fod, "NAN ");
                else fprintf(fod, "%s ", vocab[doc[d][k]].word);
            }
            fprintf(fod, "\n");
            
            cnt++;
            if (cnt == max_sample)
                break;
        }
        fprintf(foc, "%d\n", cnt);
    }
    fclose(foc);
    fclose(fod);
}

void TrainModel()
{
    LearnVocabFromTrainFile();
    ReadDoc();
    ReadSeed();
    Output();
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        return 0;
    }
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) strcpy(data_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-seed", argc, argv)) > 0) strcpy(seed_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(out_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-count", argc, argv)) > 0) strcpy(cnt_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-label", argc, argv)) > 0) strcpy(label, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-max", argc, argv)) > 0) max_sample = atoi(argv[i + 1]);
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    TrainModel();
    return 0;
}