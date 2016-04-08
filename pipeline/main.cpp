#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <map>
#include <algorithm>
#include "NeuNet.h"

#define MAX_STRING 100
#define MAX_SENTENCE 200
#define gradient_cutoff 0
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocab_word {
    int cn;
    char word[MAX_STRING];
};

struct Sample
{
    int label;
    std::vector<int> cont_l, cent, cont_r;
};

typedef Eigen::Matrix< real, Eigen::Dynamic,
Eigen::Dynamic, Eigen::ColMajor | Eigen::AutoAlign >
BLPMatrix;

vocab_word *vocab;
int *vocab_hash;
int vocab_size, label_size = 2;
int train_size = 0, test_size = 0, data_size;
char data_file[MAX_STRING], vector_file[MAX_STRING], model_file[MAX_STRING], output_file[MAX_STRING];
int debug_mode = 2, epoch, iters = 20, batch_size = 100, window = 3, mode;
real init_lr = 0.1, lr;

std::vector<Sample> data_set;
std::vector<Sample>::iterator iter_p, iter_q, iter_r;

/********** look-up table **********/
BLPMatrix vec;
int vector_size, feature_size;
real vec_initlr;

Neuron neu0, neu1, neu2, neu3;
Layer_conv conv0;
Layer_pooling_kmax pool1;
Layer_tanh act;
Layer_synapse syn2;
Layer_loss_softmax loss;

Neuron neu0l, neu0r, neu1l, neu1r;
Layer_synapse syn0l, syn0r, syn1l, syn1r;

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

int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

int AddWordToVocab(char *word, int pst) {
    unsigned int hash;
    strcpy(vocab[pst].word, word);
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = pst;
    return pst;
}

void LearnVocabFromTrainFile()
{
    FILE *fi = fopen(vector_file, "rb");
    if (fi == NULL) {
        printf("Vector file not found\n");
        exit(1);
    }
    
    fscanf(fi, "%d %d", &vocab_size, &vector_size);
    
    vocab = (struct vocab_word *)calloc(vocab_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    for (int k = 0; k < vocab_hash_size; k++) vocab_hash[k] = -1;
    vec.resize(vector_size, vocab_size);
    
    char word[MAX_STRING], ch;
    real f;
    for (int k = 0; k != vocab_size; k++)
    {
        fscanf(fi, "%s", word);
        ch = fgetc(fi);
        AddWordToVocab(word, k);
        for (int c = 0; c != vector_size; c++)
        {
            fread(&f, sizeof(real), 1, fi);
            vec(c, k) = f;
        }
    }
    fclose(fi);
    if (debug_mode > 0) {
        printf("Vocab size: %d\n", vocab_size);
        printf("Vector dim: %d\n", vector_size);
    }
}

void BuildNet()
{
    /********** global paras **********/
    iters = 20;
    init_lr = 0.05;
    batch_size = 8;
    feature_size = 100;
    
    /********** look-up table **********/
    vec_initlr = init_lr;
    
    /********** layer 0 **********/
    conv0.alloc(vector_size, feature_size, 3);
    conv0.set_init_lr(init_lr);
    conv0.set_mmt(0.5);
    conv0.set_wd(0.0001);
    conv0.init();
    
    pool1.alloc(feature_size, 100);
    pool1.resize(feature_size, 1);
    
    syn2.alloc(feature_size, label_size);
    syn2.set_init_lr(init_lr);
    syn2.set_mmt(0.5);
    syn2.set_wd(0.0001);
    syn2.init();
    
    loss.alloc(label_size);
    
    /********** layer left **********/
    syn0l.alloc(window * vector_size, feature_size);
    syn0l.set_init_lr(init_lr);
    syn0l.set_mmt(0.5);
    syn0l.set_wd(0.0001);
    syn0l.init();
    
    syn0r.alloc(window * vector_size, feature_size);
    syn0r.set_init_lr(init_lr);
    syn0r.set_mmt(0.5);
    syn0r.set_wd(0.0001);
    syn0r.init();
    
    syn1l.alloc(feature_size, label_size);
    syn1l.set_init_lr(init_lr);
    syn1l.set_mmt(0.5);
    syn1l.set_wd(0.0001);
    syn1l.init();
    
    syn1r.alloc(feature_size, label_size);
    syn1r.set_init_lr(init_lr);
    syn1r.set_mmt(0.5);
    syn1r.set_wd(0.0001);
    syn1r.init();
}

real ComputeLr(real init_value)
{
    return init_value;
}

void ReadData()
{
    FILE *fi;
    int curlabel, curword;
    std::vector<int> cursen;
    char word[MAX_STRING];
    Sample smp;
    
    data_size = 0;
    
    fi = fopen(data_file, "rb");
    while (1)
    {
        if (fscanf(fi, "%d", &curlabel) != 1) break;
        ReadWord(word, fi);
        
        cursen.clear();
        while (1)
        {
            ReadWord(word, fi);
            if (strcmp(word, "</s>") == 0) break;
            curword = SearchVocab(word);
            //if (curword == -1) continue;
            cursen.push_back(curword);
        }
        smp.cont_l = cursen;
        
        cursen.clear();
        while (1)
        {
            ReadWord(word, fi);
            if (strcmp(word, "</s>") == 0) break;
            curword = SearchVocab(word);
            if (curword == -1) continue;
            cursen.push_back(curword);
        }
        smp.cent = cursen;
        
        cursen.clear();
        while (1)
        {
            ReadWord(word, fi);
            if (strcmp(word, "</s>") == 0) break;
            curword = SearchVocab(word);
            //if (curword == -1) continue;
            cursen.push_back(curword);
        }
        smp.cont_r = cursen;
        
        data_size++;
        smp.label = curlabel;
        data_set.push_back(smp);
    }
    fclose(fi);
}

void flush()
{
    neu0.flush();
    neu1.flush();
    neu2.flush();
    neu3.flush();
    
    neu0l.flush();
    neu1l.flush();
    
    neu0r.flush();
    neu1r.flush();
}

void forward()
{
    // cent
    
    conv0.f_prop(neu0, neu1);
    
    pool1.f_prop(neu1, neu2);
    
    act.f_prop(neu2, neu2);
    
    syn2.f_prop(neu2, neu3);
    
    // cont left
    
    syn0l.f_prop(neu0l, neu1l);
    
    act.f_prop(neu1l, neu1l);
    
    syn1l.f_prop(neu1l, neu3);
    
    // cont right
    
    syn0r.f_prop(neu0r, neu1r);
    
    act.f_prop(neu1r, neu1r);
    
    syn1r.f_prop(neu1r, neu3);
    
    // loss
    
    loss.f_prop(neu3, neu3);
}

void backward()
{
    // loss
    
    loss.b_prop(neu3, neu3);
    
    // cent
    
    syn2.b_prop(neu2, neu3);
    
    act.b_prop(neu2, neu2);
    
    pool1.b_prop(neu1, neu2);
    
    conv0.b_prop(neu0, neu1);
    
    // cont left
    
    syn1l.b_prop(neu1l, neu3);
    
    act.b_prop(neu1l, neu1l);
    
    syn0l.b_prop(neu0l, neu1l);
    
    // cont right
    
    syn1r.b_prop(neu1r, neu3);
    
    act.b_prop(neu1r, neu1r);
    
    syn0r.b_prop(neu0r, neu1r);
}

void gradient()
{
    conv0.c_grad(neu0, neu1);
    
    syn2.c_grad(neu2, neu3);
    
    syn0l.c_grad(neu0l, neu1l);
    
    syn1l.c_grad(neu1l, neu3);
    
    syn0r.c_grad(neu0r, neu1r);
    
    syn1r.c_grad(neu1r, neu3);
}

void update()
{
    conv0.update_adagrad();
    
    syn2.update_adagrad();
    
    syn0l.update_adagrad();
    
    syn1l.update_adagrad();
    
    syn0r.update_adagrad();
    
    syn1r.update_adagrad();
}

void SaveModel()
{
    FILE *fo;
    int rows, cols;
    
    fo = fopen(model_file, "wb");
    
    rows = (int)(conv0._para.rows());
    cols = (int)(conv0._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
        fprintf(fo, "%lf ", conv0._para(i, j));
    fprintf(fo, "\n");
    
    rows = (int)(syn2._para.rows());
    cols = (int)(syn2._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
        fprintf(fo, "%lf ", syn2._para(i, j));
    fprintf(fo, "\n");
    
    rows = (int)(syn0l._para.rows());
    cols = (int)(syn0l._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
        fprintf(fo, "%lf ", syn0l._para(i, j));
    fprintf(fo, "\n");
    
    rows = (int)(syn1l._para.rows());
    cols = (int)(syn1l._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
        fprintf(fo, "%lf ", syn1l._para(i, j));
    fprintf(fo, "\n");
    
    rows = (int)(syn0r._para.rows());
    cols = (int)(syn0r._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
        fprintf(fo, "%lf ", syn0r._para(i, j));
    fprintf(fo, "\n");
    
    rows = (int)(syn1r._para.rows());
    cols = (int)(syn1r._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
        fprintf(fo, "%lf ", syn1r._para(i, j));
    fprintf(fo, "\n");
    
    fclose(fo);
}

void LoadModel()
{
    FILE *fi;
    int rows, cols;
    float f;
    
    fi = fopen(model_file, "rb");
    
    rows = (int)(conv0._para.rows());
    cols = (int)(conv0._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
    {
        fscanf(fi, "%f", &f);
        conv0._para(i, j) = f;
    }
    
    rows = (int)(syn2._para.rows());
    cols = (int)(syn2._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
    {
        fscanf(fi, "%f", &f);
        syn2._para(i, j) = f;
    }
    
    rows = (int)(syn0l._para.rows());
    cols = (int)(syn0l._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
    {
        fscanf(fi, "%f", &f);
        syn0l._para(i, j) = f;
    }
    
    rows = (int)(syn1l._para.rows());
    cols = (int)(syn1l._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
    {
        fscanf(fi, "%f", &f);
        syn1l._para(i, j) = f;
    }
    
    rows = (int)(syn0r._para.rows());
    cols = (int)(syn0r._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
    {
        fscanf(fi, "%f", &f);
        syn0r._para(i, j) = f;
    }
    
    rows = (int)(syn1r._para.rows());
    cols = (int)(syn1r._para.cols());
    for (int i = 0; i != rows; i++) for (int j = 0; j != cols; j++)
    {
        fscanf(fi, "%f", &f);
        syn1r._para(i, j) = f;
    }
    
    fclose(fi);
}

double Test(std::vector<Sample>::iterator begin, std::vector<Sample>::iterator end)
{
    int cent_len, word, curlabel, ncorrect = 0, ntotal = end - begin, T = 0;
    std::vector<Sample>::iterator iter;
    FILE *fo;
    if (mode == 2) fo = fopen(output_file, "wb");
    for (iter = begin; iter != end; iter++, T++)
    {
        if (T % 100 == 0)
        {
            real accu = 0;
            if (T != 0) accu = (real)ncorrect / T;
            printf("%cEpoch: %d Accuracy: %f Progress: %.3lf%%", 13, epoch + 1, accu, T / (real)(ntotal + 1) * 100);
            fflush(stdout);
        }
        
        curlabel = (iter->label);
        
        cent_len = (int)((iter->cent).size());
        
        flush();
        
        // load neu0
        neu0.resize(vector_size, cent_len);
        for (int k = 0; k != cent_len; k++)
        {
            word = (iter->cent)[k];
            if (word == -1) neu0.ac.col(k).setZero();
            else neu0.ac.col(k) = vec.col(word);
        }
        
        // load neu0l
        neu0l.resize(vector_size, window);
        for (int k = 0; k != window; k++)
        {
            word = (iter->cont_l)[k];
            if (word == -1) neu0l.ac.col(k).setZero();
            else neu0l.ac.col(k) = vec.col(word);
        }
        
        // load neu0r
        neu0r.resize(vector_size, window);
        for (int k = 0; k != window; k++)
        {
            word = (iter->cont_r)[k];
            if (word == -1) neu0r.ac.col(k).setZero();
            else neu0r.ac.col(k) = vec.col(word);
        }
        
        forward();
        
        if (mode == 2)
        {
            for (int k = 0; k != label_size; k++)
                fprintf(fo, "%lf ", neu3.ac(k));
            fprintf(fo, "\n");
        }
        
        int correct = 1;
        for (int k = 0; k != label_size; k++)
            if (neu3.ac(curlabel) < neu3.ac(k))
                correct = 0;
        if (correct) ncorrect++;
    }
    return ncorrect / (double)(end - begin);
}

void Train(std::vector<Sample>::iterator begin, std::vector<Sample>::iterator end)
{
    int cent_len, word, curlabel, ntotal = end - begin, T = 0;
    std::vector<Sample>::iterator iter;
    for (iter = begin; iter != end; iter++, T++)
    {
        if (T % 100 == 0)
        {
            printf("%cEpoch: %d Progress: %.3lf%%", 13, epoch + 1, T / (real)(ntotal + 1) * 100);
            fflush(stdout);
        }
        
        curlabel = (iter->label);
        
        cent_len = (int)((iter->cent).size());
        
        flush();
        
        // load neu0
        neu0.resize(vector_size, cent_len);
        for (int k = 0; k != cent_len; k++)
        {
            word = (iter->cent)[k];
            if (word == -1) neu0.ac.col(k).setZero();
            else neu0.ac.col(k) = vec.col(word);
        }
        
        // load neu0l
        neu0l.resize(vector_size, window);
        for (int k = 0; k != window; k++)
        {
            word = (iter->cont_l)[k];
            if (word == -1) neu0l.ac.col(k).setZero();
            else neu0l.ac.col(k) = vec.col(word);
        }
        
        // load neu0r
        neu0r.resize(vector_size, window);
        for (int k = 0; k != window; k++)
        {
            word = (iter->cont_r)[k];
            if (word == -1) neu0r.ac.col(k).setZero();
            else neu0r.ac.col(k) = vec.col(word);
        }
        
        forward();
        
        loss.set_label(curlabel);
        
        backward();
        
        gradient();
        
        // update vector
        //lr = ComputeLr(vec_initlr);
        //for (int k = 0; k != len; k++)
        //{
        //    word = (iter->sen)[k];
        //    vec.col(word) += lr * neu0.er.col(k);
        //}
        
        // batch
        if ((T + 1) % batch_size == 0 || (T + 1 == ntotal))
            update();
    }
}


void TrainModel()
{
    Eigen::initParallel();
    Eigen::setNbThreads(4);
    
    LearnVocabFromTrainFile();
    ReadData();
    BuildNet();
    
    if (mode == 0)
    {
        std::random_shuffle(data_set.begin(), data_set.end());
        
        train_size = data_size * 0.7;
        test_size = data_size - train_size;
        
        iter_p = data_set.begin();
        iter_q = data_set.end();
        iter_r = data_set.begin() + train_size;
        
        if (debug_mode > 0) {
            printf("Label size: %d\n", label_size);
            printf("Data  size: %d\n", data_size);
            printf("Train size: %d\n", train_size);
            printf("Test  size: %d\n", test_size);
        }
        
        double accuracy;
        for (epoch = 0; epoch != iters; epoch++)
        {
            std::random_shuffle(iter_p, iter_r);
            
            Train(iter_p, iter_r);
            //printf("\n");
            accuracy = Test(iter_r, iter_q);
            //printf("\n");
            printf("%cEpoch: %d Test-Accuracy: %lf                        \n", 13, epoch + 1, accuracy);
        }
    }
    if (mode == 1)
    {
        std::random_shuffle(data_set.begin(), data_set.end());
        
        train_size = data_size * 0.7;
        test_size = data_size - train_size;
        
        iter_p = data_set.begin();
        iter_q = data_set.end();
        iter_r = data_set.begin() + train_size;
        
        if (debug_mode > 0) {
            printf("Label size: %d\n", label_size);
            printf("Data  size: %d\n", data_size);
            printf("Train size: %d\n", train_size);
            printf("Test  size: %d\n", test_size);
        }
        
        double accuracy;
        for (epoch = 0; epoch != iters; epoch++)
        {
            std::random_shuffle(iter_p, iter_r);
            
            Train(iter_p, iter_r);
            //printf("\n");
            accuracy = Test(iter_r, iter_q);
            //printf("\n");
            printf("%cEpoch: %d Test-Accuracy: %lf                        \n", 13, epoch + 1, accuracy);
        }
        
        SaveModel();
    }
    if (mode == 2)
    {
        LoadModel();
        
        Test(data_set.begin(), data_set.end());
        
        printf("\n");
    }
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
    if ((i = ArgPos((char *)"-vector", argc, argv)) > 0) strcpy(vector_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-model", argc, argv)) > 0) strcpy(model_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-mode", argc, argv)) > 0) mode = atoi(argv[i + 1]);
    TrainModel();
    return 0;
}