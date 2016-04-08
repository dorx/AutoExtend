#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>

typedef float real;
typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

real random(real min, real max);

class Neuron
{
protected:
	real *_ac, *_er;
	int _numCols, _numRows, _numElements, _maxElements;
public:
	Eigen::Map<MatrixType> ac, er;

	Neuron() : ac(NULL, 0, 0), er(NULL, 0, 0)
	{
		_numCols = 0; _numRows = 0; _numElements = 0; _maxElements = 0;
		_ac = NULL; _er = NULL;
	}

	~Neuron();

	void alloc(int m, int n = 1);
	void destroy();

	void resize(int m, int n = 1);
	void flush();

	real* get_ac();
	real* get_er();

	int rows();
	int cols();
	int eles();

	void output_ac();
	void output_er();
};

class Layer_synapse
{
protected:
	MatrixType _grad, _hist;
	int _cnt_grad;
	real _init_lr, _mmt, _wd;

	int _dim1, _dim2;
public:
	MatrixType _para;
	
	Layer_synapse();
	~Layer_synapse();

	void alloc(int dim1, int dim2);
	void destroy();
	void flush();
	void flush_para();
	void flush_grad();
	void flush_hist();

	void init();

	void set_init_lr(real lr);
	real get_init_lr();
	void set_mmt(real mmt);
	real get_mmt();
	void set_wd(real wd);
	real get_wd();
	void set_cnt(int cnt);
	int get_cnt();

	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
	void c_grad(Neuron &neu1, Neuron &neu2);
	void update(real lr);
	void update_adagrad();
};

class Layer_conv
{
protected:
	MatrixType _grad, _hist;
	int _cnt_grad;
	real _init_lr, _mmt, _wd;

	int _dim1, _dim2, _win;
public:
	MatrixType _para;
	
	Layer_conv();
	~Layer_conv();

	void alloc(int dim1, int dim2, int win);
	void destroy();
	void flush();
	void flush_para();
	void flush_grad();
	void flush_hist();

	void init();

	void set_init_lr(real lr);
	real get_init_lr();
	void set_mmt(real mmt);
	real get_mmt();
	void set_wd(real wd);
	real get_wd();
	void set_cnt(int cnt);
	int get_cnt();

	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
	void c_grad(Neuron &neu1, Neuron &neu2);
	void update(real lr);
	void update_adagrad();
};

class Layer_pooling_max
{
protected:
	int _dim, _maxElements;
	int *_pst;
public:
	Layer_pooling_max();
	~Layer_pooling_max();

	void alloc(int dim);
	void resize(int dim);
	void destroy();
	void flush();

	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
};

class Layer_pooling_kmax
{
protected:
	int _dim, _amax, _kmax, _maxElements;
	int *_pst;
public:
	Layer_pooling_kmax();
	~Layer_pooling_kmax();

	void alloc(int dim, int kmax = 1);
	void resize(int dim, int kmax = 1);
	void destroy();
	void flush();

	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
};

class Layer_dropout
{
protected:
	real _dim, _drop_rate;
	int *_mask;
	std::string _mode;
public:
	Layer_dropout();
	~Layer_dropout();

	void alloc(int dim);
	void destroy();
	void set_mode(std::string mode);
	void set_droprate(real deop_rate);

	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
};


class Layer_sigmoid
{
public:
	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
};

class Layer_tanh
{
public:
	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
};

class Layer_hardtanh
{
public:
	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
};

class Layer_rect
{
public:
	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
};

class Layer_softsign
{
public:
	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
};

class Layer_loss_softmax
{
protected:
	int _dim;
	real *_tar;
public:
	Layer_loss_softmax();
	~Layer_loss_softmax();
	void alloc(int dim);
	void destroy();
	void set_tar(int *tar);
	void set_label(int lb);
	void f_prop(Neuron &neu1, Neuron &neu2);
	void b_prop(Neuron &neu1, Neuron &neu2);
};

class Layer_concat
{
public:
	void f_prop(std::vector<Neuron *> neuvec, Neuron &neu2);
	void b_prop(std::vector<Neuron *> neuvec, Neuron &neu2);
};