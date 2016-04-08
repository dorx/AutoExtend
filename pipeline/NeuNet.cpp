#include "NeuNet.h"

real random(real min, real max)
{
	return rand() / (double)RAND_MAX*(max - min) + min;
}

/*************************************
neuron
*************************************/

Neuron::~Neuron()
{
	if (_ac != NULL) free(_ac);
	if (_er != NULL) free(_er);
	_numCols = 0; _numRows = 0; _numElements = 0; _maxElements = 0;
	_ac = NULL; _er = NULL;
	new (&ac) Eigen::Map<MatrixType>(NULL, 0, 0);
	new (&er) Eigen::Map<MatrixType>(NULL, 0, 0);
}

void Neuron::alloc(int m, int n)
{
	_maxElements = m * n;
	if (_ac != NULL) free(_ac);
	if (_er != NULL) free(_er);
	_ac = (real *)malloc(_maxElements * sizeof(real));
	_er = (real *)malloc(_maxElements * sizeof(real));
}

void Neuron::destroy()
{
	if (_ac != NULL) free(_ac);
	if (_er != NULL) free(_er);
	_numCols = 0; _numRows = 0; _numElements = 0; _maxElements = 0;
	_ac = NULL; _er = NULL;
	new (&ac) Eigen::Map<MatrixType>(NULL, 0, 0);
	new (&er) Eigen::Map<MatrixType>(NULL, 0, 0);
}

void Neuron::resize(int m, int n)
{
	_numElements = m * n;
	_numRows = m;
	_numCols = n;
	if (_numElements > _maxElements)
	{
		_maxElements = _numElements;
		if (_ac != NULL) free(_ac);
		if (_er != NULL) free(_er);
		_ac = (real *)calloc(_maxElements, sizeof(real));
		_er = (real *)calloc(_maxElements, sizeof(real));
	}
	new (&ac) Eigen::Map<MatrixType>(_ac, m, n);
	new (&er) Eigen::Map<MatrixType>(_er, m, n);
}

void Neuron::flush()
{
	for (int k = 0; k != _maxElements; k++)
	{
		_ac[k] = 0;
		_er[k] = 0;
	}
}

real* Neuron::get_ac()
{
	return _ac;
}

real* Neuron::get_er()
{
	return _er;
}

int Neuron::rows()
{
	return _numRows;
}

int Neuron::cols()
{
	return _numCols;
}

int Neuron::eles()
{
	return _numElements;
}

void Neuron::output_ac()
{
	Eigen::Map<MatrixType> x(_ac, _numRows, _numCols);
}

void Neuron::output_er()
{
	Eigen::Map<MatrixType> x(_er, _numRows, _numCols);
}

/*************************************
synapse layer
*************************************/

Layer_synapse::Layer_synapse()
{
	_cnt_grad = 0;
	_init_lr = 0;
	_mmt = 0;
	_wd = 0;
	_dim1 = 0;
	_dim2 = 0;
}

Layer_synapse::~Layer_synapse()
{
	_cnt_grad = 0;
	_init_lr = 0;
	_mmt = 0;
	_wd = 0;
	_dim1 = 0;
	_dim2 = 0;
}

void Layer_synapse::alloc(int dim1, int dim2)
{
	_para.resize(dim1, dim2);
	_grad.resize(dim1, dim2);
	_hist.resize(dim1, dim2);
	_para.setZero();
	_grad.setZero();
	_hist.setConstant(0.0001);
	_dim1 = dim1;
	_dim2 = dim2;
	_cnt_grad = 0;
	_init_lr = 0;
	_mmt = 0;
	_wd = 0;
}

void Layer_synapse::destroy()
{
	_para.resize(0, 0);
	_grad.resize(0, 0);
	_hist.resize(0, 0);
	_dim1 = 0;
	_dim2 = 0;
	_cnt_grad = 0;
	_init_lr = 0;
	_mmt = 0;
	_wd = 0;
}

void Layer_synapse::flush()
{
	_para.setZero();
	_grad.setZero();
	_hist.setConstant(0.0001);
}

void Layer_synapse::flush_para()
{
	_para.setZero();
}

void Layer_synapse::flush_grad()
{
	_grad.setZero();
}

void Layer_synapse::flush_hist()
{
	_hist.setConstant(0.0001);
}

void Layer_synapse::init()
{
	real r = sqrt(6.0 / (_dim1 + _dim2));
	for (int i = 0; i != _dim1; i++) for (int j = 0; j != _dim2; j++)
		_para(i, j) = random(-r, r);
}

void Layer_synapse::set_init_lr(real lr)
{
	_init_lr = lr;
}

real Layer_synapse::get_init_lr()
{
	return _init_lr;
}

void Layer_synapse::set_mmt(real mmt)
{
	_mmt = mmt;
}

real Layer_synapse::get_mmt()
{
	return _mmt;
}

void Layer_synapse::set_wd(real wd)
{
	_wd = wd;
}

real Layer_synapse::get_wd()
{
	return _wd;
}

void Layer_synapse::set_cnt(int cnt)
{
	_cnt_grad = cnt;
}

int Layer_synapse::get_cnt()
{
	return _cnt_grad;
}

void Layer_synapse::f_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.eles() != _dim1)
	{
		printf("Layer_synapse: f_prop error!\n");
		exit(1);
	}
	neu2.resize(_dim2);
	Eigen::Map<MatrixType> x(neu1.get_ac(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_ac(), neu2.eles(), 1);
	y.noalias() += _para.transpose() * x;
}

void Layer_synapse::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.eles() != _dim1 || neu2.eles() != _dim2)
	{
		printf("Layer_synapse: b_prop error!\n");
		exit(1);
	}
	Eigen::Map<MatrixType> x(neu1.get_er(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_er(), neu2.eles(), 1);
	x.noalias() += _para * y;
}

void Layer_synapse::c_grad(Neuron &neu1, Neuron &neu2)
{
	if (neu1.eles() != _dim1 || neu2.eles() != _dim2)
	{
		printf("Layer_synapse: c_grad error!\n");
		exit(1);
	}
	Eigen::Map<MatrixType> x(neu1.get_ac(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_er(), neu2.eles(), 1);
	_grad.noalias() += x * y.transpose();
	_cnt_grad++;
}

void Layer_synapse::update(real lr)
{
	_grad /= _cnt_grad;
	_hist = _hist * _mmt + _grad * lr;
	_para = (1 - _wd) * _para + _hist;
	_grad.setZero();
	_cnt_grad = 0;
}

void Layer_synapse::update_adagrad()
{
	_grad /= _cnt_grad;
	_hist.array() += _grad.array() * _grad.array();
	_para.array() = (1 - _wd) * _para.array() + _init_lr * _grad.array() / _hist.array().sqrt();
	_grad.setZero();
	_cnt_grad = 0;
}

/*************************************
conv layer
*************************************/

Layer_conv::Layer_conv()
{
	_cnt_grad = 0;
	_init_lr = 0;
	_mmt = 0;
	_dim1 = 0;
	_dim2 = 0;
	_win = 0;
	_wd = 0;
}

Layer_conv::~Layer_conv()
{
	_cnt_grad = 0;
	_init_lr = 0;
	_mmt = 0;
	_dim1 = 0;
	_dim2 = 0;
	_win = 0;
	_wd = 0;
}

void Layer_conv::alloc(int dim1, int dim2, int win)
{
	_para.resize(dim1 * win, dim2);
	_grad.resize(dim1 * win, dim2);
	_hist.resize(dim1 * win, dim2);
	_para.setZero();
	_grad.setZero();
    _hist.setConstant(0.0001);
	_dim1 = dim1;
	_dim2 = dim2;
	_win = win;
	_cnt_grad = 0;
	_init_lr = 0;
	_mmt = 0;
	_wd = 0;
}

void Layer_conv::destroy()
{
	_para.resize(0, 0);
	_grad.resize(0, 0);
	_hist.resize(0, 0);
	_dim1 = 0;
	_dim2 = 0;
	_win = 0;
	_cnt_grad = 0;
	_init_lr = 0;
	_mmt = 0;
	_wd = 0;
}

void Layer_conv::flush()
{
	_para.setZero();
	_grad.setZero();
	_hist.setConstant(0.0001);
}

void Layer_conv::flush_para()
{
	_para.setZero();
}

void Layer_conv::flush_grad()
{
	_grad.setZero();
}

void Layer_conv::flush_hist()
{
	_hist.setConstant(0.0001);
}

void Layer_conv::init()
{
	real r = sqrt(6.0 / (_dim1 * _win + _dim2));
	for (int i = 0; i != _dim1 * _win; i++) for (int j = 0; j != _dim2; j++)
		_para(i, j) = random(-r, r);
}

void Layer_conv::set_init_lr(real lr)
{
	_init_lr = lr;
}

real Layer_conv::get_init_lr()
{
	return _init_lr;
}

void Layer_conv::set_mmt(real mmt)
{
	_mmt = mmt;
}

real Layer_conv::get_mmt()
{
	return _mmt;
}

void Layer_conv::set_wd(real wd)
{
	_wd = wd;
}

real Layer_conv::get_wd()
{
	return _wd;
}

void Layer_conv::set_cnt(int cnt)
{
	_cnt_grad = cnt;
}

int Layer_conv::get_cnt()
{
	return _cnt_grad;
}

void Layer_conv::f_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.rows() != _dim1)
	{
		printf("Layer_conv: f_prop error!\n");
		exit(1);
	}
	neu2.resize(_dim2, neu1.cols() + _win - 1);
	int dim = neu1.rows();
	int dn = neu1.cols(), dk = _win, dx = dn + 2 * dk - 2;
	real *f = (real *)calloc(dx * dim, sizeof(real));
	memcpy(f + dim * (dk - 1), neu1.get_ac(), dn * dim * sizeof(real));
	Eigen::Map<MatrixType> x(NULL, 0, 0);
	Eigen::Map<MatrixType> y(NULL, 0, 0);
	for (int i = 0; i != dn + dk - 1; i++)
	{
		new (&x) Eigen::Map<MatrixType>(f + i * dim, dim * dk, 1);
		new (&y) Eigen::Map<MatrixType>(neu2.get_ac() + i * _dim2, _dim2, 1);
		y.noalias() += _para.transpose() * x;
	}
	free(f);
}

void Layer_conv::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.rows() != _dim1 || neu2.rows() != _dim2)
	{
		printf("Layer_conv: b_prop error!\n");
		exit(1);
	}
	int dim = neu1.rows();
	int dn = neu1.cols(), dk = _win, dx = dn + 2 * dk - 2;
	real *f = (real *)calloc(dx * dim, sizeof(real) * sizeof(real));
	memcpy(f + dim * (dk - 1), neu1.get_er(), dn * dim);
	Eigen::Map<MatrixType> x(NULL, 0, 0);
	Eigen::Map<MatrixType> y(NULL, 0, 0);
	for (int i = 0; i != dn + dk - 1; i++)
	{
		new (&x) Eigen::Map<MatrixType>(f + i * dim, dim * dk, 1);
		new (&y) Eigen::Map<MatrixType>(neu2.get_er() + i * _dim2, _dim2, 1);
		x.noalias() += _para * y;
	}
	memcpy(neu1.get_er(), f + dim * (dk - 1), dn * dim * sizeof(real));
	free(f);
}

void Layer_conv::c_grad(Neuron &neu1, Neuron &neu2)
{
	if (neu1.rows() != _dim1 || neu2.rows() != _dim2)
	{
		printf("Layer_conv: c_grad error!\n");
		exit(1);
	}
	int dim = neu1.rows();
	int dn = neu1.ac.cols(), dk = _win, dx = dn + 2 * dk - 2;
	real *f = (real *)calloc(dx * dim, sizeof(real));
	memcpy(f + dim * (dk - 1), neu1.get_ac(), dn * dim * sizeof(real));
	Eigen::Map<MatrixType> x(NULL, 0, 0);
	Eigen::Map<MatrixType> y(NULL, 0, 0);
	for (int i = 0; i != dn + dk - 1; i++)
	{
		new (&x) Eigen::Map<MatrixType>(f + i * dim, dim * dk, 1);
		new (&y) Eigen::Map<MatrixType>(neu2.get_er() + i * _dim2, _dim2, 1);
		_grad.noalias() += x * y.transpose();
	}
	_cnt_grad++;
	free(f);
}

void Layer_conv::update(real lr)
{
	_grad /= _cnt_grad;
	_hist = _hist * _mmt + _grad * lr;
	_para = (1 - _wd) * _para + _hist;
	_grad.setZero();
	_cnt_grad = 0;
}

void Layer_conv::update_adagrad()
{
	_grad /= _cnt_grad;
	_hist.array() += _grad.array() * _grad.array();
	_para.array() = (1 - _wd) * _para.array() + _init_lr * _grad.array() / _hist.array().sqrt();
	_grad.setZero();
	_cnt_grad = 0;
}

/*************************************
sigmoid layer
*************************************/
void Layer_sigmoid::f_prop(Neuron &neu1, Neuron &neu2)
{
	neu2.resize(neu1.rows(), neu1.cols());
	Eigen::Map<MatrixType> x(neu1.get_ac(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_ac(), neu2.eles(), 1);
	y.array() = 1 / ((-x.array()).exp() + 1);
}

void Layer_sigmoid::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.eles() != neu2.eles())
	{
		printf("Layer_sigmoid: b_prop error!\n");
		exit(1);
	}
	Eigen::Map<MatrixType> x(neu1.get_er(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_er(), neu2.eles(), 1);
	Eigen::Map<MatrixType> z(neu2.get_ac(), neu2.eles(), 1);
	x.array() = y.array() * z.array() * (1 - z.array());
}

/*************************************
tanh layer
*************************************/
void Layer_tanh::f_prop(Neuron &neu1, Neuron &neu2)
{
	neu2.resize(neu1.rows(), neu1.cols());
	Eigen::Map<MatrixType> x(neu1.get_ac(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_ac(), neu2.eles(), 1);
	y.array() = 2 / ((-2 * x.array()).exp() + 1) - 1;
}

void Layer_tanh::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.eles() != neu2.eles())
	{
		printf("Layer_sigmoid: b_prop error!\n");
		exit(1);
	}
	Eigen::Map<MatrixType> x(neu1.get_er(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_er(), neu2.eles(), 1);
	Eigen::Map<MatrixType> z(neu2.get_ac(), neu2.eles(), 1);
	x.array() = y.array() * (1 + z.array()) * (1 - z.array());
}

/*************************************
Layer_hardtanh
*************************************/
void Layer_hardtanh::f_prop(Neuron &neu1, Neuron &neu2)
{
	neu2.resize(neu1.rows(), neu1.cols());
	Eigen::Map<MatrixType> x(neu1.get_ac(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_ac(), neu2.eles(), 1);
	y.array() = x.array().max(-1).min(1);
}

void Layer_hardtanh::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.eles() != neu2.eles())
	{
		printf("Layer_sigmoid: b_prop error!\n");
		exit(1);
	}
	Eigen::Map<MatrixType> x(neu1.get_er(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_er(), neu2.eles(), 1);
	Eigen::Map<MatrixType> z(neu2.get_ac(), neu2.eles(), 1);
	for (int k = 0; k != neu1.eles(); k++)
		x(k) = y(k) * (z(k) > -1 && z(k) < 1);
}

/*************************************
rect layer
*************************************/
void Layer_rect::f_prop(Neuron &neu1, Neuron &neu2)
{
	neu2.resize(neu1.rows(), neu1.cols());
	Eigen::Map<MatrixType> x(neu1.get_ac(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_ac(), neu2.eles(), 1);
	y.array() = x.array().max(0);
}

void Layer_rect::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.eles() != neu2.eles())
	{
		printf("Layer_sigmoid: b_prop error!\n");
		exit(1);
	}
	Eigen::Map<MatrixType> x(neu1.get_er(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_er(), neu2.eles(), 1);
	Eigen::Map<MatrixType> z(neu2.get_ac(), neu2.eles(), 1);
	for (int k = 0; k != neu1.eles(); k++)
		x(k) = y(k) * (z(k) > 0);
}

/*************************************
softsign layer
*************************************/
void Layer_softsign::f_prop(Neuron &neu1, Neuron &neu2)
{
	neu2.resize(neu1.rows(), neu1.cols());
	Eigen::Map<MatrixType> x(neu1.get_ac(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_ac(), neu2.eles(), 1);
	y.array() /= (1 + x.array().abs());
}

void Layer_softsign::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.eles() != neu2.eles())
	{
		printf("Layer_sigmoid: b_prop error!\n");
		exit(1);
	}
	Eigen::Map<MatrixType> x(neu1.get_er(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_er(), neu2.eles(), 1);
	Eigen::Map<MatrixType> z(neu2.get_ac(), neu2.eles(), 1);
	x.array() = y.array() * (1 - z.array().abs()).pow(2);
}

/*************************************
max pooling layer
*************************************/

Layer_pooling_max::Layer_pooling_max()
{
	_dim = 0;
	_maxElements = 0;
	_pst = NULL;
}

Layer_pooling_max::~Layer_pooling_max()
{
	_dim = 0;
	_maxElements = 0;
	if (_pst != NULL) free(_pst);
	_pst = NULL;
}

void Layer_pooling_max::alloc(int dim)
{
	if (_pst != NULL) free(_pst);
	_maxElements = dim;
	_pst = (int *)calloc(_maxElements, sizeof(int));
}

void Layer_pooling_max::resize(int dim)
{
	if (dim > _maxElements)
	{
		printf("Layer_pooling_max: resize error!\n");
		exit(1);
	}
	_dim = dim;
}

void Layer_pooling_max::destroy()
{
	_dim = 0;
	_maxElements = 0;
	if (_pst != NULL) free(_pst);
	_pst = NULL;
}

void Layer_pooling_max::flush()
{
	memset(_pst, 0, _maxElements * sizeof(int));
}

void Layer_pooling_max::f_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.rows() != _dim)
	{
		printf("Layer_pooling_max: f_prop error!\n");
		exit(1);
	}
	neu2.resize(_dim, 1);
	int m = neu1.rows(), n = neu1.cols();
	for (int k = 0; k != m; k++)
	{
		int max_index = -1;
		real max_value = -RAND_MAX;
		for (int i = 0; i != n; i++) if (neu1.ac(k, i) > max_value)
		{
			max_value = neu1.ac(k, i);
			max_index = i;
		}
		_pst[k] = max_index;
		neu2.ac(k, 0) = max_value;
	}
}

void Layer_pooling_max::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.rows() != _dim)
	{
		printf("Layer_pooling_max: b_prop error!\n");
		exit(1);
	}
	for (int k = 0; k != _dim; k++) neu1.er(k, _pst[k]) = neu2.er(k, 0);
}

/*************************************
kmax pooling layer
*************************************/

Layer_pooling_kmax::Layer_pooling_kmax()
{
	_dim = 0;
	_amax = 0;
	_kmax = 0;
	_maxElements = 0;
	_pst = NULL;
}

Layer_pooling_kmax::~Layer_pooling_kmax()
{
	_dim = 0;
	_amax = 0;
	_kmax = 0;
	_maxElements = 0;
	if (_pst != NULL) free(_pst);
	_pst = NULL;
}

void Layer_pooling_kmax::alloc(int dim, int kmax)
{
	if (_pst != NULL) free(_pst);
	_maxElements = dim * kmax;
	_pst = (int *)calloc(_maxElements, sizeof(int));
}

void Layer_pooling_kmax::resize(int dim, int kmax)
{
	if (dim * kmax > _maxElements)
	{
		printf("Layer_pooling_max: resize error!\n");
		exit(1);
	}
	_dim = dim;
	_amax = 0;
	_kmax = kmax;
}

void Layer_pooling_kmax::destroy()
{
	_dim = 0;
	_amax = 0;
	_kmax = 0;
	_maxElements = 0;
	if (_pst != NULL) free(_pst);
	_pst = NULL;
}

void Layer_pooling_kmax::flush()
{
	memset(_pst, 0, _maxElements * sizeof(int));
}

void Layer_pooling_kmax::f_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.rows() != _dim)
	{
		printf("Layer_pooling_max: f_prop error!\n");
		exit(1);
	}
	neu2.resize(neu1.rows(), _kmax);
	int dim = neu1.rows(), dn = neu1.cols();
	if (_kmax > dn) _amax = dn;
	else _amax = _kmax;
	real *val = (real *)malloc(dn * sizeof(real));
	int *pst = (int *)malloc(dn * sizeof(int));
	for (int d = 0; d < dim; d++)
	{
		for (int i = 0; i != dn; i++) { val[i] = neu1.ac(d, i); pst[i] = i; }
		real treal; int tint;
		for (int i = 0; i != dn; i++) for (int j = i; j != dn; j++) if (val[i]<val[j])
		{
			treal = val[i]; val[i] = val[j]; val[j] = treal;
			tint = pst[i]; pst[i] = pst[j]; pst[j] = tint;
		}
		for (int i = 0; i != _amax; i++) for (int j = i; j != _amax; j++) if (pst[i]>pst[j])
		{
			tint = pst[i]; pst[i] = pst[j]; pst[j] = tint;
		}
		for (int i = 0; i != _amax; i++)
		{
			_pst[i * dim + d] = pst[i];
			neu2.ac(d, i) = neu1.ac(d, pst[i]);
		}
	}
	free(val);
	free(pst);
}

void Layer_pooling_kmax::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.rows() != _dim)
	{
		printf("Layer_pooling_max: b_prop error!\n");
		exit(1);
	}
	for (int d = 0; d != _dim; d++) for (int i = 0; i != _amax; i++)
		neu1.er(d, _pst[i * _dim + d]) = neu2.er(d, i);
}

/*************************************
softmax loss layer
*************************************/

Layer_loss_softmax::Layer_loss_softmax()
{
	_dim = 0;
	_tar = NULL;
}

Layer_loss_softmax::~Layer_loss_softmax()
{
	if (_tar != NULL) free(_tar);
	_dim = 0;
	_tar = NULL;
}

void Layer_loss_softmax::alloc(int dim)
{
	_dim = dim;
	_tar = (real *)calloc(dim, sizeof(real));
}

void Layer_loss_softmax::destroy()
{
	if (_tar != NULL) free(_tar);
	_dim = 0;
	_tar = NULL;
}

void Layer_loss_softmax::set_tar(int *tar)
{
	for (int k = 0; k != _dim; k++) _tar[k] = tar[k];
}

void Layer_loss_softmax::set_label(int lb)
{
	if (lb >= _dim)
	{
		printf("Layer_loss_softmax: set_label error!\n");
		exit(1);
	}
	for (int k = 0; k != _dim; k++)
	{
		if (k == lb) _tar[k] = 1;
		else _tar[k] = 0;
	}
}

void Layer_loss_softmax::f_prop(Neuron &neu1, Neuron &neu2)
{
	neu2.resize(neu1.rows(), neu1.cols());
	neu2.ac.array() = neu1.ac.array().exp();
	real sum = neu2.ac.sum();
	neu2.ac.array() /= sum;
}

void Layer_loss_softmax::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (neu1.eles() != _dim || neu2.eles() != _dim)
	{
		printf("Layer_loss_softmax: b_prop error!\n");
		exit(1);
	}
	for (int k = 0; k != _dim; k++)
		neu1.er(k) = _tar[k] - neu2.ac(k);
}

/*************************************
concat loss layer
*************************************/

void Layer_concat::f_prop(std::vector<Neuron *> neuvec, Neuron &neu2)
{
	int sum_eles = 0, length = neuvec.size();
	for (int k = 0; k != length; k++) sum_eles += neuvec[k]->eles();
	neu2.resize(sum_eles);
	sum_eles = 0;
	for (int k = 0; k != length; k++)
	{
		int len = neuvec[k]->eles();
		for (int i = 0; i != len; i++)
			neu2.get_ac()[sum_eles + i] = neuvec[k]->get_ac()[i];
		sum_eles += len;
	}
}

void Layer_concat::b_prop(std::vector<Neuron *> neuvec, Neuron &neu2)
{
	int sum_eles = 0, length = neuvec.size();
	for (int k = 0; k != length; k++) sum_eles += neuvec[k]->eles();
	if (sum_eles != neu2.eles())
	{
		printf("Layer_concat: b_prop error!\n");
		exit(1);
	}
	sum_eles = 0;
	for (int k = 0; k != length; k++)
	{
		int len = neuvec[k]->eles();
		for (int i = 0; i != len; i++)
			neuvec[k]->get_er()[i] = neu2.get_er()[sum_eles + i];
		sum_eles += len;
	}
}

/*************************************
drop out layer
*************************************/

Layer_dropout::Layer_dropout()
{
	_dim = 0;
	_drop_rate = 0;
	_mode.clear();
	_mask = NULL;
}

Layer_dropout::~Layer_dropout()
{
	if (_mask != NULL) free(_mask);
	_dim = 0;
	_drop_rate = 0;
	_mode.clear();
	_mask = NULL;
}

void Layer_dropout::alloc(int dim)
{
	_dim = dim;
	_mask = (int *)malloc(dim * sizeof(int));
	_drop_rate = 0;
	_mode.clear();
}

void Layer_dropout::destroy()
{
	if (_mask != NULL) free(_mask);
	_dim = 0;
	_drop_rate = 0;
	_mode.clear();
	_mask = NULL;
}

void Layer_dropout::set_mode(std::string mode)
{
	_mode = mode;
}

void Layer_dropout::set_droprate(real drop_rate)
{
	_drop_rate = drop_rate;
}

void Layer_dropout::f_prop(Neuron &neu1, Neuron &neu2)
{
	if (_dim != neu1.eles())
	{
		printf("Layer_dropout: f_prop error!\n");
		exit(1);
	}
	neu2.resize(neu1.rows(), neu1.cols());
	Eigen::Map<MatrixType> x(neu1.get_ac(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_ac(), neu2.eles(), 1);
	if (_mode == "test")
	{
		y = x * (1 - _drop_rate);
	}
	if (_mode == "train")
	{
		for (int k = 0; k != _dim; k++)
		{
			real rand_value = random(0, 1);
			if (rand_value >= _drop_rate)
			{
				_mask[k] = 1;
				y(k) = x(k);
			}
			else
			{
				_mask[k] = 0;
				y(k) = 0;
			}
		}
	}
}

void Layer_dropout::b_prop(Neuron &neu1, Neuron &neu2)
{
	if (_dim != neu1.eles() || _dim != neu2.eles())
	{
		printf("Layer_dropout: b_prop error!\n");
		exit(1);
	}
	Eigen::Map<MatrixType> x(neu1.get_er(), neu1.eles(), 1);
	Eigen::Map<MatrixType> y(neu2.get_er(), neu2.eles(), 1);
	for (int k = 0; k != _dim; k++)
	{
		if (_mask[k] == 1)
			x(k) = y(k);
		else
			x(k) = 0;
	}
}
