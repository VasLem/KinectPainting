import warnings
import sys
import logging
import numpy as np
from numpy import array, argmax, argmin, concatenate, diag, isclose
from numpy import dot, sign, zeros, zeros_like, random, trace, mean
from numpy import allclose
from numpy.linalg import inv, pinv, matrix_rank, qr, lstsq
from sklearn.decomposition import DictionaryLearning, SparseCoder
from numpy import sum as npsum
from numpy import abs as npabs
import random as rand
import cPickle as pickle
from scipy.optimize import minimize
import time
import class_objects as co

LOG = logging.getLogger('__name__')
SPLOG = logging.getLogger('SparseLogger')
FH = logging.FileHandler('sparse_coding.log',mode='w')
FH.setFormatter(logging.Formatter(
    '%(asctime)s (%(lineno)s): %(message)s',
    "%Y-%m-%d %H:%M:%S"))
SPLOG.addHandler(FH)
SPLOG.setLevel(logging.DEBUG)
if __name__=='__main__':
    CH = logging.StreamHandler(sys.stderr)
    CH.setFormatter(logging.Formatter(
        '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
    LOG.addHandler(CH)
    LOG.setLevel(logging.INFO)

def timeit(func):
    '''
    Decorator to time extraction
    '''
    def wrapper(self,*arg, **kw):
        t1 = time.time()
        res = func(self,*arg, **kw)
        t2 = time.time()
        self.time.append(t2-t1)
        del self.time[:-5000]
        return res
    return wrapper

class SparseCoding(object):

    def __init__(self, log_lev='INFO', sparse_dim_rat=None, name='',
                 dist_beta=0.1, dist_sigma=0.005, display=0):
        LOG.setLevel(log_lev)

        self.name = name
        self.codebook_comps = None
        self.active_set = None
        self.min_coeff = max([1,
            co.CONST['sparse_fss_min_coeff']])
        self.min_coeff_rat = co.CONST['sparse_fss_min_coeff_rat']
        self.gamma = co.CONST['sparse_fss_gamma']
        self.rat = None
        if isinstance(self.gamma, str):
            if self.gamma.starts_with('var'):
                try:
                    self.rat = [float(s) for s in str.split() if
                                co.type_conv.isfloat(s)][0]
                except IndexError:
                    self.rat = None
        self.inp_features = None
        self.sparse_features = None
        self.basis_constraint = 1
        self.inv_codebook_comps = None
        self.res_codebook_comps = None
        self.max_iter = 500
        self.dict_max_iter = 300
        self.display = display
        self.prev_err = 0
        self.curr_error = 0
        self.allow_big_vals = False
        self.sparse_dim_rat = sparse_dim_rat
        if sparse_dim_rat is None:
            self.sparse_dim_rat = co.CONST['sparse_dim_rat']
        self.theta = None
        self.prev_sparse_feats = None
        self.flush_flag = False
        self.sparse_feat_list = None
        self.inp_feat_list = None
        self.codebook = None
        self.time = []

    def flush_variables(self):
        '''
        Empty variables
        '''
        self.active_set = None
        self.theta = None
        self.codebook_comps = None
        self.inp_features = None
        self.inp_feat_list = None
        self.sparse_features = None
        self.flush_flag = True
        self.res_codebook_comps = None
        self.prev_err = 0
        self.curr_error = 0
        self.lbds = 0.5*np.ones(self.sparse_dim)

    def initialize(self, feat_dim,
                   init_codebook_comps=None):
        '''
        Initialises B dictionary and s
        '''
        self.sparse_dim = self.sparse_dim_rat * feat_dim
        if init_codebook_comps is not None:
            if (init_codebook_comps.shape[0] == feat_dim and
                    init_codebook_comps.shape[1] == self.sparse_dim_rat *
                feat_dim):
                self.codebook_comps = init_codebook_comps.copy()
            else:
                raise Exception('Wrong input of initial B matrix, the dimensions' +
                                ' should be ' + str(feat_dim) + 'x' +
                                str(self.sparse_dim) + ', not ' +
                                str(init_codebook_comps.shape[0]) + 'x' +
                                str(init_codebook_comps.shape[1]))
        if (self.codebook_comps is None) or self.flush_flag:
            LOG.warning('Non existent codebook, manufactuning a random one')
            self.codebook_comps = random.random((feat_dim, self.sparse_dim))
        if (self.sparse_features is None) or self.flush_flag:
            self.sparse_features = zeros((self.sparse_dim, 1))
        self.theta = zeros(self.sparse_dim)
        self.active_set = zeros((self.sparse_dim), bool)
        self.sparse_features = zeros((self.sparse_dim, 1))
        self.flush_flag = False
        self.is_trained = False

    def object_val_calc(self, codebook_comps, ksi, gamma, theta, vecs):
        '''
        Calculate objective function value
        '''
        _bs_ = np.dot(codebook_comps, vecs)
        square_term = 0.5 * npsum((ksi - _bs_)**2, axis=0)
        res = (square_term + gamma * dot(theta.T, vecs)).ravel()
        return res

    def feature_sign_search_algorithm(self,
                                      inp_features,
                                      acondtol=1e-3,
                                      ret_error=False,
                                      display_error=False,
                                      max_iter=0,
                                      single=False, timed=True,
                                      starting_points=None,
                                      training=False):
        '''
        Returns sparse features representation
        '''
        self.min_coeff_rat = co.CONST['sparse_fss_min_coeff_rat']
        self.min_coeff = max([self.min_coeff,
                              self.min_coeff_rat *
                              np.size(inp_features)])
        if self.inp_feat_list is not None:
            self.inp_feat_list.append(inp_features.ravel())
        else:
            self.inp_feat_list = [inp_features.ravel()]
        self.inp_features = inp_features.copy().reshape((-1,1))
        # Step 1
        btb = dot(self.codebook_comps.T, self.codebook_comps)
        btf = dot(self.codebook_comps.T, self.inp_features)
        if self.rat is not None:
            self.gamma = np.max(np.abs(-2 * btf)) * self.rat

        gamma = self.gamma
        if starting_points is not None:
            self.sparse_features = starting_points.reshape((self.sparse_dim,
                                                            1))
            self.theta = np.sign(self.sparse_features)
            self.active_set[:] = False
            self.active_set[self.sparse_features.ravel()!=0] = True
            step2 = 0
        else:
            step2 = 1
        count = 0
        prev_objval = 0
        if max_iter == 0:
            max_iter = self.max_iter
        else:
            self.max_iter = max_iter
        self.prev_sparse_feats = None
        prev_error = 0
        initial_energy = compute_lineq_error(inp_features, 0,
                                                              0)
        interm_error = initial_energy
        SPLOG.info('Initial Signal Energy: ' + str(initial_energy))
        SPLOG.info('Initial nonzero elements number: ' +
                  str(np.sum(inp_features!=0)))
        converged = False
        for count in range(self.max_iter):
            # Step 2    
            if step2:
                zero_coeffs = (self.sparse_features == 0)
                qp_der_outfeati = 2 * \
                    (dot(btb, self.sparse_features)
                     - btf) * zero_coeffs.reshape((-1,1))
                i = argmax(npabs(qp_der_outfeati))
                if (npabs(qp_der_outfeati[i]) > gamma
                    or npsum(self.active_set) < self.min_coeff):
                    self.theta[i] = -sign(qp_der_outfeati[i])
                    self.active_set[i] = True
            # Step 3
            codebook_comps_h = self.codebook_comps[:, self.active_set]
            sparse_feat_h = self.sparse_features[self.active_set].reshape(
                (-1,1))
            theta_h = self.theta[self.active_set].reshape((-1,1))
            _q_ = dot(codebook_comps_h.T, self.inp_features) - gamma * theta_h / 2.0
            codebook_comps_h2 = dot(codebook_comps_h.T, codebook_comps_h)
            rank = matrix_rank(codebook_comps_h2)
            zc_search = True
            if rank == codebook_comps_h2.shape[0]:
                new_sparse_f_h = np.linalg.solve(codebook_comps_h2, _q_)
            else:
                u,s,v = np.linalg.svd(codebook_comps_h2)
                col_space = u[:, :rank]
                null_space = u[:, rank:]
                #Check if q belongs in column space, ie the projection of
                #q in the column space is q itself
                q_proj = np.zeros_like(_q_).reshape(-1, 1)
                for i in range(col_space.shape[1]):
                    col = col_space[:,i].reshape(-1, 1)
                    q_proj+=((dot(_q_.reshape(1,-1),col) /
                                   np.dot(col.T, col).astype(float))*col)
                '''
                LOG.info('q|Projection: ' +
                         str(np.concatenate((_q_.reshape(-1,1),q_proj),axis=1)))
                LOG.info('Projection Energy: '+ str(np.sum(q_proj**2)))
                LOG.info('Distance between q and projection: '+str(np.linalg.norm(q_proj.ravel()-_q_.ravel())))
                '''
                if np.allclose(q_proj.ravel()-_q_.ravel(), 0, atol=1.e-6):
                    new_sparse_f_h = dot(pinv(codebook_comps_h2),_q_)
                else:
                    #direction z in nullspace of codebook_comps_h2 can not be
                    #perpendicular to _q_, because then _q_ = C(codebook_comps_h2),
                    #which was proven not to hold.
                    #I take the principal vector that belongs in null_space of
                    #codebook_comps_h2 and add it to the current sparse_feat_h
                    #so that to search for zerocrossings
                    #inside the line constructed
                    # by this vector and sparse_feat_h, which has direction,
                    # belonging to null_space of codebook_comps_h2
                    tmp_sparse_f_h = sparse_feat_h + dot(null_space,
                                         np.ones((null_space.shape[1],1)))
                    zero_points_lin_par = sparse_feat_h / (sparse_feat_h
                                                           -
                                                           tmp_sparse_f_h).astype(float)
                    # find _t_ that corresponds to the closest zero crossing to
                    # sparse_feat_h
                    _t_ind = np.argmin(np.abs(zero_points_lin_par[
                        np.isfinite(zero_points_lin_par)]))
                    _t_ = zero_points_lin_par[
                        np.isfinite(zero_points_lin_par)][_t_ind]
                    null_vec = _t_ * tmp_sparse_f_h + (1 - _t_) * sparse_feat_h
                    new_sparse_f_h = null_vec
                    zc_search = False

            if (np.prod(sign(sparse_feat_h) != sign(new_sparse_f_h))
                and zc_search):
                zero_points_lin_par = sparse_feat_h / (sparse_feat_h -
                                                       new_sparse_f_h).astype(float)
                zero_points_lin_par = concatenate((zero_points_lin_par[
                    ((zero_points_lin_par > 0) *
                     (zero_points_lin_par < 1)).astype(bool)][:], array([1])), axis=0)
                _t_ = zero_points_lin_par
                null_vecs = _t_ * new_sparse_f_h + (1 - _t_) * sparse_feat_h
                objvals = self.object_val_calc(codebook_comps_h, self.inp_features, gamma,
                                               theta_h,
                                               null_vecs).flatten()
                objval_argmin = argmin(objvals)
                objval = np.min(objvals)
                new_sparse_f_h = null_vecs[:, objval_argmin][:, None].copy()
            else:
                objval = self.object_val_calc(codebook_comps_h, self.inp_features, gamma, theta_h,
                                              new_sparse_f_h)
            self.sparse_features[self.active_set] = new_sparse_f_h.copy()
            self.active_set[self.active_set] = np.logical_not(
                isclose(new_sparse_f_h, 0))
            if npsum(self.active_set) < self.min_coeff:
                step2 = 1
                continue
            self.theta = sign(self.sparse_features)
            # Step 4
            nnz_coeff = self.sparse_features != 0
            # a

            new_qp_der_outfeati = 2 * (dot(btb, self.sparse_features) - btf)
            cond_a = (new_qp_der_outfeati +
                      gamma * sign(self.sparse_features)) * nnz_coeff
            '''
            if np.abs(objval) - np.abs(prev_objval) > 100 and not\
                    self.allow_big_vals and not count == 0:
                if self.prev_sparse_feats is not None:
                    SPLOG.info('Current Objective Function value: ' +
                              str(np.abs(objval)))
                    SPLOG.info('Previous Objective Function value: ' +
                              str(np.abs(prev_objval)))
                    SPLOG.info('Problem with big values of inv(B^T*B)' +
                              ',you might want to increase atol' +
                              ' or set flag allow_big_vals to true' +
                              ' (this might cause' +
                              ' problems)')
                    SPLOG.info('Reverting to previous iteration result ' +
                              'and exiting loop..')
                    self.sparse_features = self.prev_sparse_feats.ravel()
                    break
                else:
                    LOG.error('Current Objective Function value: ' +
                              str(np.abs(objval)))
                    LOG.error('Previous Objective Function value: ' +
                              str(np.abs(prev_objval)))
                    LOG.error('Problem with big values of inv(B^T*B),increase atol' +
                              ' or set flag allow_big_vals to true (this might cause' +
                              ' serious convergence problems)')
                    LOG.error('Exiting as algorithm has not produced any'
                              + ' output results.')
                    exit()
            '''
            prev_objval = objval
            self.prev_sparse_feats = self.sparse_features
            if allclose(cond_a, 0, atol=acondtol):
                # go to cond b:
                z_coeff = self.sparse_features == 0
                cond_b = npabs(new_qp_der_outfeati * z_coeff) <= gamma
                if npsum(cond_b) == new_qp_der_outfeati.shape[0]:
                    self.sparse_features = self.sparse_features.reshape((-1,1))
                    converged = True
                    break
                else:
                    # go to step 2
                    step2 = 1
            else:
                # go to step 3
                step2 = 0
            if count % 10 == 0:
                interm_error = compute_lineq_error(
                    self.inp_features, self.codebook_comps,
                    self.sparse_features)
                if interm_error == prev_error or interm_error > initial_energy:
                    converged=True
                    break
                else:
                    prev_error = interm_error
                SPLOG.info('\t Epoch:' + str(count))
                SPLOG.info('\t\t Intermediate Error=' +
                          str(interm_error))
                if interm_error < 0.001:
                    converged=True
                    SPLOG.info('Small error, asssuming  convergence')
                    break
        '''
        if initial_energy < interm_error:
            if not training:
                LOG.warning('FSS Algorithm did not converge, using pseudoinverse' +
                            ' of provided codebook instead')
                if self.inv_codebook_comps is None:
                    self.inv_codebook_comps = pinv(self.codebook_comps)
                self.sparse_features=dot(self.inv_codebook_comps,self.inp_features).ravel()
            else:
                SPLOG.info('FSS Algorithm did not converge,' +
                            ' removing sample from training dataset...')
                self.sparse_features = None
            return (interm_error), False, initial_energy
        else:
        '''
        if not converged:
            SPLOG.info('FSS Algorithm did not converge' +
                  ' in the given iterations')
        else:
            SPLOG.info('Successful Convergence')
        SPLOG.info('\tFinal error: ' + str(interm_error))
        SPLOG.info('\tNumber of nonzero elements: ' +
                  str(np.sum(self.sparse_features!=0)))
        if not single:
            if self.sparse_feat_list is None:
                self.sparse_feat_list = [self.sparse_features.ravel()]
            else:
                self.sparse_feat_list.append(self.sparse_features.ravel())
        if ret_error:
            return (compute_lineq_error(self.inp_features, self.codebook_comps,
                                        self.sparse_features),
                    True, initial_energy)
        self.sparse_features = self.sparse_features.ravel()
        return None, True, None

    def lagrange_dual(self, lbds, ksi, _s_, basis_constraint):
        '''
        Lagrange dual function for the minimization problem
        <ksi> is input, <_s_> is sparse,
        '''
        lbds[lbds==0] = 10**(-18) #the drawback of this method
        self.ksist = dot(ksi, _s_.T)
        interm_result = inv(
            dot(_s_, _s_.T) + diag(lbds.ravel()))
        LOG.debug('Computed Lagrange Coefficients:\n'+str(np.unique(lbds)))
        res = ((dot(ksi.T,ksi)).trace() -
               (dot(dot(self.ksist, interm_result), self.ksist.T)).trace() -
               (basis_constraint * diag(lbds.ravel())).trace())
        return -res # minimizing negative = maximizing positive

    def lagrange_dual_grad(self, lbds, ksi, _s_, basis_constraint):
        '''
        Gradient of lagrange dual function, w.r.t. elf.codebook_comps,
                         self.sparse_feat_list,
                         self.are_sparsecoded_inp) = self.pickle.load(inp)
s_
        '''
        # lbds=lbds.flatten()
        interm_result = inv(
            dot(_s_, _s_.T) + diag(lbds.ravel()))
        interm_result = dot(self.ksist, interm_result)
        interm_result = dot(interm_result.T,interm_result)
        res = diag(interm_result) - basis_constraint
        return -res # minimizing negative = maximizing positive

    def lagrange_dual_hess(self, lbds, ksi, _s_, basis_constraint):
        '''
        It is not used, but it is here in case numpy solver gets also
        the hessian as input
        '''
        interm_result = inv(
            dot(_s_, _s_.T) + diag(lbds.ravel()))
        interm_result1 = dot(self.ksist, interm_result)
        res = -2 * dot(interm_result1.T, interm_result1) * interm_result
        return -res #minimizing negative = maximizing positive
    # pylint: disable=no-member

    def conj_grad_dict_compute(self):
        '''
        Function to train nxm matrix using truncated newton method
        '''
        options = {'disp':True}
        '''
        if self.res_codebook_comps is None:
            self.res_codebook_comps = self.codebook_comps
        LOG.info(self.res_codebook_comps.shape)
        '''
        res = minimize(self.lagrange_dual,
                       self.lbds.copy(),
                       method='Newton-CG',
                      jac=self.lagrange_dual_grad,
                      #hess=self.lagrange_dual_hess,
                 #bounds=np.array(([(10**(-18), 10**10)] *
                 #                 self.sparse_feat_list.shape[0])),
                 #stepmx=50.0,
                 #maxCGit=20,
                 #maxfun=100,
                 options=options,
                 #fmin=0.1,
                 #ftol=0.1,
                 #xtol=0.001,
                 #rescale=1.5,
                 args=(self.are_sparsecoded_inp.copy(),
                             self.sparse_feat_list.copy(),
                             self.basis_constraint)
                      )
        LOG.info(res)
        self.lbds = res.x
        LOG.info(np.unique(self.lbds))
        interm_result = (self.lbds+
                       dot(self.sparse_feat_list,
                           self.sparse_feat_list.T))
        LOG.info(np.linalg.rank(interm_result))
        codebook_comps = dot(inv(interm_result),
                       self.ksist.T).T
        return codebook_comps
# pylint: enable=no-member



    def train_sparse_dictionary(self, data, sp_opt_max_iter=200,
                                init_traindata_num=200, incr_rate=2,
                                min_iterations=3, init_codebook_comps=None,
                                log_lev=None, n_jobs=4):
        if log_lev is not None:
            LOG.setLevel(log_lev)
        self.codebook_comps = DictionaryLearning(
            n_components=self.sparse_dim_rat * data.shape[1],
                                       alpha=co.CONST['sparse_alpha'],
                                       verbose=1, n_jobs=n_jobs).fit(data).components_.T


    @timeit
    def code1(self, data, max_iter=None, errors=False):
        '''
        Sparse codes a single feature
        Requires that the dictionary is already trained
        '''
        if self.codebook is None:
            self.codebook = SparseCoder(self.codebook_comps.T,n_jobs=4)
        return self.codebook.transform(data.reshape(1,-1)).ravel()

    def train_sparse_dictionary1(self, data, sp_opt_max_iter=200,
                                init_traindata_num=200, incr_rate=2,
                                min_iterations=3, init_codebook_comps=None,
                                debug=False):
        '''
        <data> is a numpy array, holding all the features(of single kind) that
        are required to train the sparse dictionary, with dimensions
        [n_features, n_samples]. The sparse dictionary is trained with a random
        subset of <data>, which is increasing in each iteration with rate
        <incr_rate> , along with the max iterations <sp_opt_max_iter> of feature
        sign search algorithm. <min_iterations> is the least number of
        iterations of the dictionary training, after total data is processed.
        '''
        self.sparse_dim = min(data.shape) * self.sparse_dim_rat
        self.flush_variables()
        try:
            import progressbar
        except:
            LOG.warning('Install module progressbar2 to get informed about the'
                        +' feature sign search algorithm progress')
            pass
        self.initialize(data.shape[0], init_codebook_comps=init_codebook_comps)
        iter_count = 0
        retry_count = 0
        LOG.info('Training dictionary: ' + self.name)
        LOG.info('Minimum Epochs number after total data is processed:' + str(min_iterations))
        reached_traindata_num = False
        reached_traindata_count = 0
        computed = data.shape[1] * [None]
        retry = False
        lar_approx = False
        while True:
            LOG.info('Epoch: ' + str(iter_count))
            loaded = False
            self.sparse_feat_list = None
            self.inp_feat_list = None
            if debug and iter_count == 0:
                LOG.warning('Debug is on, loading data from first FSS execution')
                try:
                    with open(self.name+' debug_sparse.pkl','r') as inp:
                        (self.codebook_comps,
                         self.sparse_feat_list,
                         self.are_sparsecoded_inp) = pickle.load(inp)
                        loaded=True
                except (IOError, EOFError):
                    LOG.warning('Not existent '+self.name
                                +' debug_sparse.pkl')
            if not loaded:
                train_num = min(int(init_traindata_num *
                                    (incr_rate) ** iter_count),
                                data.shape[1])
                if train_num == data.shape[1] and not reached_traindata_num:
                    reached_traindata_num = True
                    LOG.info('Total data is processed')
                if reached_traindata_num:
                    reached_traindata_count += 1
                LOG.info('Number of samples used: ' + str(train_num))
                ran = rand.sample(xrange(data.shape[1]), train_num)
                feat_sign_max_iter = min(1000,
                                         sp_opt_max_iter * incr_rate ** iter_count)
                LOG.info('Feature Sign Search maximum iterations allowed:'
                         + str(feat_sign_max_iter))
                try:
                    format_custom_text = progressbar.FormatCustomText(
                        'Mean Initial Error: %(mean_init_energy).4f,'+
                        ' Mean Final Error: %(mean).4f ,Valid Samples Ratio: %(valid).2f',
                            dict(
                                mean_init_energy=0,
                                mean=0,
                                valid=0
                            ),
                        )
                    pbar = progressbar.ProgressBar(max_value=train_num - 1,
                                                  redirect_stdout=True,
                                                   widgets=[progressbar.widgets.Percentage(),
                                                            progressbar.widgets.Bar(),
                                                            format_custom_text])
                    errors=True
                    sum_error = 0
                    sum_energy = 0
                except UnboundLocalError:
                    pbar = None
                    errors = False
                    pass
                are_sparsecoded = []
                if pbar is not None:
                    iterat = pbar(enumerate(ran))
                else:
                    iterat = enumerate(ran)
                for count, sample_count in iterat:
                    fin_error, valid, init_energy = self.feature_sign_search_algorithm(
                        data[:, sample_count],
                        max_iter=feat_sign_max_iter,
                        ret_error=errors,training=True,
                        starting_points=computed[sample_count])
                    are_sparsecoded.append(True)
                    try:
                        if iter_count > 0 and valid:
                            #do not trust first iteration sparse features, before
                            #having trained the codebooks at least once
                            computed[sample_count] = self.sparse_feat_list[-1]
                    except (TypeError,AttributeError):
                        pass
                    if valid and pbar and errors:
                        sum_error += fin_error
                        mean_error = sum_error/float(sum(are_sparsecoded))
                        sum_energy += init_energy
                        mean_init_energy = sum_energy/float(sum(are_sparsecoded))
                    if pbar is not None:
                        format_custom_text.update_mapping(mean_init_energy=
                                                          mean_init_energy,
                                                          mean=mean_error,
                                                          valid=sum(are_sparsecoded)
                                                          /float(len(are_sparsecoded)))
                    self.initialize(data.shape[0])
                self.inp_feat_list = np.transpose(np.array(self.inp_feat_list))
                self.sparse_feat_list = np.array(self.sparse_feat_list).T
                are_sparsecoded = np.array(
                    are_sparsecoded).astype(bool)
                retry = np.sum(are_sparsecoded) < 1 / 3.0 * (are_sparsecoded).size
                self.are_sparsecoded_inp = self.inp_feat_list[:, are_sparsecoded]
                if debug and iter_count==0:
                    LOG.warning('Debug is on, saving debug_sparse.pkl')
                    with open(self.name + ' debug_sparse.pkl','w') as out:
                        pickle.dump((self.codebook_comps,
                                     self.sparse_feat_list,
                                     self.are_sparsecoded_inp), out)
            prev_error = compute_lineq_error(self.are_sparsecoded_inp, self.codebook_comps,
                self.sparse_feat_list)
            if not lar_approx:
                dictionary = self.conj_grad_dict_compute()
                curr_error = compute_lineq_error(
                    self.are_sparsecoded_inp,
                    dictionary,
                    self.sparse_feat_list)
            LOG.info('Reconstruction Error: ' + str(curr_error))
            if loaded:
                mean_init_energy=0
                mean_error = 0
            if curr_error > prev_error or mean_error>1000 or retry or lar_approx:
                if (prev_error > 100 or mean_error>1000
                    or retry or lar_approx):
                    if retry_count == 2 or lar_approx:
                        if iter_count != 0:
                            iter_count = 0
                            lar_approx = True
                            init_traindata_num = data.shape[1]
                            continue
                        LOG.warning('Training has high final error but' +
                                    ' reached maximum retries. No codebook can'
                                    + ' be produced with the fast method,'+
                                     ' using Lagrange Dual, as input'+
                                    ' sparsecoded data S is'
                                    +' ill-conditioned (too low' +
                                    ' rank of the STS).'+
                                     ' Least Angle Regression Method '+
                                    ' will be used')
                        self.codebook_comps = DictionaryLearning(
                            self.sparse_dim,
                            fit_algorithm='lars',
                            code_init=self.inp_feat_list.T).fit(
                                self.are_sparsecoded_inp.T).components_.T
                        curr_error = compute_lineq_error(
                                       self.are_sparsecoded_inp,
                                       self.codebook_comps,
                                       self.sparse_feat_list)
                        LOG.info('Reconstruction Error using LARS: '
                                 + str(curr_error))
                        if curr_error > 1000:
                            LOG.info('LARS method did not converge,' +
                                     ' no codebook is produced.')
                            self.is_trained = False
                            self.codebook_comps = None
                        else:
                            break
                    LOG.warning('Training of codebook ' + self.name + ' completed with no success,'+
                                ' reinitializing (Retry:' + str(retry_count + 1) + ')')
                    self.flush_variables()
                    self.initialize(data.shape[0])
                    computed = data.shape[1] * [None]
                    retry_count += 1
                    iter_count = -1
                    reached_traindata_count = 0
                    reached_traindata_num = False
                elif (np.isclose(prev_error,curr_error,atol=0.1)
                      and reached_traindata_num and
                      reached_traindata_count > min_iterations):
                    break
            if curr_error < 0.5 and reached_traindata_num:
                break
            if (reached_traindata_num and
                reached_traindata_count > min_iterations and
                iter_count >= 0):
                    break
            iter_count += 1
            self.codebook_comps = dictionary
        self.inp_feat_list = None
        self.sparse_feat_list = None
        self.is_trained = True

    @timeit
    def code(self, data, max_iter=None, errors=False):
        '''
        Sparse codes a single feature
        Requires that the dictionary is already trained
        '''
        if max_iter is None:
            max_iter = co.CONST['sparse_fss_max_iter']
        self.initialize(data.size)
        self.feature_sign_search_algorithm(data.ravel(), max_iter=max_iter,
                                           single=True, display_error=errors,
                                           ret_error=errors)
        return self.sparse_features


    def multicode(self, data, max_iter=None, errors=False):
        '''
        Convenience method for sparsecoding multiple features.
        <data> is assumed to have dimensions [n_features, n_samples]
        output has dimensions [n_sparse, n_samples]
        '''
        feat_dim = 0
        for datum in data:
            if datum is not None:
                feat_dim = len(datum)
        if feat_dim == 0 :
            raise Exception('Bad Input, full of nans')
        sparse_features = np.zeros((len(data),
                                    self.sparse_dim_rat* feat_dim))
        for count in range(len(data)):
            if data[count] is not None and np.prod(np.isfinite(data[count][:])):
                sparse_features[count, :] = self.code(data[count][:],
                                                  max_iter, errors).ravel()
            else:
                sparse_features[count, :] = np.nan
        return sparse_features

def compute_lineq_error(prod, matrix, inp):
    return np.linalg.norm(prod - dot(matrix, inp))


def main():
    '''
    Example function
    '''
    import cv2
    import os.path
    import urllib
    if not os.path.exists('lena.jpg'):
        urllib.urlretrieve('https://www.cosy.sbg.ac' +
                           '.at/~pmeerw/Watermarking/lena_color.gif', 'lena.jpg')
    if not os.path.exists('wolves.jpg'):
        urllib.urlretrieve("https://static.decalgirl.com/assets/designs/large/twolves.jpg",
                           "wolves.jpg")

    test = cv2.imread('lena.jpg', -1)
    test = (test.astype(float)) / 255.0
    test2 = cv2.imread('wolves.jpg', 0)
    test2 = test2.astype(float) / 255.0
    test = cv2.resize(test, None, fx=0.05, fy=0.05)
    test2 = cv2.resize(test2, test.shape)
    test_shape = test.shape
    codebook_comps = None
    sparse_coding = SparseCoding(name='Images', sparse_dim_rat=2,
                                 dist_sigma=0.01, dist_beta=0.01,
                                 display=5)
    sparse_coding.train_sparse_dictionary(np.vstack((test.ravel(),
                                                     test2.ravel())).T,
                                          sp_opt_max_iter=200)
    sp_test = sparse_coding.code(test.ravel(), max_iter=500).reshape(-1,1)
    sp_test2 = sparse_coding.code(test2.ravel(), max_iter=500).reshape(-1,1)
    cv2.imshow('reconstructed lena',
               np.dot(sparse_coding.codebook.components_.T,
                                            sp_test).reshape(test.shape))
    cv2.imshow('reconstructed wolves',
               np.dot(sparse_coding.codebook.components_.T,
                                              sp_test2).reshape(test.shape))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
