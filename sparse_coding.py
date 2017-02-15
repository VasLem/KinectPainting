import warnings
import sys
import logging
import numpy as np
from numpy import array, argmax, argmin, concatenate, diag, isclose
from numpy import dot, sign, zeros, zeros_like, random, trace, mean
from numpy import allclose
from numpy.linalg import inv, pinv, matrix_rank, qr
from numpy import sum as npsum
from numpy import abs as npabs
import random as rand
from scipy.optimize import fmin_tnc
import class_objects as co

LOG = logging.getLogger('__name__')
CH = logging.StreamHandler(sys.stderr)
CH.setFormatter(logging.Formatter(
    '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel(logging.INFO)


class SparseCoding(object):

    def __init__(self, log_lev='INFO', des_dim=None, name='',
                 dist_beta=0.1, dist_sigma=0.005, display=0):
        LOG.setLevel(log_lev)
        self.name = name
        self.bmat = None
        self.active_set = None
        self.inp_features = None
        self.sparse_features = None
        self.basis_constraint = 1
        self.res_lbd = None
        self.max_iter = 500
        self.dict_max_iter = 300
        self.display = display
        self.prev_err = 0
        self.curr_error = 0
        self.allow_big_vals = False
        self.des_dim = des_dim
        if des_dim is None:
            self.des_dim = co.CONST['des_dim']
        self.dist_sigma = dist_sigma
        self.dist_beta = dist_beta
        self.theta = None
        self.prev_sparse_feats = None
        self.flush_flag = False
        self.sparse_feat_list = None
        self.inp_feat_list = None

    def flush_variables(self):
        '''
        Empty variables
        '''
        self.active_set = None
        self.theta = None
        self.bmat = None
        self.inp_features = None
        self.sparse_features = None
        self.flush_flag = True
        self.res_lbd = np.ones(self.des_dim, np.float64)

    def initialize(self, feat_dim,
                   init_bmat=None):
        '''
        Initialises B dictionary and s
        '''
        if init_bmat is not None:
            if (init_bmat.shape[0] == feat_dim and
                    init_bmat.shape[1] == self.des_dim):
                self.bmat = init_bmat.copy()
            else:
                raise Exception('Wrong input of initial B matrix, the dimensions' +
                                ' should be ' + str(feat_dim) + 'x' +
                                str(self.des_dim) + ', not ' +
                                str(init_bmat.shape[0]) + 'x' +
                                str(init_bmat.shape[1]))
        if (self.bmat is None) or self.flush_flag:
            self.bmat = random.random((feat_dim, self.des_dim)) - 0.5
            self.bmat -= mean(self.bmat, axis=1)[:, None]
            sigm = np.sqrt(npsum(self.bmat * self.bmat, axis=0))
            sigm[sigm == 0] = 1
            self.bmat /= sigm
        if (self.sparse_features is None) or self.flush_flag:
            self.sparse_features = zeros((self.des_dim, 1))
        self.theta = zeros(self.des_dim)
        self.active_set = zeros((self.des_dim), bool)
        self.sparse_features = zeros((self.des_dim, 1))
        self.flush_flag = False
        self.is_trained = False

    def object_val_calc(self, bmat, ksi, gamma, theta, vecs):
        '''
        Calculate objective function value
        '''
        _bs_ = np.dot(bmat, vecs)
        square_term = 0.5 * npsum((ksi - _bs_)**2, axis=0)
        res = (square_term + gamma * dot(theta.T, vecs)).ravel()
        return res

    def feature_sign_search_algorithm(self,
                                      inp_features,
                                      acondtol=1e-3,
                                      ret_error=False,
                                      display_error=False,
                                      max_iter=0,
                                      single=False, timed=True):
        '''
        Returns sparse features representation
        '''
        if self.inp_feat_list is not None:
            self.inp_feat_list.append(inp_features.ravel())
        else:
            self.inp_feat_list = [inp_features.ravel()]
        self.inp_features = np.atleast_2d(inp_features.copy()).T
        # Step 1
        btb = dot(self.bmat.T, self.bmat)
        btf = dot(self.bmat.T, self.inp_features)
        gamma = min([2 * self.dist_sigma**2 * self.dist_beta,
                     co.CONST['max_gamma'], np.max(-2 * btf) / 10])
        # Set max iterations
        step2 = 1
        singularity_met = False
        count = 0
        prev_objval = 0
        if max_iter == 0:
            max_iter = self.max_iter
        else:
            self.max_iter = max_iter
        self.prev_sparse_feats = None
        for count in range(self.max_iter):
            # Step 2
            if step2:
                zero_coeffs = (self.sparse_features == 0)
                qp_der_outfeati = 2 * \
                    (dot(btb, self.sparse_features) - btf) * zero_coeffs
                i = argmax(npabs(qp_der_outfeati))
                if npabs(qp_der_outfeati[i]) >= gamma:
                    self.theta[i] = -sign(qp_der_outfeati[i])
                    self.active_set[i] = True
                elif count == 0:
                    gamma = 0.8 * npabs(qp_der_outfeati[i])
                    self.theta[i] = -sign(qp_der_outfeati[i])
                    self.active_set[i] = True
            # Step 3
            bmat_h = self.bmat[:, self.active_set]
            sparse_feat_h = self.sparse_features[self.active_set]
            theta_h = self.theta[self.active_set]
            _q_ = dot(bmat_h.T, self.inp_features) - gamma * theta_h / 2.0
            bmat_h2 = dot(bmat_h.T, bmat_h)
            try:
                new_sparse_f_h = np.linalg.solve(bmat_h2, _q_)
                if np.max(np.abs(new_sparse_f_h)) > 1000:
                    raise np.linalg.linalg.LinAlgError
            except np.linalg.linalg.LinAlgError:
                rank = matrix_rank(bmat_h2)
                col_space = qr(bmat_h2.T)[0][:, :rank]
                dot_prod = np.sum((_q_ * col_space), axis=0)
                sub_projection = dot_prod / \
                    np.sum(col_space * col_space, axis=0)
                projected_q = npsum(sub_projection * col_space, axis=1)
                if np.allclose(projected_q.ravel(), _q_.ravel(), atol=1e-03):
                    invbmat_h2 = pinv(bmat_h2)
                else:
                    # has not be met with present data, so I just add small
                    # diagonal quantities
                    warnings.warn('Weird singularity met')
                    singularity_met = True
                    invbmat_h2 = inv(bmat_h2 + 0.001 *
                                     np.eye(bmat_h2.shape[0]))
                    if np.max(np.abs(invbmat_h2)) > 1000:
                        raise Exception('Wrongly computed bmat, try to run' +
                                        ' again or change input features')
                new_sparse_f_h = dot(invbmat_h2, _q_)

            if np.prod(sign(sparse_feat_h) != sign(new_sparse_f_h)):
                zero_points_lin_par = sparse_feat_h / (sparse_feat_h -
                                                       new_sparse_f_h).astype(float)
                zero_points_lin_par = concatenate((zero_points_lin_par[
                    ((zero_points_lin_par > 0) *
                     (zero_points_lin_par < 1)).astype(bool)][:], array([1])), axis=0)
                _t_ = zero_points_lin_par
                null_vecs = _t_ * new_sparse_f_h + (1 - _t_) * sparse_feat_h
                objvals = self.object_val_calc(bmat_h, self.inp_features, gamma,
                                               theta_h,
                                               null_vecs).flatten()
                objval_argmin = argmin(objvals)
                objval = np.min(objvals)
                new_sparse_f_h = null_vecs[:, objval_argmin][:, None].copy()
            else:
                objval = self.object_val_calc(bmat_h, self.inp_features, gamma, theta_h,
                                              new_sparse_f_h)
            self.sparse_features[self.active_set] = new_sparse_f_h.copy()
            self.active_set[self.active_set] = np.logical_not(
                isclose(new_sparse_f_h, 0))
            self.theta = sign(self.sparse_features)
            # Step 4
            nnz_coeff = self.sparse_features != 0
            # a

            new_qp_der_outfeati = 2 * (dot(btb, self.sparse_features) - btf)
            cond_a = (new_qp_der_outfeati +
                      gamma * sign(self.sparse_features)) * nnz_coeff
            if np.abs(objval) - np.abs(prev_objval) > 100 and not\
                    self.allow_big_vals and not count == 0:
                if self.prev_sparse_feats is not None:
                    LOG.debug('Current Objective Function value: ' +
                              str(np.abs(objval)))
                    LOG.debug('Previous Objective Function value: ' +
                              str(np.abs(prev_objval)))
                    LOG.debug('Problem with big values of inv(B^T*B)' +
                              ',you might want to increase atol' +
                              ' or set flag allow_big_vals to true' +
                              ' (this might cause' +
                              ' problems)')
                    LOG.debug('Reverting to previous iteration result ' +
                              'and exiting loop..')
                    self.sparse_features = self.prev_sparse_feats
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
            prev_objval = objval
            self.prev_sparse_feats = self.sparse_features
            if allclose(cond_a, 0, atol=acondtol):
                # go to cond b:
                z_coeff = self.sparse_features == 0
                cond_b = npabs(new_qp_der_outfeati * z_coeff) <= gamma
                if npsum(cond_b) == new_qp_der_outfeati.shape[0]:
                    if not single:
                        if self.sparse_feat_list is None:
                            self.sparse_feat_list = [
                                self.sparse_features.ravel()]
                        else:
                            self.sparse_feat_list.append(
                                self.sparse_features.ravel())
                    if ret_error:
                        final_error = compute_lineq_error(self.inp_features,
                                                          self.bmat,
                                                          self.sparse_features)
                        LOG.debug('Reconstrunction error after ' +
                                  'output vector correction: ' + str(final_error))
                        if display_error:
                            LOG.debug('Final Error' + str(final_error))
                        return final_error, singularity_met
                    return None, None
                else:
                    # go to step 2
                    step2 = 1
            else:
                # go to step 3
                step2 = 0
            if count % 100 == 0:
                interm_error = compute_lineq_error(
                    self.inp_features, self.bmat,
                    self.sparse_features)
                LOG.debug('\t Epoch:' + str(count))
                LOG.debug('\t\t Intermediate Error=' +
                          str(interm_error))
                if interm_error < 0.001:
                    LOG.debug('Too small error, asssuming  convergence')
                    break
        LOG.debug('Algorithm did not converge' +
                  ' in the given iterations with' +
                  ' error ' + str(interm_error) +
                  ', you might want to change' +
                  ' tolerance or increase iterations')
        if not single:
            if self.sparse_feat_list is None:
                self.sparse_feat_list = [self.sparse_features.ravel()]
            else:
                self.sparse_feat_list.append(self.sparse_features.ravel())
        if ret_error:
            return (compute_lineq_error(self.inp_features, self.bmat,
                                        self.sparse_features),
                    singularity_met)
        return None, None

    def lagrange_dual(self, lbd, ksi, _s_):
        '''
        Lagrange dual function for the minimization problem
        <ksi> is input, <_s_> is sparse,
        <lbd> is the lasso constraint coefficient
        '''
        ksi = self.inp_feat_list
        ksist = dot(ksi, _s_.T)
        self.res_lbd = np.array(lbd)
        try:
            interm_result = inv(dot(_s_, _s_.T) + diag(lbd))
        except np.linalg.linalg.LinAlgError:
            LOG.warning('Singularity met while computing Lagrange Dual')
            LOG.debug('\t sum(lbds)= ' + str(npsum(lbd)))
            LOG.debug('\t trace(dot(_s_,_s_.T))=' +
                      str(trace(dot(_s_, _s_.T))))
            interm_result = inv(dot(_s_, _s_.T) +
                                diag(lbd) +
                                0.01 * self.basis_constraint *
                                np.eye(lbd.shape[0]))
        res = (dot(dot(ksist, interm_result), ksist.T).trace() +
               (self.basis_constraint * diag(lbd)).trace())
        return res

    def lagrange_dual_grad(self, lbds, ksi, _s_):
        '''
        Gradient of lagrange dual function, w.r.t. _s_
        '''
        # lbds=lbds.flatten()
        try:
            interm_result = dot(dot(ksi, _s_.T),
                                inv(dot(_s_, _s_.T) + diag(lbds)))
        except np.linalg.linalg.LinAlgError:
            LOG.warning('Singularity met while computing ' +
                        'Lagrange Dual Gradient')
            LOG.debug('\t sum(lbds)= ' + str(npsum(lbds)))
            LOG.debug('\t trace(dot(_s_,_s_.T))= ' +
                      str(trace(dot(_s_, _s_.T))))
            interm_result = dot(dot(ksi, _s_.T),
                                inv(dot(_s_, _s_.T) +
                                    diag(lbds) +
                                    self.basis_constraint *
                                    np.eye(lbds.shape[0])))
        res = zeros_like(lbds)
        for count in range(res.shape[0]):
            res[count] = -(np.dot(interm_result[:, count].T,
                                  interm_result[:, count]) -
                           self.basis_constraint)
        return res.reshape(lbds.shape)

    def lagrange_dual_hess(self, lbds, ksi, _s_):
        '''
        It is not used, but it is here in case numpy solver gets also
        the hessian as input
        '''
        ksist = dot(ksi, _s_.T)
        try:
            interm_result0 = inv(dot(_s_, _s_.T) + diag(lbds))
        except np.linalg.linalg.LinAlgError:
            LOG.warning('Singularity met while computing' +
                        'Lagrange Dual Hessian')
            LOG.debug('\t sum(lbds)= ' + str(npsum(lbds)))
            LOG.debug('\t trace(dot(_s_,_s_.T))= ' +
                      str(trace(dot(_s_, _s_.T))))
            interm_result0 = inv(
                dot(_s_, _s_.T) + diag(lbds) + np.eye(lbds.shape[0]))
        interm_result1 = dot(interm_result0, ksist.T)
        res = 2 * dot(interm_result1, interm_result1.T) * interm_result0
        return res
    # pylint: disable=no-member

    def conj_grad_dict_compute(self):
        '''
        Function to train nxm matrix using truncated newton method
        '''
        self.prev_err = compute_lineq_error(self.inp_feat_list, self.bmat,
                                            self.sparse_feat_list)
        LOG.debug('Reconstruction error before dictionary training: ' +
                  str(self.prev_err))
        fmin_tnc(self.lagrange_dual,
                 (self.res_lbd).tolist(),
                 fprime=self.lagrange_dual_grad,
                 bounds=np.array(([(10**(-18), 10**4)] *
                                  self.sparse_feat_list.shape[0])),
                 stepmx=50.0,
                 maxCGit=20,
                 maxfun=100,
                 disp=self.display,
                 fmin=0.1,
                 ftol=0.1,
                 xtol=0.001,
                 rescale=1.5,
                 args=(self.inp_feat_list.copy(),
                       self.sparse_feat_list.copy()))

        try:
            bmat = dot(dot(self.inp_feat_list, self.sparse_feat_list.T),
                       inv(dot(self.sparse_feat_list,
                               self.sparse_feat_list.T) + diag(self.res_lbd)))
        except np.linalg.linalg.LinAlgError:
            LOG.warning('Singularity met while training dictionary')
            bmat = dot(dot(self.inp_feat_list, self.sparse_feat_list.T),
                       inv(dot(self.sparse_feat_list,
                               self.sparse_feat_list.T) +
                           diag(self.res_lbd) +
                           0.01 * self.basis_constraint *
                           np.eye(self.res_lbd.shape[0])))
        LOG.debug('Reconstruction error after dictionary correction: ' +
                  str(compute_lineq_error(self.inp_feat_list,
                                          bmat, self.sparse_feat_list)))
        return bmat
# pylint: enable=no-member

    def train_sparse_dictionary(self, data, sp_opt_max_iter=200,
                                init_traindata_num=20, incr_rate=2,
                                min_iterations=3, init_bmat=None):
        '''
        <data> is a numpy array, holding all the features(of single kind) that
        are required to train the sparse dictionary, with dimensions
        [n_features, n_samples]. The sparse dictionary is trained with a random
        subset of <data>, which is increasing in each iteration with rate
        <incr_rate> , along with the max iterations <sp_opt_max_iter> of feature
        sign search algorithm. <min_iterations> is the least number of
        iterations of the dictionary training, after total data is processed.
        '''
        self.flush_variables()
        try:
            import progressbar
        except:
            LOG.warning('Install module progressbar2 to get informed about the'
                        +' feature sign search algorithm progress')
            pass
        self.initialize(data.shape[0], init_bmat=init_bmat)
        iter_count = 0
        retry_count = 0
        LOG.info('Training dictionary: ' + self.name)
        LOG.info('Minimum Epochs number after total data is processed:' + str(min_iterations))
        reached_traindata_num = False
        reached_traindata_count = 0

        while True:
            LOG.info('Epoch: ' + str(iter_count))
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
                pbar = progressbar.ProgressBar(max_value=train_num - 1,
                                              redirect_stdout=True,
                                               widgets=[progressbar.widgets.Percentage(),
                                                        progressbar.widgets.Bar(),
                                                        progressbar.widgets.DynamicMessage
                                                   ('error')])
                errors=True
                sum_error = 0
            except UnboundLocalError:
                bar = None
                errors = False
                pass
            for count, sample_count in enumerate(ran):
                fin_error, _ = self.feature_sign_search_algorithm(data[:, sample_count],
                                                   max_iter=feat_sign_max_iter,
                                                   ret_error=errors)
                if pbar is not None:
                    sum_error += fin_error
                    pbar.update(count,error=sum_error/(count+1))
                self.initialize(data.shape[0])
            self.inp_feat_list = np.transpose(np.array(self.inp_feat_list))
            self.sparse_feat_list = np.array(self.sparse_feat_list).T
            prev_error = compute_lineq_error(self.inp_feat_list, self.bmat,
                                             self.sparse_feat_list)
            dictionary = self.conj_grad_dict_compute()
            curr_error = compute_lineq_error(
                self.inp_feat_list, dictionary, self.sparse_feat_list)
            self.sparse_feat_list = None
            self.inp_feat_list = None
            LOG.info('Reconstruction Error: ' + str(curr_error))
            iter_count += 1
            if curr_error < 0.5 and reached_traindata_num:
                break
            if curr_error >= prev_error:
                if prev_error > co.CONST['max_dict_error']:
                    if retry_count == co.CONST['max_retries']:
                        LOG.warning('Training has high final error but' +
                                    ' reached maximum retries')
                        break
                    LOG.warning('Training completed with no success,'+
                                ' reinitializing (Retry:' + str(retry_count) + ')')
                    self.flush_variables()
                    self.initialize(data.shape[0])
                    iter_count = 0
            else:
                if reached_traindata_num and reached_traindata_count > min_iterations:
                    break
            self.bmat = dictionary
        self.is_trained = True

    def code(self, data, max_iter=None, errors=False):
        '''
        Sparse codes a single feature
        Requires that the dictionary is already trained
        '''
        if max_iter is None:
            max_iter = self.des_dim
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
        sparse_features = np.zeros((self.des_dim, data.shape[1]))
        for count in range(data.shape[1]):
            sparse_features[:, count] = self.code(data[:, count],
                                                  max_iter, errors).ravel()
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
    bmat = None
    sparse_coding = SparseCoding(name='Images', des_dim=2 *
                                 test.size, dist_sigma=0.01, dist_beta=0.01,
                                 display=5)
    sparse_coding.train_sparse_dictionary(np.vstack((test.ravel(),
                                                     test2.ravel())).T,
                                          sp_opt_max_iter=200)
    sp_test = sparse_coding.code(test.ravel(), max_iter=500)
    sp_test2 = sparse_coding.code(test2.ravel(), max_iter=500)
    cv2.imshow('lena', np.dot(sparse_coding.bmat, sp_test).reshape(test.shape))
    cv2.imshow(
        'wolves',
        np.dot(
            sparse_coding.bmat,
            sp_test2).reshape(
            test.shape))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
