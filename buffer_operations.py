import logging
import numpy as np
import cv2
from __init__ import initialize_logger, timeit, PRIM_X, PRIM_Y, FLNT, find_nonzero


class BufferOperations(object):

    def __init__(self, parameters, reset_time=True):
        self.logger = logging.getLogger('BufferOperations')
        initialize_logger(self.logger)
        self.parameters = parameters
        self.buffer = []
        self.depth = []
        self.testing = parameters['testing']
        self.action_type = parameters['action_type']
        self.samples_indices = []
        self.buffer_start_inds = []
        self.buffer_end_inds = []
        if not self.action_type == 'Passive':
            self.ptpca = parameters['PTPCA']
            self.ptpca_components = parameters['PTPCA_params'][
                'PTPCA_components']
        self.bbuffer = [[] for i in range(len(parameters['descriptors']))]
        if not self.action_type == 'Passive':
            self.buffer_size = parameters['dynamic_params']['buffer_size']
            try:
                self.buffer_confidence_tol = parameters['dynamic_params'][
                    'buffer_confidence_tol']
                self.ptpca = parameters['PTPCA']
                self.ptpca_components = parameters['PTPCA_params'][
                    'PTPCA_components']
            except (KeyError, IndexError, TypeError):
                self.buffer_confidence_tol = None
            self.pca_features = []
        else:
            self.buffer_size = 1
        self.sync = []
        self.frames_inds = []
        self.samples_inds = []
        self.buffer_components = []
        self.depth_components = []
        self.real_samples_inds = []
        if reset_time:
            self.time = []

    def reset(self, reset_time=False):
        self.__init__(self.parameters, reset_time=reset_time)

    def check_buffer_integrity(self, buffer):
        check_sam = True
        check_cont = True
        check_len = len(buffer) == self.buffer_size
        if check_len:
            if not self.action_type == 'Passive':

                check_cont = np.all(np.abs(np.diff(self.frames_inds[-self.buffer_size:])) <=
                                    self.buffer_size * self.buffer_confidence_tol)
                # check if buffer frames belong to the same sample, in case of
                # training
                check_sam = self.testing or len(np.unique(
                    self.samples_inds[-self.buffer_size:])) == 1
            else:
                check_cont = True
                check_sam = True
                check_len = True
        return check_len and check_cont and check_sam

    @timeit
    def perform_post_time_pca(self, inp):
        reshaped = False
        if self.buffer_size == 1:
            return
        if np.shape(inp)[0] == 1 or len(np.shape(inp)) == 1:
            reshaped = True
            inp = np.reshape(inp, (self.buffer_size, -1))
        mean, inp = cv2.PCACompute(
            np.array(inp),
            np.array([]),
            maxComponents=self.ptpca_components)
        inp = (np.array(inp) + mean)
        if reshaped:
            return inp.ravel()
        return inp

    def update_buffer_info(self, sync, samples_index=0,
                           samples=None, depth=None):
        self.frames_inds.append(sync)
        self.samples_inds.append(samples_index)
        if samples is not None:
            self.buffer_components.append(samples)
            del self.buffer_components[:-self.buffer_size]
        if depth is not None:
            self.depth_components.append(depth)
            del self.depth_components[:-self.buffer_size]

    def add_buffer(self, buffer=None, depth=None, sample_count=None,
                   already_checked=False):
        '''
        <buffer> should have always the same size.
        <self.bbuffer> is a list of buffers. It can have a size limit, after which it
        acts as a buffer (useful for shifting window
        operations (filtering etc.))
        '''
        # check buffer contiguousness
        if buffer is None:
            buffer = self.buffer_components
        if depth is None:
            fmask = np.isfinite(self.depth_components)
            if np.sum(fmask):
                depth = np.mean(np.array(self.depth_components)[fmask])
        if not already_checked:
            check = self.check_buffer_integrity(buffer[-self.buffer_size:])
        else:
            check = True
        if not self.parameters['testing_params']['online']:
            self.real_samples_inds += [-1] * (self.frames_inds[-1] + 1 -
                                              len(self.buffer))
            self.depth += [None] * (self.frames_inds[-1] + 1
                                    - len(self.buffer))
            self.buffer += [None] * (self.frames_inds[-1] + 1
                                     - len(self.buffer))
        if check:
            self.buffer_start_inds.append(self.frames_inds[-self.buffer_size])
            self.buffer_end_inds.append(self.frames_inds[-1])
            if not self.parameters['testing_params']['online']:
                self.buffer[self.frames_inds[-1]] = np.array(
                    buffer)
                self.depth[self.frames_inds[-1]] = depth
            else:
                self.buffer = np.array(buffer)
                self.depth = depth
            if not self.parameters['testing_params']['online']:
                self.real_samples_inds[self.frames_inds[-1]] = (np.unique(self.samples_inds[
                    -self.buffer_size:])[0])
        else:
            if self.parameters['testing_params']['online']:
                self.buffer = None
                self.depth = None

    def extract_buffer_list(self):
        '''
        Returns a 2d numpy array, which has as first dimension the number of
        saved features sets inside <self.bbuffer>,
        as second dimension a flattened buffer. In case it is online, the first
        dimension is 1. In case there are None samples inside, those are turned
        to None arrays.
        '''
        if self.parameters['testing_params']['online']:
            if self.bbuffer is None:
                return None
        else:
            buffer_len = 0
            for _buffer in self.buffer:
                if _buffer is not None:
                    buffer_len = np.size(_buffer)
                    break
            if not buffer_len:
                self.logger.debug('No valid buffer')
                return None
        npbuffer = np.zeros((len(self.buffer), buffer_len))
        for buffer_count in range(len(self.buffer)):
            if self.buffer[buffer_count] is None:
                self.buffer[buffer_count] = np.zeros(buffer_len)
                self.buffer[buffer_count][:] = np.nan
            npbuffer[buffer_count, ...] =\
                np.array(self.buffer[buffer_count]).T.ravel()
        return npbuffer, self.real_samples_inds, self.depth
