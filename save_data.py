import numpy as np

def save_array(array, filename, dictname=None, append=False, overwrite=False,
               use=None):
    '''
    save <array> to a file with name <filename>. <use> can be
    ['pickle','bytearray','hdf5'], with 'bytearray'(default) using 'json' module and
    'to_bytes',
    'pickle' using 'CPickle' module and 'hdf5' using h5py. If <dictname> the assumption is made that
    array will be saved inside a dictionary, making 'hdf5' the default way to
    save, and the following stands:
        -if <append> and an entry with the same <dictname> exists inside the
        dictionary, then the entry becomes a list, if not being already one,
        and the array is appended to the list. Else the array replaces the
        entry
        -if <overwrite> then the dictionary is reset, with <dictname> being the
        only entry left.
    '''
    if use is None:
        if dictname is None:
            use = 'bytearray'
        else:
            use = 'hdf5'
    if use == 'pickle':
        import cPickle as pickle
        save = pickle.dump
    elif use == ' bytearray':
        try:
            import ujson as json
        except ImportError:
            import json
        save = json.dump
    if dictname is not None:
        try:
            with
    with open('filename'
