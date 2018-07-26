import class_objects as co
class DataLoader(object):
    def __init__(self, path):
        (self.imgs, self.masks, self.sync, self.angles,
         self.centers,
         self.samples_inds) = co.imfold_oper.load_frames_data(
             path)
