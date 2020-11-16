import tensorflow as tf
import numpy as np
from collections import deque
import threading

class GPUStratifiedRandomPack():
    def __init__(self, x, y, ry, classes, pack, per_class_size, randstate=123, use_fp16=False, single_fix=False, return_inds=False, **kwargs):
        self.x = x if not use_fp16 else x.astype('float16')
        self.y = y
        self.ry = ry
        self.classes = classes
        self.pack= pack
        self.per_class_size = per_class_size
        self.packsize = classes * per_class_size
        self.randstate = randstate
        self.RNG = np.random.RandomState(randstate)
        self.use_fp16 = use_fp16
        self.single_fix = single_fix
        self.return_inds = return_inds

        self.xshape = list(np.shape(x))

        if pack is None:
            self.xshape[0] = self.packsize
            self.yshape = [self.packsize]
        else:
            self.xshape[0] = pack
            self.xshape.insert(1, self.packsize)
            self.yshape = [pack, self.packsize]

        self.inds= [None]* self.classes
        for c in range(self.classes):
            self.inds[c] = np.where(self.y==c)[0]

        self.choicesize = self.pack *self.per_class_size if (self.pack is not None) else self.per_class_size
        self.sample_replace = self.choicesize*self.classes > len(self.y)

        self.build()

    def build(self):
        if self.single_fix:
            self.sel_inds = [None] * self.classes
            for c in range(self.classes):
                self.sel_inds[c] = self.RNG.choice(self.inds[c], self.choicesize, replace=self.sample_replace)

    def reset(self):
        self.RNG = np.random.RandomState(self.randstate)
        self.build()

    def __len__(self):
        return np.Infinity

    def __iter__(self):
        if self.single_fix:
            sel_pack_x = np.empty(self.xshape, np.float32 if not self.use_fp16 else np.float16)
            sel_pack_y = np.empty(self.yshape, np.int32)
            sel_pack_ry = np.empty(self.yshape, np.int32)
            sel_pack_i = np.empty(self.yshape, np.int32)

            if self.pack is None:
                curs = np.empty((self.packsize,), np.int32)
                for c in range(self.classes):
                    curs[c*self.per_class_size: c*self.per_class_size+self.per_class_size] = self.sel_inds[c][0: self.per_class_size]

                self.RNG.shuffle(curs)
                sel_pack_x = self.x[curs]
                sel_pack_y = self.y[curs]
                sel_pack_ry = self.ry[curs]
                sel_pack_i = curs
            else:
                for p in range(self.pack):
                    curs = np.empty((self.packsize,), np.int32)
                    for c in range(self.classes):
                        curs[c*self.per_class_size: c*self.per_class_size+self.per_class_size] = self.sel_inds[c][p*self.per_class_size: p*self.per_class_size+self.per_class_size]

                    self.RNG.shuffle(curs)

                    sel_pack_x[p] = self.x[curs]
                    sel_pack_y[p] = self.y[curs]
                    sel_pack_ry[p] = self.ry[curs]
                    sel_pack_i[p] = curs

            if not self.return_inds:
                while True:
                    yield sel_pack_x, sel_pack_y, sel_pack_ry
            else:
                while True:
                    yield sel_pack_x, sel_pack_y, sel_pack_ry, sel_pack_i

        else:
            while True:
                sel_inds = [None] * self.classes
                for c in range(self.classes):
                    sel_inds[c] = self.RNG.choice(self.inds[c], self.choicesize, replace=self.sample_replace)

                sel_pack_x = np.empty(self.xshape, np.float32 if not self.use_fp16 else np.float16)
                sel_pack_y = np.empty(self.yshape, np.int32)
                sel_pack_ry = np.empty(self.yshape, np.int32)
                sel_pack_i = np.empty(self.yshape, np.int32)

                if self.pack is None:
                    curs = np.empty((self.packsize,), np.int32)
                    for c in range(self.classes):
                        curs[c*self.per_class_size: c*self.per_class_size+self.per_class_size] = sel_inds[c][0: self.per_class_size]

                    self.RNG.shuffle(curs)

                    sel_pack_x = self.x[curs]
                    sel_pack_y = self.y[curs]
                    sel_pack_ry = self.ry[curs]
                    sel_pack_i = curs
                else:
                    for p in range(self.pack):
                        curs = np.empty((self.packsize,), np.int32)
                        for c in range(self.classes):
                            curs[c*self.per_class_size: c*self.per_class_size+self.per_class_size] = sel_inds[c][p*self.per_class_size: p*self.per_class_size+self.per_class_size]

                        self.RNG.shuffle(curs)

                        sel_pack_x[p] = self.x[curs]
                        sel_pack_y[p] = self.y[curs]
                        sel_pack_ry[p] = self.ry[curs]
                        sel_pack_i[p] = curs

                if not self.return_inds:
                    yield sel_pack_x, sel_pack_y, sel_pack_ry
                else:
                    yield sel_pack_x, sel_pack_y, sel_pack_ry, sel_pack_i

class GPURandomPack():
    def __init__(self, x, y, ry, pack, packsize, randstate=123, use_fp16=False, cycle=False, infinite=False, force_shuffle=False, **kwargs):
        self.x = x if not use_fp16 else x.astype('float16')
        self.y = y
        self.ry = ry
        self.packsize = packsize
        self.randstate= randstate
        self.RNG = np.random.RandomState(randstate)
        self.use_fp16 = use_fp16
        self.infinite= infinite
        self.cycle = cycle
        self.pack = pack
        self.force_shuffle = force_shuffle

        self.xshape = list(np.shape(x))
        if pack:
            self.xshape[0] = 1
            self.xshape.insert(1, self.packsize)
            self.yshape = [1, self.packsize]
        else:
            self.xshape[0] = self.packsize
            self.yshape = [self.packsize]

        self.batchnum = int(np.ceil(len(y) / self.packsize))

        self.build()

    def build(self):
        self.inds =  self.RNG.permutation(len(self.y))

    def reset(self):
         self.RNG = np.random.RandomState(self.randstate)
         self.build()

    def __len__(self):
        return self.batchnum if not self.infinite else np.Infinity

    def __iter__(self):
        if self.force_shuffle:
            self.build()

        pcur = 0
        ecur = self.packsize
        if not self.infinite:
            if self.cycle:
                cycle_iter = 0
                while(True):
                    if ecur > len(self.inds):
                        ecur=len(self.inds)

                    if self.pack:
                        sel_pack_x = np.expand_dims(self.x[self.inds[pcur:ecur]], 0)
                        sel_pack_y = np.expand_dims(self.y[self.inds[pcur:ecur]], 0)
                        sel_pack_ry = np.expand_dims(self.ry[self.inds[pcur:ecur]], 0)
                    else:
                        sel_pack_x = self.x[self.inds[pcur:ecur]]
                        sel_pack_y = self.y[self.inds[pcur:ecur]]
                        sel_pack_ry = self.ry[self.inds[pcur:ecur]]

                    pcur+=len(sel_pack_y)
                    ecur+=len(sel_pack_y)

                    cycle_iter+=1
                    if cycle_iter>=self.batchnum:
                        pcur = 0
                        ecur = self.packsize

                    yield sel_pack_x, sel_pack_y, sel_pack_ry
            else:
                for i in range(self.batchnum):
                    if ecur > len(self.inds): 
                        ecur=len(self.inds)

                    if self.pack:
                        sel_pack_x = np.expand_dims(self.x[self.inds[pcur:ecur]], 0)
                        sel_pack_y = np.expand_dims(self.y[self.inds[pcur:ecur]], 0)
                        sel_pack_ry = np.expand_dims(self.ry[self.inds[pcur:ecur]], 0)
                    else:
                        sel_pack_x = self.x[self.inds[pcur:ecur]]
                        sel_pack_y = self.y[self.inds[pcur:ecur]]
                        sel_pack_ry = self.ry[self.inds[pcur:ecur]]

                    pcur+=len(sel_pack_y)
                    ecur+=len(sel_pack_y)

                    yield sel_pack_x, sel_pack_y, sel_pack_ry
        else:
            while True:
                sinds =  self.RNG.permutation(len(self.y))[:self.packsize]
                if self.pack:
                    sel_pack_x = np.expand_dims(self.x[sinds], 0)
                    sel_pack_y = np.expand_dims(self.y[sinds], 0)
                    sel_pack_ry = np.expand_dims(self.ry[sinds], 0)
                else:
                    sel_pack_x = self.x[sinds]
                    sel_pack_y = self.y[sinds]
                    sel_pack_ry = self.ry[sinds]

                yield sel_pack_x, sel_pack_y, sel_pack_ry