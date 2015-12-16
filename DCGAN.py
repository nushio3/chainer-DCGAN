import pickle,argparse,os,subprocess
import numpy as np
from PIL import Image
from StringIO import StringIO
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage



import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L


import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID')
args = parser.parse_args()


image_dir = './images'
out_image_dir = './out_images_%s'%(args.gpu)
out_model_dir = './out_models_%s'%(args.gpu)


nz = 100          # # of dim for Z
batchsize=25
n_epoch=10000
n_train=200000
image_save_interval = 50000

# read all images

fs = os.listdir(image_dir)
print len(fs)
dataset = []
for fn in fs:
    f = open('%s/%s'%(image_dir,fn), 'rb')
    img_bin = f.read()
    dataset.append(img_bin)
    f.close()
print len(dataset)

class ELU(function.Function):

    """Exponential Linear Unit."""
    # https://github.com/muupan/chainer-elu

    def __init__(self, alpha=1.0):
        self.alpha = numpy.float32(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (numpy.exp(y[neg_indices]) - 1)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : alpha * (exp(x) - 1)', 'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        neg_indices = x[0] < 0
        gx[neg_indices] *= self.alpha * numpy.exp(x[0][neg_indices])
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T alpha', 'T gx',
            'gx = x >= 0 ? gy : gy * alpha * exp(x)', 'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,


def elu(x, alpha=1.0):
    """Exponential Linear Unit function."""
    # https://github.com/muupan/chainer-elu
    return ELU(alpha=alpha)(x)




class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0z = L.Linear(nz, 6*6*512, wscale=0.02*math.sqrt(nz)),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn0l = L.BatchNormalization(6*6*512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )
        
    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)), (z.data.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x


class Retoucher(chainer.Chain):
    def __init__(self):
        super(Retoucher, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc1 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc0 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256)
        )
        
    def __call__(self, x, test=False):
        h = elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = elu(self.bn1(self.c1(h), test=test))
        h = elu(self.bn2(self.c2(h), test=test))

        h = F.relu(self.bn1(self.dc2(h), test=test))
        h = F.relu(self.bn0(self.dc1(h), test=test))
        y = (self.dc0(h))
        return x+1e-2*y







class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            l4l = L.Linear(6*6*512, 2, wscale=0.02*math.sqrt(6*6*512)),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = elu(self.bn1(self.c1(h), test=test))
        h = elu(self.bn2(self.c2(h), test=test))
        h = elu(self.bn3(self.c3(h), test=test))
        l = self.l4l(h)
        return l


class Discriminator2(chainer.Chain):
    def __init__(self):
        super(Discriminator2, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            l4l = L.Linear(256, 2, wscale=0.02*math.sqrt(6*6*512)),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
        )
        
    def __call__(self, x, test=False):
        h = elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = elu(self.bn1(self.c1(h), test=test))
        h = elu(self.bn2(self.c2(h), test=test))
        l = self.l4l(F.sum(h,(2,3)))
        return l





def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))


def train_dcgan_labeled(gen, retou, dis, dis2, epoch0=0):
    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_retou = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis2 = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen.setup(gen)
    o_retou.setup(retou)
    o_dis.setup(dis)
    o_dis2.setup(dis2)
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_retou.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis2.add_hook(chainer.optimizer.WeightDecay(0.00001))

    zvis = (xp.random.uniform(-1, 1, (100, nz), dtype=np.float32))
    
    for epoch in xrange(epoch0,n_epoch):
        print "epoch:", epoch
        perm = np.random.permutation(n_train)
        sum_l_dis = np.float32(0)
        sum_l_gen = np.float32(0)
        
        for i in xrange(0, n_train, batchsize):
            print i,
            # discriminator
            # 0: from dataset
            # 1: from noise

            #print "load image start ", i
            x2 = np.zeros((batchsize, 3, 96, 96), dtype=np.float32)
            for j in range(batchsize):
                try:
                    rnd = np.random.randint(len(dataset))
                    rnd2 = np.random.randint(2)

                    img = np.asarray(Image.open(StringIO(dataset[rnd])).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
                    # offset the image about the center of the image.
                    oy = (img.shape[1]-96)/2
                    ox = (img.shape[2]-96)/2
                    oy=oy/2+np.random.randint(oy)
                    ox=ox/2+np.random.randint(ox)

                    # optionally, mirror the image.
                    if rnd2==0:
                        img[:,:,:] = img[:,:,::-1]

                    x2[j,:,:,:] = (img[:,oy:oy+96,ox:ox+96]-128.0)/128.0
                except:
                    print 'read image error occured', fs[rnd]
            #print "load image done"
            
            # train generator
            z = Variable(xp.random.uniform(-1, 1, (batchsize, nz), dtype=np.float32))
            x = gen(z)
            yl = dis(x)

            L_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
            L_dis = F.softmax_cross_entropy(yl, Variable(xp.ones(batchsize, dtype=np.int32)))
            
            # train discriminator
                    
            x2 = Variable(cuda.to_gpu(x2))
            yl2 = dis(x2)
            L_dis += F.softmax_cross_entropy(yl2, Variable(xp.zeros(batchsize, dtype=np.int32)))
            
            #print "forward done"

            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()
            
            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()
            
            sum_l_gen += L_gen.data.get()
            sum_l_dis += L_dis.data.get()
            
            #print "backward done"
            
            #2nd round battle
            x.unchain_backward()
            x2.unchain_backward()

            x3=retou(x)       # let the retoucher make the generated image better
            yl1st = dis(x3)   # and try deceive the discriminator
            yl2nd = dis2(x3)  # and try deceive the discriminator2
            
            L_retou = F.softmax_cross_entropy(yl1st, Variable(xp.zeros(batchsize, dtype=np.int32)))
            L_retou += F.softmax_cross_entropy(yl2nd, Variable(xp.zeros(batchsize, dtype=np.int32)))
            L_dis2 = F.softmax_cross_entropy(yl2nd, Variable(xp.ones(batchsize, dtype=np.int32)))
            
            # train discriminator2 with the teacher images.
                    
            yl2 = dis2(x2)
            L_dis2 += F.softmax_cross_entropy(yl2, Variable(xp.zeros(batchsize, dtype=np.int32)))

            o_retou.zero_grads()
            L_retou.backward()
            o_retou.update()
            
            o_dis2.zero_grads()
            L_dis2.backward()
            o_dis2.update()


            if i%image_save_interval==0:
                plt.rcParams['figure.figsize'] = (16.0,32.0)
                plt.close('all')
                
                vissize = 100
                z = zvis
                z[50:,:] = (xp.random.uniform(-1, 1, (50, nz), dtype=np.float32))
                z = Variable(z)
                x = gen(z, test=True)
                x3 = retou(x, test=True)
                x = x.data.get()
                x3 = x3.data.get()
                imgfn = '%s/vis_%d_%d.png'%(out_image_dir, epoch,i)

                print imgfn

                def totitle(x):
                    return str(x)[0:6]

                for i_ in range(100):
                    tmp = ((np.vectorize(clip_img)(x[i_,:,:,:])+1)/2).transpose(1,2,0)
                    plt.subplot(20,10,i_+1)
                    plt.imshow(tmp)
                    plt.axis('off')
                    plt.title(totitle(z.data[i_,0]))
                for i_ in range(100):
                    tmp = ((np.vectorize(clip_img)(x3[i_,:,:,:])+1)/2).transpose(1,2,0)
                    plt.subplot(20,10,i_+101)
                    plt.imshow(tmp)
                    plt.axis('off')
                    plt.title(totitle(z.data[i_,0])+'kai')
                plt.suptitle(imgfn)
                plt.savefig(imgfn)

                subprocess.call("cp %s ~/public_html/dcgan-%d.png"%(imgfn,args.gpu),shell=True)
        serializers.save_hdf5("%s/dcgan_model_dis_%d.h5"%(out_model_dir, epoch),dis)
        serializers.save_hdf5("%s/dcgan_model_dis2_%d.h5"%(out_model_dir, epoch),dis2)
        serializers.save_hdf5("%s/dcgan_model_gen_%d.h5"%(out_model_dir, epoch),gen)
        serializers.save_hdf5("%s/dcgan_model_retou_%d.h5"%(out_model_dir, epoch),retou)
        serializers.save_hdf5("%s/dcgan_state_dis_%d.h5"%(out_model_dir, epoch),o_dis)
        serializers.save_hdf5("%s/dcgan_state_dis2_%d.h5"%(out_model_dir, epoch),o_dis2)
        serializers.save_hdf5("%s/dcgan_state_gen_%d.h5"%(out_model_dir, epoch),o_gen)
        serializers.save_hdf5("%s/dcgan_state_retou_%d.h5"%(out_model_dir, epoch),o_retou)
        print 'epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train



xp = cuda.cupy
cuda.get_device(int(args.gpu)).use()

gen = Generator()
retou = Retoucher()
dis = Discriminator()
dis2 = Discriminator2()
gen.to_gpu()
retou.to_gpu()
dis.to_gpu()
dis2.to_gpu()


try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
except:
    pass

train_dcgan_labeled(gen, retou, dis, dis2)
