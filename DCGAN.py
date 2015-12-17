import pickle,argparse,os,subprocess,sys
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
parser.add_argument('--fresh-start', '-f', action='store_true',
                    help='Start simulation anew')
args = parser.parse_args()


image_dir = './images'
out_image_dir = './out_images_%s'%(args.gpu)
out_model_dir = './out_models_%s'%(args.gpu)


subprocess.call("mkdir -p %s "%(out_image_dir),shell=True)
subprocess.call("mkdir -p %s "%(out_model_dir),shell=True)


nz = 100          # # of dim for Z
batchsize=25
n_epoch=10000
n_train=200000
image_save_interval =20000

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
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            c4 = L.Convolution2D(512, 1024, 6, stride=1, pad=0, wscale=0.02*math.sqrt(4*4*512)),
            dc4 = L.Deconvolution2D(1024,512, 6, stride=1, pad=0, wscale=0.02*math.sqrt(4*4*1024)),
            dc3 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc1 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc0 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
            bn4 = L.BatchNormalization(1024)
        )
        
    def __call__(self, x, test=False):
        h48 = elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h24 = elu(self.bn1(self.c1(h48), test=test))
        h12 = elu(self.bn2(self.c2(h24), test=test))
        h6 = elu(self.bn3(self.c3(h12), test=test))
        h1 = elu(self.bn4(self.c4(h6), test=test))
        
        h = h6 + 1e-1*F.relu(self.bn3(self.dc4(h1), test=test))
        h = h12 + 1e-1*F.relu(self.bn2(self.dc3(h), test=test))
        h = h24 + 1e-1*F.relu(self.bn1(self.dc2(h), test=test))
        h = h48 + 1e-1*F.relu(self.bn0(self.dc1(h), test=test))
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






def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

def load_dataset():
    x2 = np.zeros((batchsize, 3, 96, 96), dtype=np.float32)

    for j in range(batchsize):
        while True:
            try:
                rnd = np.random.randint(len(dataset))
                rnd2 = np.random.randint(2)
                
                img=Image.open(StringIO(dataset[rnd])).convert('RGB')
                img=img.rotate(np.random.random()*10.0-5.0, Image.BICUBIC)
                w,h=img.size
                scale = 120.0/min(w,h)*(1.0+0.2*np.random.random())
                img=img.resize((int(w*scale),int(h*scale)),Image.BICUBIC)

                img = np.asarray(img).astype(np.float32).transpose(2, 0, 1)
                # offset the image about the center of the image.
                oy = (img.shape[1]-96)/2
                ox = (img.shape[2]-96)/2
                oy=oy/2+np.random.randint(oy)
                ox=ox/2+np.random.randint(ox)
    
                # optionally, mirror the image.
                if rnd2==0:
                    img[:,:,:] = img[:,:,::-1]
    
                x2[j,:,:,:] = (img[:,oy:oy+96,ox:ox+96]-128.0)/128.0
                break
            except:
                print 'read image error occured', fs[rnd]
    return x2


def train_dcgan_labeled(gen, retou, dis, epoch0=0):
    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_retou = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen.setup(gen)
    o_retou.setup(retou)
    o_dis.setup(dis)
    if not args.fresh_start:
        serializers.load_hdf5("%s/dcgan_model_dis.h5"%(out_model_dir),dis)
        serializers.load_hdf5("%s/dcgan_model_gen.h5"%(out_model_dir),gen)
        serializers.load_hdf5("%s/dcgan_model_retou.h5"%(out_model_dir),retou)
        serializers.load_hdf5("%s/dcgan_state_dis.h5"%(out_model_dir),o_dis)
        serializers.load_hdf5("%s/dcgan_state_gen.h5"%(out_model_dir),o_gen)
        serializers.load_hdf5("%s/dcgan_state_retou.h5"%(out_model_dir),o_retou)


    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_retou.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    zvis = (xp.random.uniform(-1, 1, (100, nz), dtype=np.float32))

    x_retouch_motif = None
    retouch_fail_count = 0
    last_retouch_loss = 1.2e99
    
    for epoch in xrange(epoch0,n_epoch):
        print "epoch:", epoch
        perm = np.random.permutation(n_train)
        sum_l_dis = np.float32(0)
        sum_l_gen = np.float32(0)
        
        for i in xrange(0, n_train, batchsize):
            print "train:",i
            # discriminator
            # 0: from dataset
            # 1: from noise

            #print "load image start ", i
            x_train = load_dataset()
            #print "load image done"
            
            # train generator
            z = Variable(xp.random.uniform(-1, 1, (batchsize, nz), dtype=np.float32))
            x = gen(z)
            yl = dis(x)

            L_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
            L_dis = F.softmax_cross_entropy(yl, Variable(xp.ones(batchsize, dtype=np.int32)))
            
            # train discriminator
            x_train = Variable(cuda.to_gpu(x_train))
            yl_train = dis(x_train)

            softmax_gen = F.softmax(yl).data[:,0]
            average_softmax=np.average(cuda.to_cpu(softmax_gen))
            if average_softmax < 1e-3:
                train_sample_factor = 10.0
            elif average_softmax < 1e-2:
                train_sample_factor = 4.0
            elif average_softmax > 0.4:
                train_sample_factor = 1.0
            else:
                train_sample_factor = 2.0

            L_dis += train_sample_factor * F.softmax_cross_entropy(yl_train, Variable(xp.zeros(batchsize, dtype=np.int32)))
            
                
            

            #train retoucher
            if type(x_retouch_motif)==type(None) or retouch_fail_count >= min(1+ epoch, 10):
                print "Supply new motifs to retoucher."
                x_retouch_motif = Variable(x.data)
                retouch_fail_count = 0
                last_retouch_loss = 99e99

            x3=retou(x_retouch_motif)  # let the retoucher make the generated image better
            yl1st = dis(x3)   # and try deceive the discriminator
            
            # retoucher want their image to look like those from dataset(zeros), 
            # while discriminators want to classify them as from noise(ones)
            L_retou = F.softmax_cross_entropy(yl1st, Variable(xp.zeros(batchsize, dtype=np.int32)))
            L_dis  += F.softmax_cross_entropy(yl1st, Variable(xp.ones(batchsize, dtype=np.int32)))
            
    
            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()
            
            o_retou.zero_grads()
            L_retou.backward()
            o_retou.update()
            
            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()


            retouch_loss = float(str((L_retou).data))
            if retouch_loss >= last_retouch_loss:
                retouch_fail_count += 1
            last_retouch_loss = min(retouch_loss,last_retouch_loss)
            
            #print "backward done"

            sum_l_gen += L_gen.data.get()
            sum_l_dis += L_dis.data.get()
            
            x.unchain_backward()
            x_train.unchain_backward()
            x3.unchain_backward()
            x_retouch_motif = x3
            
            L_gen.unchain_backward()
            L_retou.unchain_backward()
            L_dis.unchain_backward()



            print "epoch:",epoch,"iter:",i,"softmax:",average_softmax, "retouch:",retouch_fail_count, retouch_loss

            if i%image_save_interval==0:
                plt.rcParams['figure.figsize'] = (16.0,64.0)
                plt.close('all')
                
                vissize = 100
                z = zvis
                z[50:,:] = (xp.random.uniform(-1, 1, (50, nz), dtype=np.float32))
                z = Variable(z)
                x = gen(z, test=True)
                x_data = x.data.get()
                imgfn = '%s/vis_%d_%d.png'%(out_image_dir, epoch,i)

                x_split = F.split_axis(x,vissize,0)


                def mktitle(x1):
                    d1 =  F.softmax(dis(x1,test=True))
                    def ppr(d):
                        f = float(str(d.data[0,0]))
                        return '{:0.3}'.format(f)
                    ret = '{}'.format(ppr(d1))
                    return ret

                for i_ in range(100):
                    tmp = ((np.vectorize(clip_img)(x_data[i_,:,:,:])+1)/2).transpose(1,2,0)
                    plt.subplot(43,10,i_+1)
                    plt.imshow(tmp)
                    plt.axis('off')
                    plt.title(mktitle(x_split[i_]),fontsize=6)

                r_p_cnt = 0
                print "vis-retouch:",
                for cnt in [1,1,1]:
                    r_p_cnt+=1
                    for r_cnt in range(cnt):
                        print r_cnt,
                        sys.stdout.flush()
                        x.unchain_backward()
                        x = retou(x, test=True)
                    x3_data = x.data.get()
                    x3_split = F.split_axis(x,vissize,0)
                    
                    for i_ in range(100):
                        tmp = ((np.vectorize(clip_img)(x3_data[i_,:,:,:])+1)/2).transpose(1,2,0)
                        plt.subplot(43,10,i_+1+110*r_p_cnt)
                        plt.imshow(tmp)
                        plt.axis('off')
                        plt.title(mktitle(x3_split[i_]),fontsize=6)
                plt.suptitle(imgfn)
                plt.savefig(imgfn)
                print imgfn

                subprocess.call("cp %s ~/public_html/dcgan-%d.png"%(imgfn,args.gpu),shell=True)

                serializers.save_hdf5("%s/dcgan_model_dis.h5"%(out_model_dir),dis)
                serializers.save_hdf5("%s/dcgan_model_gen.h5"%(out_model_dir),gen)
                serializers.save_hdf5("%s/dcgan_model_retou.h5"%(out_model_dir),retou)
                serializers.save_hdf5("%s/dcgan_state_dis.h5"%(out_model_dir),o_dis)
                serializers.save_hdf5("%s/dcgan_state_gen.h5"%(out_model_dir),o_gen)
                serializers.save_hdf5("%s/dcgan_state_retou.h5"%(out_model_dir),o_retou)
                
                # we don't have enough disk for history
                #history_dir = 'history/%d-%d'%(epoch,  i)
                #subprocess.call("mkdir -p %s "%(history_dir),shell=True)
                #subprocess.call("cp %s/*.h5 %s "%(out_model_dir,history_dir),shell=True)


        print 'epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train



xp = cuda.cupy
cuda.get_device(int(args.gpu)).use()

gen = Generator()
retou = Retoucher()
dis = Discriminator()
gen.to_gpu()
retou.to_gpu()
dis.to_gpu()


try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
except:
    pass

train_dcgan_labeled(gen, retou, dis)
