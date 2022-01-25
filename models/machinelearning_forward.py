import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os, glob, cv2
from diffuser_utils import *

class Forward_Model(torch.nn.Module):
    # Model initialization 
    def __init__(self, mask,               # mask: response matrix for the spectral filter array
                 num_ims = 2,              # number of blurred images to simulate
                 w_init = None,            # initialization for the blur kernels e.g., w_init = [.01, 0.01]
                 cuda_device = 0,          # cuda device parameter 
                 blur_type = 'symmetric',  # symmetric or asymmetric blur kernel 
                 optimize_blur = False):   # choose whether to learn the best blur or not (warning, not stable)
        super(Forward_Model, self).__init__()
        
        self.cuda_device = cuda_device
        self.blur_type  = blur_type
       
    
         ## Initialize constants
        self.DIMS0 = mask.shape[0]  # Image Dimensions
        self.DIMS1 = mask.shape[1]  # Image Dimensions
        self.PAD_SIZE0 = int((self.DIMS0))     # Pad size
        self.PAD_SIZE1 = int((self.DIMS1))     # Pad size
        
        self.num_ims = num_ims
        
        
        
        if w_init is None: #if no blur specified, use default
            if self.blur_type == 'symmetric':
                w_init = np.linspace(.002, .065, num_ims)
            else:
                w_init = np.linspace(.002, .1, num_ims)
                w_init =  np.repeat(np.array(w_init)[np.newaxis], self.num_ims, axis = 0).T
                w_init[:,1] *=.5
             
        print('optimize blur', optimize_blur)
        if self.blur_type == 'symmetric':
            #self.w_init =  np.repeat(np.array(w_init[0])[np.newaxis], self.num_ims, axis = 0)
            self.w_init = w_init
            self.w_blur = torch.nn.Parameter(torch.tensor(self.w_init, dtype = torch.float32, 
                                                          device = self.cuda_device,
                                                         requires_grad = optimize_blur))
        else:
            self.w_init =  w_init
            self.w_blur = torch.nn.Parameter(torch.tensor(self.w_init, dtype = torch.float32, 
                                                          device = self.cuda_device,
                                                         requires_grad = optimize_blur))

        # set up grid 
        x=np.linspace(-1,1,self.DIMS0); 
        y=np.linspace(-1,1,self.DIMS1); 
        X,Y=np.meshgrid(x,y)
        
        self.X = torch.tensor(X, dtype=torch.float32, device=self.cuda_device)
        self.Y = torch.tensor(Y, dtype=torch.float32, device=self.cuda_device)
        
        self.mask = np.transpose(mask, (2,0,1))
        self.mask_var = torch.tensor(self.mask, dtype=torch.float32, device=self.cuda_device).unsqueeze(0)
        self.psf = np.empty((num_ims, self.DIMS0, self.DIMS1))
        
    # Forward model for blur
    def Hfor(self, i):
        if self.blur_type == 'symmetric':
            psf= torch.exp(-((self.X/self.w_blur[i])**2+(self.Y/self.w_blur[i])**2))
        else:
            psf= torch.exp(-((self.X/self.w_blur[i,0])**2+(self.Y/self.w_blur[i,1])**2))
            
        #psf = psf/torch.linalg.norm(psf, ord=float('inf'))
        
        self.psf[i] = psf.detach().cpu().numpy()
        
        h_complex = pad_zeros_torch(self,torch.complex(psf,torch.zeros_like(psf)).unsqueeze(0))
        H = torch.fft.fft2(ifftshift2d(h_complex), norm = 'ortho') # ---------try ortho everywhere

        HX = H*self.Xi[:, i]
        out = torch.fft.ifft2(HX)
        out_r= out.real
        
        return out_r
    
    
    #calculates the adjoint of the simulated measurements:
    #   sim_meas contains num_ims simulated meas' in the form [1, num_ims, meassize1, meassize2]
    #   psf is the psf of the model
    #   see https://waller-lab.github.io/DiffuserCam/tutorial/algorithm_guide.pdf
    #   and https://waller-lab.github.io/DiffuserCam/tutorial/GD.html
    def Hadj(self, sim_meas, psf, num_ims):
        s0, s1 = self.PAD_SIZE0//2, self.PAD_SIZE1//2
        pad = (s0, s0, s1, s1)
        sim_meas = torch.unsqueeze(sim_meas, 2)
        #print('sim_meas shape:', sim_meas.shape)
        sm = sim_meas * self.mask_var
        sm = F.pad(sm, pad, 'constant', 0)
        smc = torch.complex(sm, torch.zeros_like(sm))
        smfft = torch.fft.fft2(smc, dim = (-2, -1), norm = 'ortho')
        #print('2:',smfft.shape)
        
        psf = torch.tensor(psf).to(self.cuda_device)
        h_complex = pad_zeros_torch(self,torch.complex(psf,torch.zeros_like(psf)).unsqueeze(0))
        Hconj = torch.unsqueeze(torch.conj(torch.fft.fft2(ifftshift2d(h_complex), norm = 'ortho')), 2)
        
        adj_meas = torch.fft.ifft2(Hconj*smfft, dim = (-2, -1), norm = 'ortho').real
        #print('adj_meas shape:', adj_meas.shape)
        return adj_meas#.float()
    
    # forward call for the model
    def forward(self, in_image):
        #x = my_pad(self, in_image)  
        x = in_image # pad image
        xc = torch.complex(x, torch.zeros_like(x))  # make into a complex number (needed for FFT)
        self.Xi = torch.fft.fft2(xc, norm = 'ortho')                # Take FFT of image 

        # Generate simulated images for each blur amount 
        out_list = []
        for i in range(0,self.num_ims):
            output = self.mask_var * crop_forward(self,  self.Hfor(i))
            
            output = torch.sum(output, 1)
            
            #output= output/torch.max(output)
            out_list.append(output)
            
        final_output = torch.stack(out_list, 1)
        print(final_output.shape)
        #final_output_hadj = self.Hadj(final_output, self.psf, self.num_ims)

        return final_output#_hadj