# -*- coding: utf-8 -*-
"""
@author: Paul Xing
"""

import torch
import torch.fft
from torch.nn import functional as F
import numpy as np

## nextpow2
def nextpow2(a):
    b = np.ceil(np.log(a)/np.log(2))
    return b


def diff(x, dim=-1):
    if dim==-1:
        return F.pad(x[...,1:]-x[...,:-1], (1,0))

def unwrap(phi, dim=-1):
    dphi = diff(phi, dim)
    dphi_m = ((dphi+np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m==-np.pi)&(dphi>0)] = np.pi
    phi_adj = dphi_m-dphi
    phi_adj[dphi.abs()<np.pi] = 0
    return phi + phi_adj.cumsum(dim)



def sinc(input_array):
    output_array = torch.sin(np.pi*input_array)/(np.pi*input_array)
    output_array[input_array==0]=1
    return output_array



def BfIQFlatLinearTorch(In_real, In_im, pmig, device = 'cuda'):
    """
    Parameters
    ______________
    inputs order must be
        (Batch size, number of frames, number of transmit angles, fast times, number of elements)

    real and imaginary part must be separated


    Returns
    ______________
    outputs order is
        (Batch size, number of frames, Nz, Nx)
    """

    assert(In_real.size() == In_im.size())
    if len(In_real.shape) < 5:
        In_real = In_real.unsqueeze(1) #add the frame dimension
        In_im = In_im.unsqueeze(1) #add the frame dimension


    #fixed dimension order for torch library
    [Nbatch, Nframe, Ntx, NtIQ,  Nchannel] = In_real.shape

    if 'c' not in pmig:
        pmig['c'] = 1540
        print('A Propagation velocity must be set ')
        print(pmig['c'])
        print(' m/s has been set as default')

    if 'fc' not in pmig:
        pmig['fc'] = 3e6
        print('A Central frequency must be set ')
        print(pmig['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in pmig:
        pmig['lbd'] = pmig['c']/pmig['fc']

    if 'fs' not in pmig:
        pmig['fs'] = 4*pmig['fc']
        print('A Sampling frequency must be set ')
        print(pmig['fs'] /1e6)
        print('MHz has been set as default')

    if 'sub' not in pmig:
        pmig['sub'] = 2

    if 'fsIQ' not in pmig:
        pmig['fsIQ'] = pmig['fs']/pmig['sub']

    if 'pitch' not in pmig:
        pmig['pitch'] = pmig['lbd']

    if 'width' not in pmig:
        pmig['width'] = pmig['lbd']/2

    Aperture = (Nchannel-1)*pmig['pitch']
    X0 = Aperture/2

    if 'theta' not in pmig:
        pmig['theta'] = 0

    if 't0' not in pmig:
        pmig['t0'] = 0

    # Grid definition
    if 'dx' not in pmig:
        pmig['dx'] = pmig['pitch']

    if 'dz' not in pmig:
        pmig['dz'] = pmig['lbd']

    if 'Nx' not in pmig:
        pmig['Nx'] = Nchannel

    if 'Nz' not in pmig:
        pmig['Nz'] = Nchannel

    Npix = pmig['Nx']*pmig['Nz']

    if 'Xmin' not in pmig:
        pmig['Xmin'] = 0 -(pmig['Nx']-1)*pmig['dx']/2

    if 'Zmin' not in pmig:
        pmig['Zmin'] = 0

    if 'fnum' not in pmig:
        pmig['fnum'] = 1


    theta = torch.FloatTensor(pmig['theta'])
    invc  = 1/pmig['c']


    ## Linear Array definition
    xele = torch.linspace(0, Nchannel-1, steps = Nchannel, dtype=torch.float32)*pmig['pitch'] - X0
    xele = xele.view(1,1,Nchannel)
    yele = torch.zeros(1,1,Nchannel)
    zele = torch.zeros(1,1,Nchannel)


    ## Cartesian Grid definition
    xp = torch.linspace(0, pmig['Nx']-1, steps = pmig['Nx'], dtype=torch.float32)*pmig['dx'] + pmig['Xmin']
    zp = torch.linspace(0, pmig['Nz']-1, steps = pmig['Nz'], dtype=torch.float32)*pmig['dz'] + pmig['Zmin']
    xp, zp = torch.meshgrid(xp,zp)

    Zpix = torch.reshape(zp,(1,Npix))
    Xpix = torch.reshape(xp,(1,Npix))
    Ypix = torch.zeros(1,Npix)

    demoddelay  = (4*np.pi*pmig['fc']*Zpix*invc).permute(1,0)
    phaseRot    = 2*np.pi*pmig['fc']/pmig['fsIQ']

    r_e = torch.cat((xele, yele, zele), 0)
    r_p = torch.cat((Xpix, Ypix, Zpix), 0).unsqueeze(-1)



    #receive and transmit delay computation
    RXdelay = torch.sqrt(torch.sum( (r_p.expand(3,Npix,Nchannel) - r_e.expand(3,Npix,Nchannel) )**2,dim=0) )*invc
    RXdelay = RXdelay.unsqueeze(0).permute(1,2,0)

    TXdelay = (torch.matmul(torch.cos(torch.abs(theta)).unsqueeze(1),Zpix)*invc).permute(1,0)
    TXdelay = TXdelay + (torch.matmul(torch.sin(torch.abs(theta)).unsqueeze(1)*(theta.unsqueeze(1)>0).float(),(Xpix + X0))*invc).permute(1,0)
    TXdelay = TXdelay + (torch.matmul(torch.sin(torch.abs(theta)).unsqueeze(1)*(theta.unsqueeze(1)<0).float(),(X0 - Xpix))*invc).permute(1,0)
    TXdelay = TXdelay + pmig['t0']


    idex = (RXdelay.expand(Npix,Nchannel,Ntx) + TXdelay.unsqueeze(1).expand(Npix,Nchannel,Ntx))*pmig['fsIQ']
    deltaDelay = phaseRot*idex - demoddelay.unsqueeze(-1).expand(Npix,Nchannel,Ntx)

    ## Fnumber
    dist     = torch.abs(r_p[0,:,:].expand(Npix,Nchannel) - r_e[0,:,:].expand(Npix,Nchannel))

    fnummask = (2*dist*pmig['fnum']<r_p[2,:,:].expand(Npix,Nchannel)).unsqueeze(0).permute(1,2,0).expand(Npix,Nchannel,Ntx).float()
    fnummask = fnummask/torch.sum(fnummask,1).unsqueeze(1).expand(Npix,Nchannel,Ntx)

    #torch beamforming grid
    grid_idex=2*idex/NtIQ -1 #recentering index value for torch grid computation

    #channel grid
    grid_ch=torch.linspace(-1, 1, steps = Nchannel).view(1,Nchannel,1).expand(Npix,Nchannel,Ntx)

    #transmit grid
    grid_t=torch.linspace(-1, 1, steps = Ntx).view(1,1,Ntx).expand(Npix,Nchannel,Ntx)

    #final grid for interpolation
    grid=torch.stack((grid_ch, grid_idex, grid_t),3).unsqueeze(0).expand(Nbatch,Npix,Nchannel, Ntx,3)
    grid = grid.to(device)

    #interpolation with grid_sample
    amp_r =  F.grid_sample(In_real, grid = grid, align_corners=True).to(device)
    amp_i = F.grid_sample(In_im, grid = grid, align_corners=True).to(device)

    phase_r = (torch.cos(-deltaDelay)*fnummask).to(device)
    phase_i = (torch.sin(-deltaDelay)*fnummask).to(device)

    #summation over channels and transmit angles
    IQ_r  = torch.sum(torch.sum(amp_r*phase_r -  amp_i*phase_i,-1).to(device), -1).to(device)
    IQ_i  = torch.sum(torch.sum(amp_i*phase_r +  amp_r*phase_i,-1).to(device), -1).to(device)

    #permutation is equivalent to fortran-style reshape in numpy
    IQ_r = torch.reshape(IQ_r, (Nbatch, Nframe, pmig['Nx'] ,pmig['Nz'])).permute(0,1,3,2)
    IQ_i =torch.reshape(IQ_i,(Nbatch, Nframe, pmig['Nx'] ,pmig['Nz'])).permute(0,1,3,2)


    return IQ_r, IQ_i








#RX correction
def Torch_BeamformerRX(In_real, In_im, Law_real, Law_im, pmig, device='cuda'):
    """
    Parameters
    ______________
    inputs order must be
        (Batch size, number of frames, number of transmit angles, fast times, number of elements)

    real and imaginary part must be separated


    Returns
    ______________
    outputs order is
        (Batch size, number of frames, Nz, Nx)
    """

    assert(In_real.size() == In_im.size())
    assert(Law_real.size() == Law_im.size())



    assert(In_real.size() == In_im.size())
    if len(In_real.shape) < 5:
        In_real = In_real.unsqueeze(1) #add the frame dimension
        In_im = In_im.unsqueeze(1) #add the frame dimension


    #fixed dimension order for torch library
    [Nbatch, Nframe, Ntx, NtIQ,  Nchannel] = In_real.shape


    if 'c' not in pmig:
        pmig['c'] = 1540
        print('A Propagation velocity must be set ')
        print(pmig['c'])
        print(' m/s has been set as default')

    if 'fc' not in pmig:
        pmig['fc'] = 3e6
        print('A Central frequency must be set ')
        print(pmig['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in pmig:
        pmig['lbd'] = pmig['c']/pmig['fc']

    if 'fs' not in pmig:
        pmig['fs'] = 4*pmig['fc']
        print('A Sampling frequency must be set ')
        print(pmig['fs'] /1e6)
        print('MHz has been set as default')

    if 'sub' not in pmig:
        pmig['sub'] = 2

    if 'fsIQ' not in pmig:
        pmig['fsIQ'] = pmig['fs']/pmig['sub']

    if 'pitch' not in pmig:
        pmig['pitch'] = pmig['lbd']

    if 'width' not in pmig:
        pmig['width'] = pmig['lbd']/2

    Aperture = (Nchannel-1)*pmig['pitch']
    X0 = Aperture/2

    if 'theta' not in pmig:
        pmig['theta'] = 0

    if 't0' not in pmig:
        pmig['t0'] = 0

    # Grid definition
    if 'dx' not in pmig:
        pmig['dx'] = pmig['pitch']

    if 'dz' not in pmig:
        pmig['dz'] = pmig['lbd']

    if 'Nx' not in pmig:
        pmig['Nx'] = Nchannel

    if 'Nz' not in pmig:
        pmig['Nz'] = Nchannel

    Npix = pmig['Nx']*pmig['Nz']

    if 'Xmin' not in pmig:
        pmig['Xmin'] = 0 -(pmig['Nx']-1)*pmig['dx']/2

    if 'Zmin' not in pmig:
        pmig['Zmin'] = 0

    if 'fnum' not in pmig:
        pmig['fnum'] = 1


    theta = torch.FloatTensor(pmig['theta'])
    invc  = 1/pmig['c']


    ## Linear Array definition
    xele = torch.linspace(0, Nchannel-1, steps = Nchannel)*pmig['pitch'] - X0
    xele = xele.view(1,1,Nchannel)
    yele = torch.zeros(1,1,Nchannel)
    zele = torch.zeros(1,1,Nchannel)


    ## Cartesian Grid definition
    xp = torch.linspace(0, pmig['Nx']-1, steps = pmig['Nx'])*pmig['dx'] + pmig['Xmin']
    zp = torch.linspace(0, pmig['Nz']-1, steps = pmig['Nz'])*pmig['dz'] + pmig['Zmin']
    xp, zp = torch.meshgrid(xp,zp)

    Zpix = torch.reshape(zp,(1,Npix))
    Xpix = torch.reshape(xp,(1,Npix))
    Ypix = torch.zeros(1,Npix)

    demoddelay  = (4*np.pi*pmig['fc']*Zpix*invc).permute(1,0)
    phaseRot    = 2*np.pi*pmig['fc']/pmig['fsIQ']

    r_e = torch.cat((xele, yele, zele), 0)
    r_p = torch.cat((Xpix, Ypix, Zpix), 0).unsqueeze(-1)


    Law_amp2 = torch.sqrt(Law_real**2+Law_im**2)
    Law_delay =torch.atan2(Law_im, Law_real)
    Law_delay = unwrap(Law_delay)/(2*np.pi*pmig['fc'])

    Law_amp2 =  Law_amp2.view(Nbatch,1,Nchannel,1)
    Law_delay = Law_delay.view(Nbatch,1,Nchannel,1)



    #receive and transmit delay computation
    #RXdelay = torch.sqrt(torch.sum( (r_p.repeat(1,1,Nchannel) - r_e.repeat(1,Npix,1) )**2,dim=0) )*invc
    RXdelay = torch.sqrt(torch.sum( (r_p.expand(3,Npix,Nchannel) - r_e.expand(3,Npix,Nchannel) )**2,dim=0) )*invc
    RXdelay = RXdelay.unsqueeze(0).permute(1,2,0)
    RXdelay = RXdelay.unsqueeze(0).expand(Nbatch,Npix,Nchannel,Ntx)
    RXdelay = RXdelay + Law_delay.expand(Nbatch,Npix,Nchannel,Ntx) #delay correction in receive


    TXdelay = (torch.matmul(torch.cos(torch.abs(theta)).unsqueeze(1),Zpix)*invc).permute(1,0)
    TXdelay = TXdelay + (torch.matmul(torch.sin(torch.abs(theta)).unsqueeze(1)*(theta.unsqueeze(1)>0).float(),(Xpix + X0))*invc).permute(1,0)
    TXdelay = TXdelay + (torch.matmul(torch.sin(torch.abs(theta)).unsqueeze(1)*(theta.unsqueeze(1)<0).float(),(X0 - Xpix))*invc).permute(1,0)
    TXdelay = TXdelay + pmig['t0']

    TXdelay = TXdelay.unsqueeze(1).unsqueeze(0).expand(Nbatch, Npix,Nchannel,Ntx)

    #TXdelay = TXdelay# + Law_delay as function of theta




    #idex = (RXdelay.repeat(1,1,Ntx) + TXdelay.unsqueeze(1).repeat(1,Nchannel,1))*pmig['fsIQ']
    idex = (RXdelay + TXdelay)*pmig['fsIQ']

    #deltaDelay = phaseRot*idex - demoddelay.unsqueeze(-1).repeat(1,Nchannel,Ntx)
    deltaDelay = phaseRot*idex- demoddelay.unsqueeze(-1).expand(Npix,Nchannel,Ntx)

    ## Fnumber
    #dist     = torch.abs(r_p[0,:,:].repeat(1,Nchannel) - r_e[0,:,:].repeat(Npix,1))
    dist     = torch.abs(r_p[0,:,:].expand(Npix,Nchannel) - r_e[0,:,:].expand(Npix,Nchannel))

    #fnummask = (2*dist*pmig['fnum']<r_p[2,:,:].repeat(1,Nchannel)).unsqueeze(0).permute(1,2,0).repeat(1,1,Ntx).float()
    fnummask = (2*dist*pmig['fnum']<r_p[2,:,:].expand(Npix,Nchannel)).unsqueeze(0).permute(1,2,0).expand(Npix,Nchannel,Ntx).float()
    #fnummask = fnummask/torch.sum(fnummask,1).unsqueeze(1).repeat(1,Nchannel,1)
    fnummask = fnummask/torch.sum(fnummask,1).unsqueeze(1).expand(Npix,Nchannel,Ntx)



    #torch beamforming grid
    grid_idex=(2*idex/NtIQ -1) #recentering index value for torch grid computation

    #channel grid
    grid_ch=torch.linspace(-1, 1, steps = Nchannel).view(1,Nchannel,1).expand(Nbatch,Npix,Nchannel,Ntx)

    #transmit grid
    grid_t=torch.linspace(-1, 1, steps = Ntx).view(1,1,Ntx).expand(Nbatch,Npix,Nchannel,Ntx)

    #final grid for interpolation
    grid=torch.stack((grid_ch, grid_idex, grid_t),-1)
    grid=grid.to(device)


    #interpolation with grid_sample
    amp_r =  F.grid_sample(In_real, grid = grid, align_corners=True).to(device)
    amp_i = F.grid_sample(In_im, grid = grid, align_corners=True).to(device)


    #phase_r = (torch.cos(-deltaDelay)*fnummask*(1/Law_amp2).expand(Nbatch,Npix,Nchannel,Ntx)).to(device)
    #phase_i = (torch.sin(-deltaDelay)*fnummask*(1/Law_amp2).expand(Nbatch,Npix,Nchannel,Ntx)).to(device)
    phase_r = (torch.cos(-deltaDelay)*fnummask).to(device)
    phase_i = (torch.sin(-deltaDelay)*fnummask).to(device)


    #summation over channels and transmit angles
    IQ_r  = torch.sum(torch.sum(amp_r*phase_r -  amp_i*phase_i,-1).to(device),-1).to(device)
    IQ_i  = torch.sum(torch.sum(amp_i*phase_r +  amp_r*phase_i,-1).to(device),-1).to(device)


    #permutation is equivalent to fortran-style reshape in numpy
    IQ_r = torch.reshape(IQ_r, (Nbatch, Nframe, pmig['Nx'] ,pmig['Nz'])).permute(0,1,3,2)
    IQ_i =torch.reshape(IQ_i,(Nbatch, Nframe, pmig['Nx'] ,pmig['Nz'])).permute(0,1,3,2)



    return IQ_r, IQ_i



def BfIQFlatLinearMtxTorch(Raw_shape, pmig):

    if len(Raw_shape) < 5:
        Nframe = 1
        [Nbatch, Ntx, NtIQ,  Nchannel] = Raw_shape
    else:
        #fixed dimension order for torch library
        [Nbatch, Nframe, Ntx, NtIQ,  Nchannel] = Raw_shape

    if 'c' not in pmig:
        pmig['c'] = 1540
        print('A Propagation velocity must be set ')
        print(pmig['c'])
        print(' m/s has been set as default')

    if 'fc' not in pmig:
        pmig['fc'] = 3e6
        print('A Central frequency must be set ')
        print(pmig['fc']/1e6)
        print(' MHz has been set as default')

    if 'lbd' not in pmig:
        pmig['lbd'] = pmig['c']/pmig['fc']

    if 'fs' not in pmig:
        pmig['fs'] = 4*pmig['fc']
        print('A Sampling frequency must be set ')
        print(pmig['fs'] /1e6)
        print('MHz has been set as default')

    if 'sub' not in pmig:
        pmig['sub'] = 2

    if 'fsIQ' not in pmig:
        pmig['fsIQ'] = pmig['fs']/pmig['sub']

    if 'pitch' not in pmig:
        pmig['pitch'] = pmig['lbd']

    if 'width' not in pmig:
        pmig['width'] = pmig['lbd']/2

    Aperture = (Nchannel-1)*pmig['pitch']
    X0 = Aperture/2

    if 'theta' not in pmig:
        pmig['theta'] = 0

    if 't0' not in pmig:
        pmig['t0'] = 0

    # Grid definition
    if 'dx' not in pmig:
        pmig['dx'] = pmig['pitch']

    if 'dz' not in pmig:
        pmig['dz'] = pmig['lbd']

    if 'Nx' not in pmig:
        pmig['Nx'] = Nchannel

    if 'Nz' not in pmig:
        pmig['Nz'] = Nchannel

    if 'batchsize' not in pmig:
        pmig['batchsize'] = 8

    Npix            = pmig['Nx']*pmig['Nz']
    batchsize       = pmig['batchsize']
    NpixPerBatch    = int(Npix/batchsize)



    if 'Xmin' not in pmig:
        pmig['Xmin'] = 0 -(pmig['Nx']-1)*pmig['dx']/2

    if 'Zmin' not in pmig:
        pmig['Zmin'] = 0

    if 'fnum' not in pmig:
        pmig['fnum'] = 1


    theta = torch.FloatTensor(pmig['theta'])
    invc  = 1/pmig['c']


    ## Linear Array definition
    xele = torch.linspace(0, Nchannel-1, steps = Nchannel)*pmig['pitch'] - X0
    xele = xele.view(1,1,Nchannel)
    yele = torch.zeros(1,1,Nchannel)
    zele = torch.zeros(1,1,Nchannel)

    r_e = torch.cat((xele, yele, zele), 0)

    idxCh   = torch.linspace(0, Nchannel-1, Nchannel, dtype=torch.float32).view(1,Nchannel)
    idxTx   = torch.linspace(0, Ntx-1, Ntx, dtype=torch.float32).view(1,1,Ntx)

    ##
    Ninterp     = pmig['order']*2+1
    Ndata       = NtIQ * Nchannel * Ntx
    NNz         = Ninterp*NpixPerBatch*Nchannel*Ntx # Non-Zeros values


    ## Cartesian Grid definition
    xp = torch.linspace(0, pmig['Nx']-1, steps = pmig['Nx'], dtype=torch.float32)*pmig['dx'] + pmig['Xmin']
    zp = torch.linspace(0, pmig['Nz']-1, steps = pmig['Nz'], dtype=torch.float32)*pmig['dz'] + pmig['Zmin']
    xp, zp = torch.meshgrid(xp,zp)

    Zpix = torch.reshape(zp,(1, NpixPerBatch,batchsize))
    Xpix = torch.reshape(xp,(1, NpixPerBatch,batchsize))
    Ypix = torch.zeros(1,NpixPerBatch,batchsize)

    idxPix = torch.linspace(0,NpixPerBatch-1,NpixPerBatch,dtype=torch.float32).unsqueeze(-1)

    idxPix = idxPix.expand(NpixPerBatch, Nchannel*Ntx)
    idxPix = torch.reshape(idxPix, (NpixPerBatch*Nchannel*Ntx,1))
    idxPix = idxPix.expand(NpixPerBatch*Nchannel*Ntx, Ninterp)

    phaseRot    = 2*np.pi*pmig['fc']/pmig['fsIQ']

    ## Memory allocation
    idxRow      = torch.empty( NNz , batchsize, dtype = torch.long)
    idxCol      = torch.empty( NNz , batchsize, dtype = torch.long)
    values_r      = torch.empty( NNz , batchsize, dtype = torch.float32)
    values_i      = torch.empty( NNz , batchsize, dtype = torch.float32)



    for ibatch in range(batchsize):

        x_p_batch  = Xpix[:,:,ibatch]
        y_p_batch  = Ypix[:,:,ibatch]
        z_p_batch  = Zpix[:,:,ibatch]

        r_p_batch = torch.cat((x_p_batch, y_p_batch, z_p_batch), 0).unsqueeze(-1)

        ## Time of flight
        demoddelay  = (4*np.pi*pmig['fc']*r_p_batch[2,:]*invc)

        #receive and transmit delay computation
        RXdelay = torch.sqrt(torch.sum( (r_p_batch.expand(3,NpixPerBatch,Nchannel) - r_e.expand(3,NpixPerBatch,Nchannel) )**2,dim=0) )*invc
        RXdelay = RXdelay.unsqueeze(0).permute(1,2,0)

        TXdelay = (torch.matmul(torch.cos(torch.abs(theta)).unsqueeze(1),z_p_batch)*invc).permute(1,0)
        TXdelay = TXdelay + (torch.matmul(torch.sin(torch.abs(theta)).unsqueeze(1)*(theta.unsqueeze(1)>0).float(),(x_p_batch + X0))*invc).permute(1,0)
        TXdelay = TXdelay + (torch.matmul(torch.sin(torch.abs(theta)).unsqueeze(1)*(theta.unsqueeze(1)<0).float(),(X0 - x_p_batch))*invc).permute(1,0)
        TXdelay = TXdelay + pmig['t0']



        idxt = (RXdelay.expand(NpixPerBatch,Nchannel,Ntx) + TXdelay.unsqueeze(1).expand(NpixPerBatch,Nchannel,Ntx))*pmig['fsIQ']
        deltaDelay = phaseRot*idxt - demoddelay.unsqueeze(-1).expand(NpixPerBatch,Nchannel,Ntx)

        del demoddelay
        ## Fnumber
        ApertureF   = (r_p_batch[2,:,:]/pmig['fnum']).expand(NpixPerBatch, Nchannel)

        D_e         = torch.sqrt((r_p_batch[0,:].expand(NpixPerBatch,Nchannel) - r_e[0,:].expand(NpixPerBatch, Nchannel) )**2 )

        ApodFnum    = (torch.cos(np.pi * D_e/ApertureF/2)**2)  * ( D_e/ApertureF/2 < 1/2)
        ApodFnum    = ApodFnum/(1 + torch.sum(ApodFnum, 1).unsqueeze(-1).expand( NpixPerBatch, Nchannel)) /Nchannel
        ApodFnum    = ApodFnum.unsqueeze(-1).expand(NpixPerBatch,Nchannel,Ntx)

        ApodFnum = ApodFnum.permute(1,2,0)

        del D_e


        ## Delay To Index & Weight
        I           = (idxt<(pmig['order']+1)) + (idxt>( NtIQ - pmig['order'] - 1))
        idxt[I]     = NtIQ - pmig['order'] - 1
        idxt        = idxt + NtIQ * idxCh.unsqueeze(-1).expand(NpixPerBatch,Nchannel,Ntx) + NtIQ * Nchannel * idxTx.expand(NpixPerBatch,Nchannel,Ntx)

        idxt        = torch.reshape(idxt,(NpixPerBatch*Nchannel*Ntx,))        # [NpixPerBatch*Nchannel*Ntx]

        deltaDelay  = torch.reshape(deltaDelay,(NpixPerBatch*Nchannel*Ntx,))  # [NpixPerBatch*Nchannel*Ntx]
        ApodFnum    = torch.reshape(ApodFnum,(NpixPerBatch*Nchannel*Ntx,))    # [NpixPerBatch*Nchannel*Ntx]
        I           = torch.reshape(I,(NpixPerBatch*Nchannel*Ntx,))           # [NpixPerBatch*Nchannel*Ntx]

        ## DASmtx coefficients
        idxtf       = idxt.unsqueeze(-1).expand(NpixPerBatch*Nchannel*Ntx,Ninterp)
        idxtf       = idxtf + torch.linspace(0,Ninterp-1, Ninterp).unsqueeze(0).expand(NpixPerBatch*Nchannel*Ntx, Ninterp)
        weight      = sinc( idxtf - idxt.unsqueeze(-1).expand(NpixPerBatch*Nchannel*Ntx, Ninterp) )

        weight_r      = (ApodFnum * torch.cos( -deltaDelay )).unsqueeze(-1).expand(NpixPerBatch*Nchannel*Ntx, Ninterp ) * weight
        weight_i      = (ApodFnum * torch.sin( -deltaDelay )).unsqueeze(-1).expand(NpixPerBatch*Nchannel*Ntx, Ninterp ) * weight

        del weight


        idxRow[:,ibatch] = torch.reshape( idxPix , (NNz,) )
        idxCol[:,ibatch] = torch.reshape( idxtf  , (NNz,) )

        values_r[:,ibatch] = torch.reshape( weight_r , (NNz,) )
        values_i[:,ibatch] = torch.reshape( weight_i , (NNz,) )


    return idxRow, idxCol, values_r, values_i, pmig


def DoBfTorch(In_real, In_im, idxRow, idxCol, values_r, values_i , pmig, device='cuda'):
    assert(In_real.size() == In_im.size())
    if len(In_real.shape) < 5:
        In_real = In_real.unsqueeze(1) #add the frame dimension
        In_im = In_im.unsqueeze(1) #add the frame dimension


    #fixed dimension order for torch library
    [Nbatch, Nframe, Ntx, NtIQ,  Nchannel] = In_real.shape



    In_real    = torch.reshape(In_real.permute(0,1,2,-1,3), (Nbatch,-1, Nframe)) #permute to matche reshape index order
    In_im      = torch.reshape(In_im.permute(0,1,2,-1,3), (Nbatch,-1, Nframe))

    Npix    = pmig['Nz']*pmig['Nx']
    NpixPerBatch = int(Npix/pmig['batchsize'])
    Ndata = In_real.shape[1]

    IQ_r   = torch.empty(Nbatch, NpixPerBatch,pmig['batchsize'], Nframe , dtype = torch.float32)
    IQ_i   = torch.empty(Nbatch, NpixPerBatch,pmig['batchsize'], Nframe , dtype = torch.float32)


    for batch in range(Nbatch):
        for ibatch in range(pmig['batchsize']):
            DASmtx_r = torch.sparse.FloatTensor(torch.stack([idxRow[:,ibatch],idxCol[:,ibatch]]), values_r[:, ibatch], torch.Size([NpixPerBatch,Ndata])).to(device)
            DASmtx_i = torch.sparse.FloatTensor(torch.stack([idxRow[:,ibatch],idxCol[:,ibatch]]), values_i[:, ibatch], torch.Size([NpixPerBatch,Ndata])).to(device)


            IQ_r[batch,:,ibatch,:]  = torch.mm(DASmtx_r , In_real[batch,:]) - torch.mm(DASmtx_i , In_im[batch,:])
            IQ_i[batch,:,ibatch,:]  = torch.mm(DASmtx_r , In_im[batch,:]) + torch.mm(DASmtx_i , In_real[batch,:])


    IQ_r = torch.reshape(IQ_r, (Nbatch, Nframe, pmig['Nx'] ,pmig['Nz'])).permute(0,1,3,2)
    IQ_i =torch.reshape(IQ_i,(Nbatch, Nframe, pmig['Nx'] ,pmig['Nz'])).permute(0,1,3,2)

    return IQ_r, IQ_i
