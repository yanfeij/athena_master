#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('Agg')

import sys
sys.settrace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pylab import *
import struct
import array
import os
import glob
import h5py
from scipy.interpolate import griddata


# In[2]:


from ReduceSimpleData import *
from mpi4py import MPI

# In[3]:


files=sorted(glob.glob('disk.out1*athdf'))
num_file=len(files)


# In[4]:
nrloc=4
rloc=np.zeros(nrloc)
rloc[0]=60.0
rloc[1]=100.0
rloc[2]=140.0
rloc[3]=180.0
rindex=np.zeros(nrloc, dtype=np.uint32)



# In[7]:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

count=0

for n in range(0,num_file,nprocs):
    fi=n+rank
    filename=files[fi]
    print(filename)
    data=ReduceData(filename)
    x1v=data['x1v']
    x2v=data['x2v']
    x2f=data['x2f']



    time=data['Time']
    rho=data['rho']  
    rhovr=data['rhovr']
    pgas=data['press']
    gast=data['gast']
    vel1=data['vel1']
    vel2=data['vel2']

    sigma_a=data['Sigma_a_0']
    sigma_p=data['Sigma_p_0']
    sigma_s=data['Sigma_s_0']
    
    Er=data['Er']
    Fr1=data['Fr1']
    Fr2=data['Fr2']
    Fr01=data['Fr01']
    Fr02=data['Fr02']

    Ek1=data['Ek1']
    Ek2=data['Ek2']
    Ek3=data['Ek3']

    BrBphi=data['BrBphi']
    BtBphi=data['BthetaBphi']

    rhovrvphi=data['rhovrvphi']
    rhovtvphi=data['rhovthetavphi']

    rhovp=data['rhovphi']
    rhovt=data['rhovtheta']

    Bcc1=data['Bcc1']
    Bcc2=data['Bcc2']
    Bcc3=data['Bcc3']
    PB1=data['PB1']
    PB2=data['PB2']
    PB3=data['PB3']
    PB=PB1+PB2+PB3
    lambda_a=data['lambda_a']
    dim=rho.shape
    ntheta=dim[0]
    nr=dim[1]
    radius=x1v
    
    if count == 0:
        ST_lambda=np.zeros((4,ntheta))
        ST_pg=np.zeros((4,ntheta))

    
    sigma_r=(sigma_a+sigma_s)
    


    ST_PB1=Bcc1**2
    ST_PB2=Bcc2**2
    ST_PB3=Bcc3**2

    ST_lambda[0,:]=lambda_a[:,rindex[0]]
    ST_lambda[1,:]=lambda_a[:,rindex[1]]
    ST_lambda[2,:]=lambda_a[:,rindex[2]]
    ST_lambda[3,:]=lambda_a[:,rindex[3]]

    ST_pg[0,:]=pgas[:,rindex[0]]
    ST_pg[1,:]=pgas[:,rindex[1]]
    ST_pg[2,:]=pgas[:,rindex[2]]
    ST_pg[3,:]=pgas[:,rindex[3]]
    
    count=count+1
        

    outfile='hist_'+files[fi][10:15]+'.npz'
# In[8]:

    np.savez(outfile,time=time,radius=radius,rloc=rloc,theta=x2v,x1f=data['x1f'],x2f=x2f,
         rho=rho,Er=Er,sigma=sigma_r,sigma_p=sigma_p,Fr1=Fr1,Fr2=Fr2,Fr01=Fr01,Fr02=Fr02,
         B1=Bcc1,B2=Bcc2,B3=Bcc3,B1sq=ST_PB1,B2sq=ST_PB2,B3sq=ST_PB3,vel1=vel1,vel2=vel2,
         PB1=PB1,PB2=PB2,PB3=PB3,pg=pgas,gast=gast,rhovr=rhovr,MRIlambda=ST_lambda,Ek1=Ek1,Ek2=Ek2,Ek3=Ek3,
         BrBp=BrBphi,BtBp=BtBphi,rhovrvphi=rhovrvphi,rhovtvphi=rhovtvphi,rhovp=rhovp,rhovt=rhovt,
         Sr1=data['Sr1'],Sr2=data['Sr2'],St1=data['St1'],St2=data['St2'])



# In[10]:





# In[ ]:




