import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
import torchvision


def normalize_sequential(raw_data):
    """
    Takes in 2-d data (len sequence, len latent) representing entire uninterupted sequence. Normalizes and standardizes 
    to -1 to 1 based on mean, max of EACH latent dim separately
    """
    if type(raw_data) != torch.Tensor: raw_data=torch.tensor(raw_data)
    z_mean = raw_data.mean(dim=0).unsqueeze(0); print(z_mean.shape)
    centered = raw_data - z_mean
    m, _ = centered.max(dim=0); mm, _ = centered.min(dim=0); 
    m_abs = torch.max(torch.abs(m),torch.abs(mm)).unsqueeze(0); print(m_abs.shape)
    normed_data = centered / m_abs
    return normed_data, z_mean, m_abs

def denormalize_sequential(raw_data, data_mean, data_abs): return raw_data * data_abs + data_mean



# Model. MUST MANUALLY CHANGE N_IN to reflect size of z

N_Z=100; N_ACTIONS=1; N_IN = (N_Z+N_ACTIONS); N_HIDDEN = 512; N_GAUSSIANS = 5; N_LSTM_LAYERS = 2

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(N_IN, N_HIDDEN, num_layers=N_LSTM_LAYERS) 
        # this outputs the hidden state, which must be run through linear for result.
        
        self.z_pi = nn.Linear(N_HIDDEN, N_GAUSSIANS*N_IN)
        self.z_sigma = nn.Linear(N_HIDDEN, N_GAUSSIANS*N_IN)
        self.z_mu = nn.Linear(N_HIDDEN, N_GAUSSIANS*N_IN)

    def forward(self, input_package): 
        x, hidden = input_package[0], input_package[1]
        
        # "out" will give you access to all hidden states in the sequence
        # "hidden" will allow you to continue the sequence and backpropagate,
        # by passing it as an argument  to the lstm at a later time
        
        # understand these mechanics more.
        out, hidden = self.lstm(x, hidden) # i was confusing what was h_t, c_t. Do more research on this

        # also predicting distributions to draw from for MDN
        pi = self.z_pi(out) # output of linear is all mushed together. separating out.
        
        # Do i even need softmax here? Moved this into loss function itself
        #pi = F.softmax(pi, dim=2) #ONNX can't handle dim 2
        
        #pi = pi / TEMPERATURE
        
        sigma = self.z_sigma(out)
        sigma = torch.exp(sigma)
        #sigma = sigma * (TEMPERATURE ** 0.5)
        
        mu = self.z_mu(out)
        
        return [pi, mu, sigma], hidden
    
    
def make_hidden(n_seqs, cuda):
    # Before we've done anything, we dont have any hidden state.
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    if cuda:
        return (torch.zeros(N_LSTM_LAYERS, n_seqs, N_HIDDEN).cuda(), # for cell, there are no layers. for lstm, using single layer
                torch.zeros(N_LSTM_LAYERS, n_seqs, N_HIDDEN).cuda())
    else:
        return (torch.zeros(N_LSTM_LAYERS, n_seqs, N_HIDDEN), # for cell, there are no layers. for lstm, using single layer
                torch.zeros(N_LSTM_LAYERS, n_seqs, N_HIDDEN))
    
from torch.distributions import Normal

# Making own sample function. Verify this, good chance I don't understand it fully.

def sample_mdn(mdn_coefs):
    pi = mdn_coefs[0]; 
    pi = F.softmax(pi, dim=2)
    
    pi = pi.squeeze(0).squeeze(0).permute(1,0)

    pi_ix = torch.distributions.categorical.Categorical(pi).sample().unsqueeze(1);

    mu = mdn_coefs[1].squeeze(0).squeeze(0).permute(1,0); 
    sigma = mdn_coefs[2].squeeze(0).squeeze(0).permute(1,0);

    mu_z = torch.gather(mu, 1, pi_ix).squeeze();
    sigma_z = torch.gather(sigma, 1, pi_ix).squeeze();

    gauss_z = torch.distributions.normal.Normal(mu_z, sigma_z)
    
    return gauss_z.sample().cpu().numpy()

#sample_mdn(mdn_coefs)

# from sonic version
# Check this loss function. Good chance it's wrong.
def mdn_loss_function(out_pi, out_sigma, out_mu, y):
    """
    Mixed Density Network loss function, see : 
    https://mikedusenberry.com/mixture-density-networks
    """
    EPSILON = 1e-6
    
    out_pi = F.softmax(out_pi, dim=2) #ONNX can't handle dim 2

    result = Normal(loc=out_mu, scale=out_sigma)
    result = torch.exp(result.log_prob(y))
    result = torch.sum(result * out_pi, dim=2) #changed this
    result = -torch.log(EPSILON + result)
    return torch.mean(result)


# from dusenberry
# takes in params for a single dim. for single item from single seq. WROTE OWN VERSION NOW TO WORK ON BATCH
def sample_preds(pi, sigmasq, mu, samples=10): # sampling for a single pt
  # rather than sample the single conditional mode at each
  # point, we could sample many points from the GMM produced
  # by the model for each point, yielding a dense set of
  # predictions
  N, K = pi.shape
  _, KT = mu.shape
  T = int(KT / K)
  #out = torch.zeros(N, samples, T)  # s samples per example
  out = torch.zeros(samples)  # s samples per example
  for i in range(N): 
    for j in range(samples):
      # pi must sum to 1, thus we can sample from a uniform
      # distribution, then transform that to select the component
      u = np.random.uniform()  # sample from [0, 1)
      # split [0, 1] into k segments: [0, pi[0]), [pi[0], pi[1]), ..., [pi[K-1], pi[K])
      # then determine the segment `u` that falls into and sample from that component
      prob_sum = 0
      for k in range(K):
        prob_sum += pi.data[i, k]
        if u < prob_sum:
          # sample from the kth component
          for t in range(T):
            sample = np.random.normal(mu.data[i, k*T+t], sigmasq.data[i, k]) # this used to have sqrt
            #sample = np.random.normal(mu.data[i, k*T+t], 0)
            #out[i, j, t] = sample
            out[j] = sample
          break
  return out











# GAN CREATION
class ConvBlock(nn.Module):
    def __init__(self, ni, no, ks, stride, bn=True, pad=None):
        super().__init__()
        if pad is None: pad = ks//2//stride
        self.conv = nn.Conv2d(ni, no, ks, stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(no) if bn else None
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.bn(x) if self.bn else x
      
class DCGAN_D(nn.Module):
    # image size, number of channels in image, number of channels out in next layer, number of extra layers
    def __init__(self, isize, nc, ndf, n_extra_layers=0):
        super().__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        self.initial = ConvBlock(nc, ndf, 4, 2, bn=False)
        csize,cndf = isize/2,ndf
        self.extra = nn.Sequential(*[ConvBlock(cndf, cndf, 3, 1)
                                    for t in range(n_extra_layers)])

        pyr_layers = []
        while csize > 4:
            pyr_layers.append(ConvBlock(cndf, cndf*2, 4, 2))
            cndf *= 2; csize /= 2
        self.pyramid = nn.Sequential(*pyr_layers)
        
        self.final = nn.Conv2d(cndf, 1, 4, padding=0, bias=False)

    def forward(self, input):
        x = self.initial(input)
        x = self.extra(x)
        x = self.pyramid(x)
        return self.final(x).mean(0).view(1) # returns the mean prediction for all images in batch
      
class DeconvBlock(nn.Module):
    def __init__(self, ni, no, ks, stride, pad, bn=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ni, no, ks, stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(no)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.bn(x) if self.bn else x
      

class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super().__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize!=isize: cngf*=2; tisize*=2
        layers = [DeconvBlock(nz, cngf, 4, 1, 0)]

        csize, cndf = 4, cngf
        while csize < isize//2:
            layers.append(DeconvBlock(cngf, cngf//2, 4, 2, 1))
            cngf //= 2; csize *= 2

        layers += [DeconvBlock(cngf, cngf, 3, 1, 1) for t in range(n_extra_layers)]
        layers.append(nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        self.features = nn.Sequential(*layers)

    def forward(self, input): return F.sigmoid(self.features(input)) # changed this to Sigmoid to force to 0 to 1
  

# helper functions to toggle btwn trainable and not trainable. Sets requires_grad on and off.

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())

def set_trainable_attr(m,b):
    m.trainable=b
    for p in m.parameters(): p.requires_grad=b

def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    if len(c)>0:
        for l in c: apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))
    
def to_np(t):
    return t.detach().cpu().numpy()















# VAE

#H_DIM = 256 # 32x32
#H_DIM = 6400 #this is for 64 X 64
H_DIM = 215296 # this is for 256 x 256

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
      
class UnFlatten(nn.Module):
    def forward(self, input, size=H_DIM): 
        return input.view(input.size(0), size, 1, 1)
    
class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=H_DIM, z_dim=100): # hardcoding h_dim based on how flat final step is
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=1), # made this one, otherwise was halving size of input image too many times (mnist)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2), # 6 for 64x64, 5 for 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2), #4, 2 for 64x64, 4 and 1 for 32
            nn.ConvTranspose2d(16, image_channels, kernel_size=4, stride=2), # adding this for 256 x 256
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z
    
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(self.fc3(z))
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return z, recon_x, mu, logvar
    
    
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

def pad_z(z): #adds 1 and 1 to end of z dim, as expected by G
    return z.unsqueeze(2).unsqueeze(3)
