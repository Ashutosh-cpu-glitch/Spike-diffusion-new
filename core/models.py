import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ============================
# Spiking Autoencoder
# ============================

class SpikingAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, beta=0.9):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, spike_input):

        # spike_input shape: (T, B, D)
        T, B, D = spike_input.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_out = []

        for t in range(T):

            cur1 = self.fc1(spike_input[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out.append(spk2)

        return torch.stack(spk_out)


# ============================
# Spiking Denoiser (Diffusion)
# ============================

class SpikingDenoiser(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, beta=0.95, max_timesteps=100):
        super().__init__()

        # timestep embedding
        self.t_embed = nn.Embedding(max_timesteps, hidden_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, spike_input, t):

        # spike_input shape: (T, B, D)
        T, B, D = spike_input.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # timestep embedding
        t_emb = self.t_embed(t)

        spk_out = []

        for step in range(T):

            cur1 = self.fc1(spike_input[step]) + t_emb
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out.append(spk2)

        # average spike activity over time
        return torch.mean(torch.stack(spk_out), dim=0)