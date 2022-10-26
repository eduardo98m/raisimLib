import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal


class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.action_mean = None

    def sample(self, obs):
        self.action_mean = self.architecture(obs).cpu().numpy()
        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob

    def evaluate(self, obs, actions):
        self.action_mean = self.architecture(obs)
        return self.distribution.evaluate(self.action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        self.distribution.update()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.architecture(x)

class Teacher(nn.Module):
    def __init__(
            self, 
            shape, 
            non_priv_range, 
            priv_range, 
            actionvation_fn, 
            output_size,
            alpha=0.75
        ):
        super(Teacher, self).__init__()
        self.activation_fn = actionvation_fn
        self.non_priv_shape = non_priv_range
        self.priv_shape = priv_range
        self.input_shape = [
            self.non_priv_shape[1] + self.priv_shape[1] - \
            (self.non_priv_shape[0] + self.priv_shape[0])
        ]
        self.output_shape = [output_size]

        encoder_modules = [
            nn.Linear(priv_range[1] - priv_range[0], 72),
            self.activation_fn(),
            nn.Linear(72, 64),
            self.activation_fn()
        ]
        scale = [np.sqrt(2)] * 2
        self.encoder = nn.Sequential(*encoder_modules)
        self.init_weights(self.encoder, scale)

        classifier_modules = [
            nn.Linear(64 + non_priv_range[1] - non_priv_range[0], shape[0]), 
            self.activation_fn()
        ]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            classifier_modules.append(nn.Linear(shape[idx], shape[idx+1]))
            classifier_modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        classifier_modules.append(nn.Linear(shape[-1], output_size))
        self.classifier = nn.Sequential(*classifier_modules)
        scale.append(np.sqrt(2))
        self.init_weights(self.classifier, scale)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        priv_data = x[:, self.priv_shape[0]:self.priv_shape[1]]
        non_priv_data = x[:, self.non_priv_shape[0]:self.non_priv_shape[1]]

        x = self.encoder(priv_data)
        x = torch.cat((x, non_priv_data), 1)
        return self.classifier(x) 

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        priv_data = x[:, self.priv_shape[0]:self.priv_shape[1]]

        return self.encoder(priv_data)

class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, size, init_std, fast_sampler, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size, dim], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.std_np = self.std.detach().cpu().numpy()

    def update(self):
        self.std_np = self.std.detach().cpu().numpy()

    def sample(self, logits):
        self.fast_sampler.sample(logits, self.std_np, self.samples, self.logprob)
        return self.samples.copy(), self.logprob.copy()

    def evaluate(self, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
