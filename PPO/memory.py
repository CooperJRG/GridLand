import torch


class Memory:
    """
    Class for handling the memory buffer for Proximal Policy Optimization (PPO).

    Args:
        device (torch.device): Device to which the tensors should be sent.
    """

    def __init__(self, device):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.device = device

    def clear_memory(self):
        """
        Clears the memory buffers.
        """
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

    def add(self, state, action, logprob, reward, is_terminal, value):
        """
        Adds a new interaction to the memory buffer.

        Args:
            state (torch.Tensor): The state at the current timestep.
            action (torch.Tensor): The action taken at the current timestep.
            logprob (torch.Tensor): The log probability of the action taken at the current timestep.
            reward (torch.Tensor): The reward received at the current timestep.
            is_terminal (torch.Tensor): Indicator whether the current timestep is terminal.
            value (torch.Tensor): The value received at the current timestep.
        """
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.logprobs.append(logprob.detach())
        self.rewards.append(reward.detach())
        self.is_terminals.append(is_terminal.detach())
        self.values.append(value.detach())

    def get(self):
        """
        Returns the memory buffers as detached PyTorch tensors on the correct device.

        Returns:
            tuple: Tuple of states, actions, logprobs, rewards, and is_terminals tensors.
        """
        states = torch.stack(self.states).to(self.device).detach()
        actions = torch.stack(self.actions).to(self.device).detach()
        logprobs = torch.stack(self.logprobs).to(self.device).detach()
        rewards = torch.stack(self.rewards).to(self.device).detach()
        is_terminals = torch.stack(self.is_terminals).to(self.device).detach()
        values = torch.stack(self.values).to(self.device).detach()

        return states, actions, logprobs, rewards, is_terminals, values
