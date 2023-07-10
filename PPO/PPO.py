import numpy as np
import torch


class PPO:
    def __init__(self, dual_network, optimizer=None, lr=0.0001, eps_clip=0.2,
                 value_coef=0.5, entropy_coef=0.01, gamma=0.99, gae_lambda=0.95):
        """
        Initialize the PPO Agent.

        Args:
            dual_network (DualNetwork): The dual network model used for this agent.
            optimizer (torch.optim.Optimizer): The optimizer for training the model.
            lr (float): The learning rate for the optimizer.
            eps_clip (float): Hyperparameter used to clip the policy update.
            value_coef (float): Coefficient for the value loss in the total loss calculation.
            entropy_coef (float): Coefficient for the entropy bonus in the total loss calculation.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): The decay rate for GAE.
        """
        self.dual_network = dual_network
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        if optimizer is None:
            self.optimizer = torch.optim.Adam(dual_network.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

    def select_action(self, state):
        """
        Given a state, select an action.

        Args:
            state: The current state of the environment.

        Returns:
            The selected action, action_logprobs
        """
        # Convert state to tensor
        state = self.to_tensor(state)
        # Pass state through policy network
        policy, value = self.dual_network(state)
        # Create policy distribution
        policy_dist = torch.distributions.Categorical(policy)
        # Sample action from the distribution output by policy network
        action = policy_dist.sample()
        # Calculate log probability of action
        action_logprob = policy_dist.log_prob(action)
        # Return action, action_logprob
        return action, action_logprob, value

    def evaluate(self, state, action):
        """
        Evaluate the policy and value function at a given state and action.

        Args:
            state: The current state of the environment.
            action: The action taken by the agent.

        Returns:
            action_logprobs, state_value, dist_entropy
        """
        # Pass state through dual network
        policy, state_value = self.dual_network(state)
        if torch.isnan(policy).any():
            print(state)
        # Create policy distribution
        policy_dist = torch.distributions.Categorical(policy)
        # Calculate log probability of action
        action_logprob = policy_dist.log_prob(action)
        # Calculate entropy of action distribution
        dist_entropy = policy_dist.entropy()
        # Return action_logprobs, state_value, dist_entropy
        return action_logprob, state_value, dist_entropy

    def update(self, memory, num_epochs, mini_batch_size):
        """
        Update the parameters of the policy and value networks.

        Args:
            memory: The memory buffer containing stored experiences.
            num_epochs: Number of times to iterate over the entire batch.
            mini_batch_size: Number of samples to use in each mini-batch.
        """
        # Fetch stored experiences from memory
        old_states, old_actions, old_logprobs, old_rewards, old_is_terminals, old_values = memory.get()

        returns = self.calculate_returns(old_rewards, old_is_terminals, old_values)

        batch_size = old_states.size(0)
        for _ in range(num_epochs):
            for i in range(0, batch_size, mini_batch_size):
                states, actions, logprobs, returns_batch = self.fetch_minibatch(old_states, old_actions, old_logprobs,
                                                                                returns, i, mini_batch_size)

                loss = self.compute_loss(states, actions, logprobs, returns_batch)

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        memory.clear_memory()

    def calculate_returns(self, old_rewards, old_is_terminals, old_values):
        """
        Calculate and normalize the returns from old rewards, terminal states and value estimates using GAE.

        Args:
            old_rewards (torch.Tensor): The old rewards.
            old_is_terminals (torch.Tensor): The old terminal states.
            old_values (torch.Tensor): The old value estimates.

        Returns:
            torch.Tensor: The normalized returns.
        """
        # Squeeze tensors to remove unnecessary dimension
        old_rewards = old_rewards.squeeze()
        old_is_terminals = old_is_terminals.squeeze()
        old_values = old_values.squeeze()

        advantages = torch.zeros_like(old_rewards).to(self.dual_network.device)

        advantage = 0

        for t in reversed(range(len(old_rewards))):
            if old_is_terminals[t]:
                delta = old_rewards[t] - old_values[t]
                last_gae_lambda = 0
            else:
                if t == len(old_values) - 1:
                    next_value = old_values[t]
                else:
                    next_value = old_values[t + 1]
                delta = old_rewards[t] + self.gamma * next_value - old_values[t]
                last_gae_lambda = self.gamma * self.gae_lambda
            advantage = delta + last_gae_lambda * advantage
            advantages[t] = advantage
        returns = advantages + old_values
        return returns

    def fetch_minibatch(self, old_states, old_actions, old_logprobs, returns, start, mini_batch_size):
        """
        Fetch a mini-batch of states, actions, old log probabilities, and returns.

        Args:
            old_states (torch.Tensor): The old states.
            old_actions (torch.Tensor): The old actions.
            old_logprobs (torch.Tensor): The old log probabilities.
            returns (torch.Tensor): The returns.
            start (int): The start index for the mini-batch.
            mini_batch_size (int): The mini-batch size.

        Returns:
            tuple: A tuple of torch.Tensors representing the mini-batch.
        """
        mini_batch_indices = torch.arange(start, min(start + mini_batch_size, old_states.size(0))).long()
        mini_batch_indices = mini_batch_indices.to(self.dual_network.device)

        return old_states[mini_batch_indices], old_actions[mini_batch_indices], \
            old_logprobs[mini_batch_indices], returns[mini_batch_indices]

    def compute_loss(self, states, actions, old_logprobs, returns_batch):
        """
        Compute the loss for the current mini-batch of states, actions, old log probabilities, and returns.

        Args:
            states (torch.Tensor): The states.
            actions (torch.Tensor): The actions.
            old_logprobs (torch.Tensor): The old log probabilities.
            returns_batch (torch.Tensor): The returns for the current batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        logprobs_new, state_values, dist_entropy = self.evaluate(states, actions)
        advantages = returns_batch - state_values.detach()

        ratios = torch.exp(logprobs_new - old_logprobs.detach())

        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        surrogate_loss = -torch.min(surrogate1, surrogate2)
        value_loss = 0.5 * (returns_batch - state_values).pow(2)
        loss = surrogate_loss + self.value_coef * value_loss - self.entropy_coef * dist_entropy

        return loss

    def to_tensor(self, obj):
        """
        Convert a given object to a PyTorch tensor.

        If the object is a PyTorch tensor, it is returned as is.
        If the object is a list or a numpy array, it is converted to a PyTorch tensor and returned.
        If the object is neither of these, a ValueError is raised.

        Parameters:
        obj: The input object, which can be a PyTorch tensor, a list, or a numpy array.

        Returns:
        A PyTorch tensor representation of the input object.

        Raises:
        ValueError: If the input object is neither a PyTorch tensor, a list, nor a numpy array.
        """
        # Check if the object is a PyTorch tensor.
        if torch.is_tensor(obj):
            return obj.to(self.dual_network.device)
        # If not, check if it's a list or a numpy array.
        elif isinstance(obj, (list, np.ndarray)):
            return torch.tensor(obj).to(self.dual_network.device)
        else:
            raise ValueError("Object is not a PyTorch tensor, a list, or a numpy array.")
