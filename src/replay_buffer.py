import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    """Replay buffer to store and sample trajectories for GFlowNet training."""
    
    def __init__(self, capacity: int = 1000, sampling_strategy: str = 'uniform'):
        """
        Args:
            capacity (int): Maximum size of the replay buffer. Default is 1000.
            sampling_strategy (str): Strategy for sampling ('uniform', 'prioritized', 'reservoir', 'stratified').
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.sampling_strategy = sampling_strategy
        self.priorities = deque(maxlen=capacity)  # For prioritized sampling
    
    def add(self, trajectory):
        """Add a new trajectory to the buffer.
        
        Args:
            trajectory: A tuple (states, actions, rewards, next_states, dones)
            where each element is a sequence representing the trajectory.
        """
        self.buffer.append(trajectory)
        if self.sampling_strategy == 'prioritized':
            # Initially assign max priority for new trajectories (could be based on cumulative reward or TD error)
            priority = max(self.priorities, default=1.0)
            self.priorities.append(priority)
    
    def sample(self, batch_size: int):
        """Sample a batch of trajectories based on the chosen strategy."""
        if self.sampling_strategy == 'uniform':
            return self._uniform_sample(batch_size)
        elif self.sampling_strategy == 'prioritized':
            return self._prioritized_sample(batch_size)
        elif self.sampling_strategy == 'reservoir':
            return self._reservoir_sample(batch_size)
        elif self.sampling_strategy == 'stratified':
            return self._stratified_sample(batch_size)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
    def _uniform_sample(self, batch_size):
        """Sample trajectories uniformly."""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def _prioritized_sample(self, batch_size):
        """Sample trajectories based on priorities."""
        priorities_np = np.array(self.priorities)
        probabilities = priorities_np / priorities_np.sum()  # Normalize to create probabilities
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        return [self.buffer[i] for i in indices]
    
    def _reservoir_sample(self, batch_size):
        """Reservoir sampling for diverse experiences."""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def _stratified_sample(self, batch_size):
        """Stratified sampling based on reward bins (as an example)."""
        # For simplicity, we will stratify by cumulative reward of the trajectory
        rewards = np.array([sum(experience[2]) for experience in self.buffer])  # Sum of rewards for each trajectory
        reward_bins = np.digitize(rewards, bins=[-1, 1])  # Low < -1, Medium [-1, 1], High > 1
        
        # Sample proportional to bin size
        bin_indices = [np.where(reward_bins == i)[0] for i in range(3)]
        bin_sizes = [len(indices) for indices in bin_indices]
        
        samples = []
        for bin_idx, bin_size in enumerate(bin_sizes):
            if bin_size > 0:
                bin_sample_size = int(batch_size * bin_size / len(self.buffer))  # Proportional sampling
                chosen_indices = np.random.choice(bin_indices[bin_idx], bin_sample_size, replace=False)
                samples.extend([self.buffer[i] for i in chosen_indices])
        
        return samples
    
    def update_priorities(self, indices, priorities):
        """Update the priorities of the trajectories in the buffer."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
