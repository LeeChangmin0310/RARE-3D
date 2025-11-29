import numpy as np


class SumTree(object):
    """
    PER 구현을 위한 SumTree.
    capacity: leaf 개수 (메모리 크기)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        """
        priority: p
        data: transition tuple (s, a, r, s_, done)
        """
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # propagate the change up
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """
        v ~ U(0, total_priority)
        """
        parent_index = 0

        while True:
            left = 2 * parent_index + 1
            right = left + 1

            if left >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left]:
                    parent_index = left
                else:
                    v -= self.tree[left]
                    parent_index = right

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class Memory(object):
    """
    Prioritized Experience Replay 메모리

    alpha: priority exponent
    beta: IS weight exponent (annealing)
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_increment_per_sampling=1e-5):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.absolute_error_upper = 1.0  # clipped abs error 상한

    def _get_priority(self, error):
        error = np.abs(error) + 1e-6
        clipped = np.minimum(error, self.absolute_error_upper)
        return np.power(clipped, self.alpha)

    def store(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch_idx = np.empty((n,), dtype=np.int32)
        batch_memory = []
        ISWeights = np.empty((n, 1), dtype=np.float32)

        pri_seg = self.tree.total_priority / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.capacity:]) / self.tree.total_priority
        min_prob = max(min_prob, 1e-6)

        for i in range(n):
            a = pri_seg * i
            b = pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)

            prob = p / self.tree.total_priority
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            batch_idx[i] = idx
            batch_memory.append(data)

        return batch_idx, batch_memory, ISWeights

    def batch_update(self, tree_idx, errors):
        for ti, e in zip(tree_idx, errors):
            p = self._get_priority(e)
            self.tree.update(ti, p)
