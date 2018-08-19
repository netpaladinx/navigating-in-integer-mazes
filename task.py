from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import defaultdict

import random
import numpy as np


class IntegerMaze(object):
    def __init__(self, min_integer=0, max_integer=9999, max_num_ops=10, sel_num_ops=5, len_path=4, num_paths=10, operand_min=1, operand_max=100, random_seed=1111):
        """
        The default setting:
            Range of integers: [0, 10000) (i.e. #integers = 10000)
            Possible Ops:
                Add(a): (x + a) % 10000
                Subtract(a): (x - a) % 10000
                Multiply(a): (x * a) % 10000
                Divide(a): (x / a) % 10000
                where a \in [1, 100]
            #Ops in the op pool: 10
            #Ops selected: 5
            Length of an Op path: 4
        """
        self.min_integer = min_integer
        self.max_integer = max_integer + 1
        self.max_num_ops = max_num_ops
        self.sel_num_ops = sel_num_ops
        self.len_path = len_path
        self.num_paths = num_paths
        self.operand_min = operand_min
        self.operand_max = operand_max
        self.random_seed = random_seed

        random.seed(random_seed)
        np.random.seed(self.random_seed)

        self.op_pool = self.build_op_pool(self.max_num_ops)
        self.sel_ops = random.sample(self.op_pool, self.sel_num_ops)
        self.paths = self.build_op_paths(self.sel_ops, self.len_path, self.num_paths)
        self._edgeIds_rels, self._edges = self.build_maze()
        self.op_counts = {}

    def build_op_pool(self, size):
        pool = []
        operand_min = self.operand_min
        operand_max = self.operand_max
        for i in range(operand_min, operand_max+1):
            pool.append(lambda x, i=i: ('+%d' % i, (x+i - self.min_integer) % (self.max_integer - self.min_integer) + self.min_integer))
        for i in range(operand_min, operand_max+1):
            pool.append(lambda x, i=i: ('-%d' % i, (x-i - self.min_integer) % (self.max_integer - self.min_integer) + self.min_integer))
        for i in range(operand_min, operand_max+1):
            pool.append(lambda x, i=i: ('*%d' % i, (x*i - self.min_integer) % (self.max_integer - self.min_integer) + self.min_integer))
        for i in range(operand_min, operand_max+1):
            pool.append(lambda x, i=i: ('/%d' % i, (int(x/i) - self.min_integer) % (self.max_integer - self.min_integer) + self.min_integer))
        return random.sample(pool, size)

    def build_maze(self):
        edge_to_rels = defaultdict(list)
        for i1 in range(self.min_integer, self.max_integer):
            for j, f in enumerate(self.op_pool):
                i2 = f(i1)[1]
                edge_to_rels[(i1, i2)].append(j)
        edges = sorted(edge_to_rels.keys())
        edgeIds_rels = []
        for i, e in enumerate(edges):
            for r in edge_to_rels[e]:
                edgeIds_rels.append([i, r])
        return edgeIds_rels, edges

    def build_op_paths(self, sel_ops, len_path, num_paths):
        paths = []
        for i in range(num_paths):
            paths.append(np.random.choice(sel_ops, len_path))
        return paths

    def edges_from(self):
        return np.array(self._edges)[:,0]

    def edges_to(self):
        return np.array(self._edges)[:,1]

    def num_edges(self):
        return len(self._edges)

    def edges(self):
        return np.array(self._edges)

    def edges_rels(self):
        rels_mat = np.zeros([self.num_edges(), self.max_num_ops])
        eid_r = np.array(self._edgeIds_rels)
        e_ind = eid_r[:, 0]
        r_ind = eid_r[:, 1]
        rels_mat[e_ind, r_ind] = 1.
        return rels_mat

    def print_paths(self):
        for i in range(self.num_paths):
            path = self.paths[i]
            print(self.path_str(path))

    def path_str(self, path):
        return ', '.join(map(lambda f: f(0)[0], path))

    def sampler(self, num, debug=False):
        samples = []
        max_p, min_p, total_p = 0., -1., 0.
        for i in np.random.randint(len(self.paths), size=num):
            src = np.random.randint(self.min_integer, high=self.max_integer)
            dst = src
            for f in self.paths[i]:
                dst = f(dst)[1]
            if debug:
                p = self.test_possibilities(src, dst)
                if max_p < p:
                    max_p = p
                if min_p == -1. or min_p > p:
                    min_p = p
                total_p += p
                # print('src: %d, dst: %d, path: %s, possibilities: %d' % (src, dst, self.path_str(self.paths[i]), self.test_possibilities(src, dst)))
            samples.append([src, dst])
        avg_p = total_p / num
        print('min_possibilities: %.4f, max_possibilities: %.4f, avg_possibilities: %.4f' % (min_p, max_p, avg_p))
        return samples

    def test_possibilities(self, src, dst):
        count = 0
        for f1 in self.op_pool:
            v1 = f1(src)[1]
            for f2 in self.op_pool:
                v2 = f2(v1)[1]
                for f3 in self.op_pool:
                    v3 = f3(v2)[1]
                    for f4 in self.op_pool:
                        v4 = f4(v3)[1]
                        if v4 == dst:
                            count += 1
                            key = '%s, %s, %s, %s' % (f1(0)[0], f2(0)[0], f3(0)[0], f4(0)[0])
                            self.op_counts[key] = self.op_counts.get(key, 0) + 1
        return count

    def hit(self, path_str):
        _hit = False
        for path in self.paths:
            if path_str == self.path_str(path):
                _hit = True
        return _hit

if __name__ == '__main__':
    integer_maze = IntegerMaze()
    integer_maze.print_paths()
    samples = integer_maze.sampler(1000, debug=True)

    K = 100
    i = 0
    for k, v in sorted(integer_maze.op_counts.iteritems(), key=lambda (k,v): (-v,k)):
        if i < K:
           print('%s: %s %s' % (k, v, '*' if integer_maze.hit(k) else ''))
        else:
            break
        i += 1

    for e in integer_maze.edges_rels:
        print(e)

    for i, f in enumerate(integer_maze.op_pool):
        print('%d %s' % (i, f(0)[0]))