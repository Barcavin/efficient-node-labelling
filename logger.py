import sys
import torch
from collections import defaultdict

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = defaultdict(list)

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 #and run < len(self.results)
        self.results[run].append(result)
    
    def get_argmax(self, result, last_best=True):
        if last_best:
            # get last max value index by reversing result tensor
            argmax = result.size(0) - result[:, 0].flip(dims=[0]).argmax().item() - 1
        else:
            argmax = result[:, 0].argmax().item()
        return argmax

    def print_statistics(self, run=None, file=sys.stdout):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = self.get_argmax(result)
            print(f'Run {run + 1:02d}:',file=file)
            print(f'Highest Valid: {result[:, 0].max():.2f}',file=file)
            print(f'   Final Test: {result[argmax, 1]:.2f}',file=file)
        else:
            best_results = []
            for r in self.results:
                r = self.results[r]
                r = 100 * torch.tensor(r).reshape(-1, 2)
                valid = r[:, 0].max().item()
                argmax = self.get_argmax(r)
                test = r[argmax, 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:',file=file)
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}',file=file)
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}',file=file)


