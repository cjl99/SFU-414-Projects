from utils import *
from re_solver import Solver


common_params, dataset_params, net_params, solver_params = get_params('model.cfg')
solver = Solver(True, common_params, solver_params, net_params, dataset_params)
solver.train_model()