"""
Interface for MDO lab optimization to call pretrained models
"""

import argparse
import ast
import os
import json
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable
from abc import abstractmethod
from collections import OrderedDict

import comm
import numpy as np
from mpi4py import MPI

from baseclasses import AeroSolver, AeroProblem
from pygeo.parameterization import DVGeometry, DVGeometryCustom
from pyoptsparse import OPT

from flowvae.app.wing.api import WingAPI

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is required when using the ML CFD solver with backend 'pytorch'."
    ) from exc

from flowvae.ml_operator.operator import load_model_from_checkpoint

class MLSolver(AeroSolver):
    """
    Surrogate CFD solver that uses a pretrained machine learning model to supply
    aerodynamic function values and their sensitivities with respect to surface
    geometry and selected flow-condition parameters.

    The class exposes the minimal subset of the ADflow interface that is relied
    upon in this script so it can act as a drop-in replacement during
    optimization. The ML model is assumed to accept a flattened surface point
    cloud (concatenated xyz coordinates) optionally augmented with additional
    scalar aero-condition inputs, and to produce the requested aerodynamic
    quantities. Gradients are obtained via automatic differentiation when using
    the PyTorch backend.
    """

    def __init__(
        self,
        output_keys: List[str],
        condition_keys: Optional[Dict[str, Union[List[Union[float, int]], Union[float, int]]]] = {
            'alpha': [1.5, 9.0],
            'mach': [0.75, 0.90],
            'reynolds': 20000000.0,
        },    # The dict of array input for operating conditions; including the range of them
        options: Dict = {},
        device=None,
        comm=None,
        debug: bool = False,
    ):
        if not output_keys:
            raise ValueError("At least one output key must be provided for the ML CFD solver.")

        default_options = {
            "output_dir": (str, "output"),
            "output_prefix": (str, "ml_surface"),
            "write_surface_tecplot": (bool, True)
        }

        super().__init__(
            name="MLCFD",
            category="Machine-Learned CFD",
            defaultOptions=default_options,
            options=options,
            comm=comm,
        )

        self.device = device
        self.output_keys = output_keys
        self.condition_keys = {} if condition_keys is None else condition_keys
        self.model: WingAPI
        self.debug = debug

        self.DVGeo: Optional[DVGeometry] = None
        self.surface_mesh: Optional[np.ndarray] = None
        self.allWallsGroup = ''  # placeholder
        self.curAP = None

        self._last_outputs: Dict[str, float] = {}
        self._last_surface_grad_global: Dict[str, np.ndarray] = {}
        self._last_condition_grad: Dict[str, np.ndarray] = {}
        self._last_condition_meta: List[Tuple[str, Optional[str]]] = []
        self._last_surface_counts: Optional[Iterable[int]] = None
        self._last_surface_shape: Optional[Tuple[int, int]] = None

        self.solveFailed = False
        self.fatalFail = False
        self.adjointFailed = False
        self._stored_lift_distribution = None
        self._stored_slices = None
        self.calling_counter = 0
        self._tecplot_file = None
        self._tecplot_header_written = False

    # ------------------------------------------------------------------ #
    # Interface setup helpers
    # ------------------------------------------------------------------ #
    def setModel(self, model: WingAPI, ap: AeroProblem):
        '''
        load model from standard floGen savings
        
        '''
        if self.comm is None or self.comm.rank == 0:
            self.model = model
            self.model.setAP(ap)

        if self.comm is not None:
            self.comm.barrier()

    def setAeroProblem(self, aeroProblem):
        '''
        Set the AeroProblem to be used
        -** update the geometry in DVGeo
        '''

        ptSetName = "ml_%s_coords" % aeroProblem.name

        # Tell the user if we are switching aeroProblems
        if self.curAP != aeroProblem:
            self.pp("+" + "-" * 70 + "+")
            self.pp("|  ML Switching to Aero Problem: %-41s|" % aeroProblem.name)
            self.pp("+" + "-" * 70 + "+")

        self.curAP = aeroProblem

        # Now check if we have an DVGeo object to deal with:
        if self.DVGeo is not None:
            # DVGeo appeared and we have not embedded points!
            if ptSetName not in self.DVGeo.points:
                assert self.surface_mesh is not None, "Set the surface mesh before calling setAeroProblem"
                # for DVGeoCustom, the surface flow for ml is not updated with projection, but with hard rewritten.
                # for other DVGeo, the point-set will be updated with projection when calling update
                self.DVGeo.addPointSet(self.surface_mesh.reshape(3, -1).T, ptSetName, **self.pointSetKwargs)

        # Check if our point-set is up to date:
        if not self.DVGeo.pointSetUpToDate(ptSetName):
            coords = self.DVGeo.update(ptSetName, config=aeroProblem.name)
            # global_coords, counts = self._gather_surface(coords)
            self.surface_mesh = coords.T.reshape(self.surface_mesh_shape)

        if not hasattr(self.curAP, "solveFailed"):
            self.curAP.solveFailed = False
        if not hasattr(self.curAP, "fatalFail"):
            self.curAP.fatalFail = False
        if not hasattr(self.curAP, "adjointFailed"):
            self.curAP.adjointFailed = False

    def setDVGeo(self, DVGeo: DVGeometry, useBaseline: int = 1, pointSetKwargs=None, customPointSetFamilies=None):
        '''
        Set DVGeo

        For most ML models, the surface mesh is what input to the model (TODO w/ possible algibracal process)
        
        '''
        super().setDVGeo(DVGeo, pointSetKwargs=pointSetKwargs, customPointSetFamilies=customPointSetFamilies)

        # update self surface mesh from DVGeo
        if useBaseline >= 0:
            assert isinstance(DVGeo, DVGeometryCustom), 'only DVGeometryCustom class has the baseline geometry stored'
            DVGeo: DVGeometryCustom
            if self.debug: print('Update surface mesh from DVGeo baseline geometry ...')
            if self.debug: print('wing_coefs:', DVGeo.parameters)
            self.surface_mesh = DVGeo._updateOneModel(useBaseline, DVGeo.parameters, keep_shape=True)   # this should be single proc. for refernce
            if self.debug: print('wing_mesh', self.surface_mesh[:, 0, :3])

    def setSurfaceMesh(self, mesh: np.ndarray):
        self.surface_mesh = mesh
        # Reset point-set registration so it can be rebuilt with the new mesh.

    def _write_surface_tecplot(self, surface_outputs, fname):

        mesh = np.asarray(self.surface_mesh)
        if mesh.ndim == 4 and mesh.shape[-1] == 3:
            mesh = mesh[:, :, 0, :]
        elif mesh.ndim == 3 and mesh.shape[0] == 3:
            mesh = mesh.transpose(1, 2, 0)
        if mesh.ndim != 3 or mesh.shape[-1] != 3:
            self.pp("Tecplot output skipped: unsupported mesh shape.")
            return

        data = np.asarray(surface_outputs)
        if data.ndim == 4:
            data = data[0]
        if data.ndim != 3:
            self.pp("Tecplot output skipped: unsupported output shape.")
            return

        nvar, ci, cj = data.shape
        ni, nj, _ = mesh.shape
        if ci != ni - 1 or cj != nj - 1:
            self.pp("Tecplot output skipped: output shape does not match mesh.")
            return

        # Cell-centered -> node-centered (vectorized)
        node_vals = np.zeros((nvar, ni, nj), dtype=data.dtype)
        counts = np.zeros((ni, nj), dtype=np.int32)
        for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
            node_vals[:, di:di + ci, dj:dj + cj] += data
            counts[di:di + ci, dj:dj + cj] += 1
            
        node_vals /= counts[None, :, :]

        print(f"Writing surface Tecplot output to {fname}.dat...")
        if self._tecplot_file is None:
            self._tecplot_file = fname + ".dat"
        mode = "a" if self._tecplot_header_written else "w"
        with open(self._tecplot_file, mode, encoding="ascii") as f:
            if not self._tecplot_header_written:
                f.write('TITLE="ML Surface"\n')
                var_names = ["X", "Y", "Z"] + [f"Q{i+1}" for i in range(nvar)]
                f.write("VARIABLES=" + ",".join(f'"{v}"' for v in var_names) + "\n")
                self._tecplot_header_written = True
            f.write(f'ZONE T="step_{self.calling_counter:05d}", I={ni}, J={nj}, F=POINT\n')
            for j in range(nj):
                for i in range(ni):
                    x, y, z = mesh[i, j]
                    q = node_vals[:, i, j]
                    f.write(f"{x} {y} {z} " + " ".join(str(val) for val in q) + "\n")

    def _write_surface(self, surface_outputs):
        if not self.getOption("write_surface_tecplot"):
            return
        if self.comm is not None and self.comm.rank != 0:
            return
        
        fname = os.path.join(
            self.getOption("output_dir"),
            f"{self.getOption('output_prefix')}_{self.calling_counter:05d}",
        )
        self._write_surface_tecplot(surface_outputs, fname)

    @property
    def surface_mesh_shape(self):
        return self.surface_mesh.shape

    # def addLiftDistribution(self, *args, **kwargs):
    #     self._stored_lift_distribution = (args, kwargs)

    # def addSlices(self, *args, **kwargs):
    #     self._stored_slices = (args, kwargs)

    # ------------------------------------------------------------------ #
    # ML evaluation
    # ------------------------------------------------------------------ #
    def _build_condition_vector(self, aeroProblem: AeroProblem):
        '''
        
        TODO: Consider multicondition optimization, the `values` output can have a batch dimension 
        to enable the model batch process results under multiple operating conditions
        '''

        values = []
        meta   = []     # the correlation betw. key and design varialbes (if added to DV, else None)
        out_of_range_flag = False

        for key in self.condition_keys:
            if isinstance(self.condition_keys[key], (int, float)):
                # fix value, check if its the same with the ap
                if hasattr(aeroProblem, key):
                    assert getattr(aeroProblem, key) == self.condition_keys[key], f"The value of {key} should be fix to {self.condition_keys[key]}"
            elif isinstance(self.condition_keys[key], list):
                # range value
                assert hasattr(aeroProblem, key), f"AeroProblem does not have attribute '{key}' required by the ML model."
                val = getattr(aeroProblem, key)
                if val < self.condition_keys[key][0] or val > self.condition_keys[key][1]:
                    self.pp(f"==!!!! Warning !!!!===")
                    self.pp(f"== The value of {key} = {val} is outside the given range {self.condition_keys[key]}")
                    self.pp(f"======================")
                    out_of_range_flag = True

                values.append(float(val))
            
            else:
                raise ValueError(f'Invalid type of condition_keys["{key}"]. Only support int float and list')

            dv_name = None
            for name, dv in aeroProblem.DVs.items():
                # due to code in AeroProblem, the dv object has the attr `key` to mark the type (e.g. alpha) of the DV
                # but also has a `name` to be the key in the aeroProlbem.DVs dictionary
                # TODO: when comes with multiple aero problems, here needs to deal with offset
                if dv.key.lower() == key.lower():
                    dv_name = name
                    break
            meta.append((key, dv_name))

        return out_of_range_flag, np.asarray(values, dtype=np.float32), meta

    def _evaluate_model(self, surface_points, condition_vector):
        '''

        TODO: Consider multicondition optimization, the `values` output can have a batch dimension 
        to enable the model batch process results under multiple operating conditions
        '''
        if self.comm is not None and self.comm.rank != 0:
            return {
                "values": None,
                "surface_grad": None,
                "condition_grad": None,
            }

        inp = torch.tensor(surface_points, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
        cnd = torch.tensor(condition_vector, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)

        outputs = self.model.predict(inp, cnd)

        outputs = outputs.squeeze(0)
        values_np = outputs.detach().cpu().numpy()

        print(f'ML model gives results {values_np} for input condition {cnd}')

        if values_np.size != len(self.output_keys):
            raise ValueError(
                f"ML model returned {values_np.size} outputs, but {len(self.output_keys)} output keys were provided."
            )

        surface_rows = []
        condition_rows = []
        for i in range(values_np.size):
            grads = torch.autograd.grad(
                outputs[i],
                (inp, cnd),
                retain_graph=i < values_np.size - 1,
                create_graph=False,
                allow_unused=False,
            )
            surface_rows.append(grads[0].detach().cpu().numpy().reshape(3, -1).T)
            condition_rows.append(grads[1].detach().cpu().numpy().reshape(-1))

        values = {name: float(values_np[idx]) for idx, name in enumerate(self.output_keys)}
        surface_grads = {
            name: surface_rows[idx].astype(np.float64, copy=True) for idx, name in enumerate(self.output_keys)
        }
        condition_grads = {
            name: condition_rows[idx].astype(np.float64, copy=True) for idx, name in enumerate(self.output_keys)
        }

        # print(surface_grads, condition_grads)

        return {
            "values": values,
            "surface_grad": surface_grads,
            "condition_grad": condition_grads,
        }

    # ------------------------------------------------------------------ #
    # Public solver API
    # ------------------------------------------------------------------ #
    def __call__(self, aeroProblem):

        self.setAeroProblem(aeroProblem)

        # initial flags of failure
        self.curAP.solveFailed = False  # reset the flags in case CFD varification runs setting them to True
        self.curAP.fatalFail = False
        self.solveFailed = False    # try restart clean
        self.fatalFail = False  # force reset

        flag, condition_vector, condition_meta = self._build_condition_vector(aeroProblem)
        if flag:
            self.solveFailed = True
            self.curAP.solveFailed = True
            return self._last_outputs
        # print('model input ->', self.surface_mesh.shape, condition_vector)
        if self.comm.rank == 0:
            eval_result = self._evaluate_model(self.surface_mesh, condition_vector)
        else:
            eval_result = None
        eval_result = self.comm.bcast(eval_result, root=0)

        self._last_outputs = eval_result["values"]
        self._last_surface_grad_global = eval_result["surface_grad"]
        self._last_condition_grad = eval_result["condition_grad"]
        self._last_condition_meta = condition_meta

        self.calling_counter += 1
        if self.comm.rank == 0:
            surface_outputs = getattr(self.model, "last_surface_outputs", None)
            if surface_outputs is not None and self.getOption("write_surface_tecplot"):
                self._write_surface(surface_outputs)

        # deal failure
        if any(np.isnan(val) for val in self._last_outputs.values()):
            self.pp("> ======= ML solver returned NaN outputs.")
            self.solveFailed = True
            self.curAP.solveFailed = True

        if any(np.isnan(arr).any() for arr in self._last_surface_grad_global.values()) or any(
            np.isnan(arr).any() for arr in self._last_condition_grad.values()
        ):
            self.pp("> ======= ML solver gradient returned NaN outputs.")
            self.adjointFailed = True
            self.curAP.adjointFailed = True

        self.curAP.solveFailed = self.curAP.solveFailed or self.solveFailed
        self.curAP.fatalFail = self.curAP.fatalFail or self.fatalFail

        return self._last_outputs

    def evalFunctions(self, aeroProblem, funcs, evalFuncs=None, ignoreMissing=False):
        self.setAeroProblem(aeroProblem)
        if not self._last_outputs:
            raise RuntimeError("MLCFDSolver.evalFunctions called before the solver was evaluated.")

        if evalFuncs is None:
            evalFuncs = aeroProblem.evalFuncs

        missing = []
        for func in evalFuncs:
            key = func.lower()
            if key not in self._last_outputs:
                missing.append(func)
                continue
            full_key = f"{aeroProblem.name}_{key}"
            aeroProblem.funcNames[key] = full_key
            funcs[full_key] = self._last_outputs[key]

        if missing and not ignoreMissing:
            raise ValueError(f"Requested functions {missing} were not produced by the ML solver.")

    # def _extract_local_surface_gradient(self, func_key):
    #     grad_global = self._last_surface_grad_global.get(func_key)
    #     if grad_global is None or self._last_surface_counts is None:
    #         return None
    #     start = sum(self._last_surface_counts[: self.comm.rank])
    #     end = start + self._last_surface_counts[self.comm.rank]
    #     return grad_global[start:end]

    def evalFunctionsSens(self, aeroProblem, funcsSens, evalFuncs=None, ignoreMissing=True):
        self.setAeroProblem(aeroProblem)
        if not self._last_outputs:
            raise RuntimeError("MLCFDSolver.evalFunctionsSens called before the solver was evaluated.")

        if evalFuncs is None:
            evalFuncs = aeroProblem.evalFuncs
        
        missing = []
        for func in evalFuncs:
            key = func.lower()
            if key not in self._last_outputs:
                missing.append(func)
                continue

            full_key = f"{aeroProblem.name}_{key}"
            sens_dict = {}

            # local_grad = self._extract_local_surface_gradient(key)
            local_grad = self._last_surface_grad_global.get(key)
            if local_grad is not None and self.DVGeo is not None:
                # for ML solver, we don't want sumup all gradients, so comm=None
                geom_sens = self.DVGeo.totalSensitivity(
                    local_grad[None, :, :],  "ml_%s_coords" % aeroProblem.name, comm=None, config=aeroProblem.name
                )
                sens_dict.update(geom_sens)

            condition_grad = self._last_condition_grad.get(key)
            if condition_grad is not None and condition_grad.size > 0:
                for (cond_key, dv_name), grad_val in zip(self._last_condition_meta, condition_grad):
                    if dv_name is None:
                        continue
                    # the geometric mesh may depends on the operating conditions, so here we add the direct
                    # gradient to them together with the geometric part
                    sens_dict[dv_name] = sens_dict.get(dv_name, 0.0) + float(grad_val)

            funcsSens[full_key] = sens_dict
        
        if missing and not ignoreMissing:
            raise ValueError(f"Requested sensitivity for functions {missing} were not produced by the ML solver.")

    def getSurfaceCoordinates(self, groupName='', **kwargs):

        raise NotImplementedError("getSurfaceCoordinates is not implemented for MLSolver. Use the surface mesh directly instead.")

        if self.surface_mesh is None:
            raise RuntimeError("A mesh object is required to access surface coordinates.")
        return self.surface_mesh.reshape(3, -1).T

    def getSurfaceConnectivity(self, groupName=None):
        _ = groupName
        if self.surface_mesh is None:
            raise RuntimeError("Surface mesh data is required to build connectivity.")

        ni, nj = self.surface_mesh.shape[1:]
        if ni < 2 or nj < 2:
            conn = np.zeros((0, 4), dtype=np.int32)
            face_sizes = np.zeros(0, dtype=np.int32)
            return conn, face_sizes

        quads = []
        for i in range(ni - 1):
            row_offset = i * nj
            next_row_offset = (i + 1) * nj
            for j in range(nj - 1):
                n00 = row_offset + j
                n01 = row_offset + j + 1
                n11 = next_row_offset + j + 1
                n10 = next_row_offset + j
                quads.append([n00, n01, n11, n10])

        conn = np.asarray(quads, dtype=np.int32)
        face_sizes = np.full(conn.shape[0], 4, dtype=np.int32)
        return conn, face_sizes



def _jsonify_value(val):
    if isinstance(val, np.ndarray):
        if val.shape == ():
            return float(val)
        return val.tolist()
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, dict):
        return {k: _jsonify_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_jsonify_value(v) for v in val]
    return val

def _append_cfd_hist(path, row):

    if not os.path.exists(path):
        data = {key: [] for key in row}
    else:
        with open(path, "r", encoding="ascii") as f:
            data = json.load(f)
    for key in row:
        data.setdefault(key, [])
        data[key].append(_jsonify_value(row[key]))
    with open(path, "w", encoding="ascii") as f:
        json.dump(data, f)

@contextmanager
def _rank0_guard(comm):
    is_rank0 = (comm is None) or (comm.rank == 0)
    if not is_rank0:
        yield False
        return
    try:
        yield True
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        if comm is not None:
            comm.Abort(1)
        raise

class _OnlineGEKDeltaModel:
    """
    Online GEK model for delta = CFD - ML using function values and gradients.

    Args:
        ap: AeroProblem handle (used for objective/constraint keys).
        max_points: Sliding-window size for online training samples.
        max_dims: Maximum retained DV dimensions before GEK fitting.
        nugget: Diagonal jitter added to covariance for numerical stability.
        stable_norm: Enable x/y normalization for stable small-sample fitting.
        xscale_floor: Lower bound for per-dimension x normalization scale.
        yscale_floor: Lower bound for output-delta normalization scale.
        ell_floor: Lower bound for GP length scales in normalized space.
        exp_clip: Clamp exponent in RBF kernel evaluation to avoid underflow.
        use_active_subspace: Learn and apply gradient-based active subspace.
        active_rank: Hard cap for active-subspace dimension.
        active_energy: Target cumulative eigenvalue energy for subspace size.
        log_interval: Prediction-log print interval (stateful, used by helper methods).
    """

    def __init__(
        self,
        ap: AeroProblem,
        comm=None,
        max_points: int = 20,
        max_dims: int = 40,
        nugget: float = 1e-8,
        stable_norm: bool = False,
        xscale_floor: float = 1e-3,
        yscale_floor: float = 1e-8,
        ell_floor: float = 0.2,
        exp_clip: float = 60.0,
        use_active_subspace: bool = False,
        active_rank: int = 8,
        active_energy: float = 0.95,
        log_interval: int = 1,
    ):
        self.ap = ap
        self.comm = comm
        self.max_points = max(2, int(max_points))
        self.max_dims = max(1, int(max_dims))
        self.nugget = float(nugget)
        self.stable_norm = bool(stable_norm)
        self.xscale_floor = float(xscale_floor)
        self.yscale_floor = float(yscale_floor)
        self.ell_floor = float(ell_floor)
        self.exp_clip = float(exp_clip)
        self.use_active_subspace = bool(use_active_subspace)
        self.active_rank = max(1, int(active_rank))
        self.active_energy = float(active_energy)
        self.log_interval = max(1, int(log_interval))

        self._layout_ready = False
        self._dv_names: List[str] = []
        self._dv_shapes: Dict[str, Tuple[int, ...]] = {}
        self._offsets: Dict[str, Tuple[int, int]] = {}
        self._full_dim = 0
        self._active_idx = None

        self._X: List[np.ndarray] = []
        # Per-function training buffers:
        #   Y[key]: scalar delta samples
        #   G[key]: gradient-delta samples (flattened reduced-DV space)
        self._Y: Dict[str, List[float]] = OrderedDict()
        self._G: Dict[str, List[np.ndarray]] = OrderedDict()
        self._alpha: Dict[str, np.ndarray] = OrderedDict()
        self._Xmat = None
        self._ell = None
        self._x_mu = None
        self._x_scale = None
        self._W = None
        self._as_dim = 0
        self._y_mu: Dict[str, float] = OrderedDict()
        self._y_scale: Dict[str, float] = OrderedDict()
        self._n_train = 0
        self._fitted = False
        self._just_initialized_layout = False
        self._predict_counter = 0
        self._not_ready_warned = False

    def has_model(self) -> bool:
        return self._fitted and self._n_train >= 2 and self._Xmat is not None

    def _is_rank0(self) -> bool:
        return (self.comm is None) or (self.comm.rank == 0)

    def _init_layout(self, x: Dict[str, Any], sens_delta_by_func: Dict[str, Dict[str, Any]]):
        # Build a stable flatten/unflatten map from structured DVs -> one vector.
        self._dv_names = list(x.keys())
        pos = 0
        for dv in self._dv_names:
            arr = np.asarray(x[dv], dtype=float)
            self._dv_shapes[dv] = arr.shape
            self._offsets[dv] = (pos, pos + arr.size)
            pos += arr.size
        self._full_dim = pos

        if self._full_dim <= self.max_dims:
            self._active_idx = np.arange(self._full_dim, dtype=np.int64)
        else:
            # Keep dimensions with strongest accumulated sensitivity magnitude.
            score = np.zeros((self._full_dim,), dtype=float)
            for _, grad_dict in sens_delta_by_func.items():
                score += np.abs(self._flatten_grad_full(grad_dict))
            if np.all(score <= 0.0):
                self._active_idx = np.arange(self.max_dims, dtype=np.int64)
            else:
                self._active_idx = np.argsort(score)[-self.max_dims:]
                self._active_idx.sort()
        self._layout_ready = True
        self._just_initialized_layout = True

    def _flatten_x_full(self, x: Dict[str, Any]) -> np.ndarray:
        flat = np.zeros((self._full_dim,), dtype=float)
        for dv in self._dv_names:
            s0, s1 = self._offsets[dv]
            flat[s0:s1] = np.asarray(x[dv], dtype=float).reshape(-1)
        return flat

    def _flatten_grad_full(self, grad_dict: Dict[str, Any]) -> np.ndarray:
        flat = np.zeros((self._full_dim,), dtype=float)
        for dv in self._dv_names:
            if dv not in grad_dict:
                continue
            s0, s1 = self._offsets[dv]
            flat[s0:s1] = np.asarray(grad_dict[dv], dtype=float).reshape(-1)
        return flat

    def _flatten_x(self, x: Dict[str, Any]) -> np.ndarray:
        return self._flatten_x_full(x)[self._active_idx]

    def _flatten_grad(self, grad_dict: Dict[str, Any]) -> np.ndarray:
        return self._flatten_grad_full(grad_dict)[self._active_idx]

    def _unflatten_grad(self, grad_red: np.ndarray) -> OrderedDict:
        grad_full = np.zeros((self._full_dim,), dtype=float)
        grad_full[self._active_idx] = grad_red
        out = OrderedDict()
        for dv in self._dv_names:
            s0, s1 = self._offsets[dv]
            out[dv] = grad_full[s0:s1].reshape(self._dv_shapes[dv]).copy()
        return out

    def _build_covariance(self, X: np.ndarray, ell: np.ndarray) -> np.ndarray:
        # GEK block covariance:
        # [K_ff  K_fg]
        # [K_gf  K_gg]
        n, d = X.shape
        nd = n * d
        diff = X[:, None, :] - X[None, :, :]
        inv_l2 = 1.0 / np.maximum(ell * ell, 1e-16)
        expo = -0.5 * np.sum((diff * np.sqrt(inv_l2)[None, None, :]) ** 2, axis=2)
        if self.stable_norm:
            expo = np.clip(expo, -self.exp_clip, 0.0)
        kff = np.exp(expo)

        kfg = np.zeros((n, nd), dtype=float)
        for j in range(n):
            b = j * d
            kfg[:, b:b + d] = kff[:, [j]] * (diff[:, j, :] * inv_l2[None, :])

        kgg = np.zeros((nd, nd), dtype=float)
        eye_l = np.diag(inv_l2)
        for i in range(n):
            bi = i * d
            for j in range(n):
                bj = j * d
                rij = diff[i, j, :]
                kgg[bi:bi + d, bj:bj + d] = kff[i, j] * (eye_l - np.outer(rij * inv_l2, rij * inv_l2))

        K = np.vstack([np.hstack([kff, kfg]), np.hstack([kfg.T, kgg])])
        K[np.diag_indices_from(K)] += self.nugget
        return K

    def _fit(self):
        self._fitted = False
        self._alpha = OrderedDict()
        n = len(self._X)
        self._n_train = n
        if n < 2:
            return

        X_raw = np.vstack(self._X)
        if self.stable_norm:
            # Normalize query space to avoid extreme distances / kernel underflow.
            self._x_mu = np.mean(X_raw, axis=0)
            x_std = np.std(X_raw, axis=0)
            self._x_scale = np.maximum(x_std, self.xscale_floor)
            X = (X_raw - self._x_mu[None, :]) / self._x_scale[None, :]
        else:
            self._x_mu = np.zeros((X_raw.shape[1],), dtype=float)
            self._x_scale = np.ones((X_raw.shape[1],), dtype=float)
            X = X_raw

        d = X.shape[1]
        if self.use_active_subspace and d > 1:
            # Learn active subspace from gradient covariance in normalized space.
            grads_all = []
            for func in self.ap.evalFuncs:
                key = f"{self.ap.name}_{func}"
                if key in self._G and len(self._G[key]) == n:
                    g_raw = np.vstack(self._G[key])
                    if self.stable_norm:
                        y_raw = np.asarray(self._Y[key], dtype=float)
                        y_scale = max(float(np.std(y_raw)), self.yscale_floor)
                        g_norm = (g_raw * self._x_scale[None, :]) / y_scale
                    else:
                        g_norm = g_raw
                    grads_all.append(g_norm)
            if grads_all:
                G = np.vstack(grads_all)
                C = (G.T @ G) / max(G.shape[0], 1)
                eigvals, eigvecs = np.linalg.eigh(C)
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                if np.sum(eigvals) > 0.0:
                    cum = np.cumsum(np.clip(eigvals, 0.0, None)) / np.sum(np.clip(eigvals, 0.0, None))
                    k_energy = int(np.searchsorted(cum, self.active_energy) + 1)
                else:
                    k_energy = 1
                k = min(self.active_rank, k_energy, d)
                self._W = eigvecs[:, :k]
                self._as_dim = int(k)
                X_use = X @ self._W
            else:
                self._W = np.eye(d)
                self._as_dim = d
                X_use = X
        else:
            self._W = np.eye(d)
            self._as_dim = d
            X_use = X

        ell = np.std(X_use, axis=0)
        ell = np.maximum(np.where(ell > 1e-12, ell, 1.0), self.ell_floor if self.stable_norm else 1e-12)
        K = self._build_covariance(X_use, ell)

        for func in self.ap.evalFuncs:
            key = f"{self.ap.name}_{func}"
            if key not in self._Y or key not in self._G:
                continue
            if len(self._Y[key]) != n or len(self._G[key]) != n:
                continue
            y_raw = np.asarray(self._Y[key], dtype=float)
            g_raw = np.vstack(self._G[key])
            if self.stable_norm:
                # Train in normalized output space to keep scales comparable.
                y_mu = float(np.mean(y_raw))
                y_scale = max(float(np.std(y_raw)), self.yscale_floor)
                self._y_mu[key] = y_mu
                self._y_scale[key] = y_scale
                y = (y_raw - y_mu) / y_scale
                # dz/dx_norm = (dy/dx_raw) * x_scale / y_scale
                g_norm = (g_raw * self._x_scale[None, :]) / y_scale
            else:
                self._y_mu[key] = 0.0
                self._y_scale[key] = 1.0
                y = y_raw
                g_norm = g_raw
            # Active-subspace gradient: dy/dz = dy/dx * W
            g = g_norm @ self._W
            g = g.reshape(-1)
            obs = np.concatenate([y, g])
            try:
                self._alpha[key] = np.linalg.solve(K, obs)
            except np.linalg.LinAlgError:
                self._alpha[key] = np.linalg.lstsq(K, obs, rcond=1e-12)[0]

        self._Xmat = X_use
        self._ell = ell
        self._fitted = len(self._alpha) > 0

    def add_sample(self, x: Dict[str, Any], func_delta: Dict[str, float], sens_delta: Dict[str, Dict[str, Any]]):
        if not self._layout_ready:
            self._init_layout(x, sens_delta)

        self._X.append(self._flatten_x(x))
        if len(self._X) > self.max_points:
            self._X.pop(0)

        for func in self.ap.evalFuncs:
            key = f"{self.ap.name}_{func}"
            self._Y.setdefault(key, [])
            self._G.setdefault(key, [])
            self._Y[key].append(float(func_delta.get(key, 0.0)))
            self._G[key].append(self._flatten_grad(sens_delta.get(key, {})))
            if len(self._Y[key]) > self.max_points:
                self._Y[key].pop(0)
            if len(self._G[key]) > self.max_points:
                self._G[key].pop(0)

        self._fit()

    def add_sample_logged(self, x: Dict[str, Any], func_delta: Dict[str, float], sens_delta: Dict[str, Dict[str, Any]], tag: str = "GEK"):
        """Train on one sample and print training messages on rank0."""
        self.add_sample(x, func_delta, sens_delta)
        if self._is_rank0():
            for msg in self.consume_training_messages(tag=tag):
                print(msg)

    def model_info(self) -> Dict[str, Any]:
        red_dim = int(self._active_idx.size) if self._active_idx is not None else 0
        n = int(self._n_train)
        n_obs = n * (1 + red_dim)
        return {
            "n_train": n,
            "full_dim": int(self._full_dim),
            "red_dim": red_dim,
            "n_obs": n_obs,
            "fitted_funcs": len(self._alpha),
            "layout_initialized": bool(self._just_initialized_layout),
            "stable_norm": self.stable_norm,
            "active_subspace": self.use_active_subspace,
            "as_dim": int(self._as_dim),
            "log_interval": int(self.log_interval),
        }

    def consume_layout_init_flag(self) -> bool:
        flag = self._just_initialized_layout
        self._just_initialized_layout = False
        return flag

    def consume_not_ready_message(self, tag: str = "GEK") -> Optional[str]:
        if self.has_model() or self._not_ready_warned:
            return None
        self._not_ready_warned = True
        info = self.model_info()
        return f"{tag} predict skipped: model not ready (samples={info['n_train']}, need >=2)."

    def consume_training_messages(self, tag: str = "GEK") -> List[str]:
        info = self.model_info()
        msgs: List[str] = []
        if self.consume_layout_init_flag():
            msgs.append(f"{tag} layout initialized: full_dim={info['full_dim']}, reduced_dim={info['red_dim']}.")
        msgs.append(
            f"{tag} correction updated: "
            f"samples={info['n_train']}, obs={info['n_obs']}, "
            f"fitted_funcs={info['fitted_funcs']}/{len(self.ap.evalFuncs)}, "
            f"as_dim={info['as_dim']}"
        )
        self._not_ready_warned = False
        return msgs

    def consume_predict_func_message(self, pred_funcs: Optional[Dict[str, float]], tag: str = "GEK") -> Optional[str]:
        if pred_funcs is None:
            return None
        self._predict_counter += 1
        if self._predict_counter % self.log_interval != 0:
            return None
        msg = [f"{tag} predict #{self._predict_counter}"]
        for func in self.ap.evalFuncs:
            key = f"{self.ap.name}_{func}"
            if key in pred_funcs:
                msg.append(f"{func}:d={pred_funcs[key]:+.6e}")
        return " | ".join(msg)

    def consume_predict_grad_message(self, pred_sens: Optional[Dict[str, Dict[str, Any]]], tag: str = "GEK") -> Optional[str]:
        if pred_sens is None:
            return None
        self._predict_counter += 1
        if self._predict_counter % self.log_interval != 0:
            return None
        norms = []
        for func in self.ap.evalFuncs:
            key = f"{self.ap.name}_{func}"
            if key in pred_sens:
                norm_val = 0.0
                for _, g in pred_sens[key].items():
                    norm_val += float(np.linalg.norm(np.asarray(g, dtype=float)))
                norms.append(f"{func}:|dg|={norm_val:.3e}")
        return f"{tag} grad predict | " + " | ".join(norms)

    def predict(self, x: Dict[str, Any], log_kind: Optional[str] = None, tag: str = "GEK"):
        if not self.has_model():
            if self._is_rank0() and log_kind is not None:
                msg = self.consume_not_ready_message(tag=tag)
                if msg:
                    print(msg)
            return None, None

        xq_raw = self._flatten_x(x)
        if self.stable_norm:
            xq = (xq_raw - self._x_mu) / self._x_scale
        else:
            xq = xq_raw
        X = self._Xmat
        ell = self._ell
        n, d = X.shape
        # Predict in active coordinates z when active subspace is enabled.
        x_use = xq @ self._W
        diff = X - x_use[None, :]
        inv_l2 = 1.0 / np.maximum(ell * ell, 1e-16)
        expo = -0.5 * np.sum((diff * np.sqrt(inv_l2)[None, :]) ** 2, axis=1)
        if self.stable_norm:
            expo = np.clip(expo, -self.exp_clip, 0.0)
        kff = np.exp(expo)

        k_grad_train = (x_use[None, :] - X) * inv_l2[None, :] * kff[:, None]
        ks_f = np.concatenate([kff, k_grad_train.reshape(-1)])

        pred_f = OrderedDict()
        pred_g = OrderedDict()
        eye_l = np.diag(inv_l2)
        for func in self.ap.evalFuncs:
            key = f"{self.ap.name}_{func}"
            if key not in self._alpha:
                continue
            alpha = self._alpha[key]
            pred_z = float(np.dot(ks_f, alpha))
            if self.stable_norm:
                # Undo output normalization back to physical CFD-ML delta.
                pred_f[key] = pred_z * self._y_scale[key] + self._y_mu[key]
            else:
                pred_f[key] = pred_z

            grad_z = np.zeros((d,), dtype=float)
            for b in range(d):
                kf_gb = diff[:, b] * inv_l2[b] * kff
                kg_gb = np.zeros((n, d), dtype=float)
                for i in range(n):
                    ri = X[i] - x_use
                    kg_gb[i, :] = kff[i] * (eye_l[:, b] - (ri * inv_l2) * (ri[b] * inv_l2[b]))
                ks_g = np.concatenate([kf_gb, kg_gb.reshape(-1)])
                grad_z[b] = float(np.dot(ks_g, alpha))
            # Back-map: dy/dx_norm = dy/dz * W^T
            grad_x_norm = grad_z @ self._W.T
            if self.stable_norm:
                # Undo x/y normalization to recover dy/dx in original DV space.
                grad_red = (grad_x_norm * self._y_scale[key]) / self._x_scale
            else:
                grad_red = grad_x_norm
            pred_g[key] = self._unflatten_grad(grad_red)

        if self._is_rank0() and log_kind == "func":
            msg = self.consume_predict_func_message(pred_f, tag=tag)
            if msg:
                print(msg)
        elif self._is_rank0() and log_kind == "grad":
            msg = self.consume_predict_grad_message(pred_g, tag=tag)
            if msg:
                print(msg)

        return pred_f, pred_g

class SolverCombined():
    """
    This class is a wrapper to combine the ML solver with a traditional CFD solver (e.g. ADflow) for multi-fidelity optimization. 
    The user can specify which functions to evaluate with the ML solver and which to evaluate with the CFD solver, and this class 
    will route the calls accordingly. This allows leveraging the speed of the ML solver for some functions while retaining the 
    accuracy of the CFD solver for others.
    
    """
    include_modes = {
        0: "just have a verfication",
        1: "CFD fail signal will be passed to optimizor",
        2: "single point replacement",
        3: "linear correction (with gradient to value)",
        4: "linear correction (without gradient to value)",
        10: "online GEK correction (train on CFD-ML delta from funcs + funcSens)",
        11: "online GEK correction with stable normalization (x/y normalization + safe kernel)",
        12: "active-subspace + normalized GEK correction",
    }
    # Default options for GEK modes (10/11/12). Pass `gek_options={...}` to override.
    gek_default_options = {
        "max_points": 100,        # GEK sample window length
        "max_dims": 40,           # Max DV dims passed into GEK before fitting
        "nugget": 1e-8,           # Covariance diagonal regularization
        "log_interval": 1,        # Print GEK prediction logs every N calls
        "xscale_floor": 1e-3,     # Mode11/12 x-normalization minimum scale
        "yscale_floor": 1e-8,     # Mode11/12 y-normalization minimum scale
        "ell_floor": 0.2,         # Mode11/12 minimum GP length scale
        "exp_clip": 60.0,         # Mode11/12 RBF exponent clipping magnitude
        "active_rank": 8,         # Mode12 max active-subspace rank
        "active_energy": 0.95,    # Mode12 target eig-energy for rank selection
    }

    def __init__(self, opt: OPT, ap: AeroProblem, comm=None, output_dir: str = "output",
                 cfd_frequency: int = 10, cfd_include_mode: int = 1, cfd_iter: int = 1,
                 gek_options: Optional[Dict[str, Any]] = None):
        '''

        majorIter updated at end of iteration
        0, 1, .... cfd_f-1, cfd_f, cfd_f+1, ... cfd_f+cfd_i
        ML         ML       ML     CFD          CFD         

        '''

        self.opt = opt
        self.ap = ap
        self.comm = comm
        self.cfd_frequency = cfd_frequency
        self.cfd_include_mode = cfd_include_mode
        self.cfd_iter = cfd_iter
        self.output_dir = output_dir
        self.enter_CFD = False
        # First-order correction model (CFD - ML) updated at each CFD run
        self._correction: dict = None
        # online GEK correction
        self.gek_options = dict(SolverCombined.gek_default_options)
        if gek_options is not None:
            unknown = set(gek_options.keys()) - set(self.gek_options.keys())
            if unknown:
                raise ValueError(f"Unknown GEK options: {sorted(unknown)}")
            self.gek_options.update(gek_options)
        if self.cfd_include_mode == 12:
            self._gek_tag = "GEK(AS)"
        elif self.cfd_include_mode == 11:
            self._gek_tag = "GEK(norm)"
        else:
            self._gek_tag = "GEK"
        self._gek = _OnlineGEKDeltaModel(
            ap=self.ap,
            comm=self.comm,
            stable_norm=(self.cfd_include_mode in [11, 12]),
            use_active_subspace=(self.cfd_include_mode == 12),
            **self.gek_options,
        )
        self._pending_gek = None

        if self.comm.rank == 0:
            print("==============================")
            print("ML-CFD combined solver defined")
            print(f"Current combination mode = {cfd_include_mode}")
            print(f"    -> {SolverCombined.include_modes.get(cfd_include_mode, 'unknown mode')}")
            print(f"ML every {cfd_frequency:d} / {(cfd_frequency + cfd_iter):d}; CFD every {cfd_iter:d} / {(cfd_frequency + cfd_iter):d}")
            if cfd_include_mode in [10, 11, 12]:
                print(
                    f"{self._gek_tag} settings: "
                    f"max_points={self.gek_options['max_points']}, "
                    f"max_dims={self.gek_options['max_dims']}, "
                    f"nugget={self.gek_options['nugget']:.1e}, "
                    f"log_interval={self.gek_options['log_interval']}"
                )
                if cfd_include_mode in [11, 12]:
                    print(
                        f"{self._gek_tag} stable norm: xscale_floor={self.gek_options['xscale_floor']:.1e}, "
                        f"yscale_floor={self.gek_options['yscale_floor']:.1e}, "
                        f"ell_floor={self.gek_options['ell_floor']:.2f}, exp_clip={self.gek_options['exp_clip']:.1f}"
                    )
                if cfd_include_mode == 12:
                    print(
                        f"{self._gek_tag} active subspace: "
                        f"rank={self.gek_options['active_rank']}, energy={self.gek_options['active_energy']:.2f}"
                    )
            print("==============================")

    @staticmethod
    def _copy_design_vars(x: Dict[str, Any]) -> OrderedDict:
        copied = OrderedDict()
        for k, v in x.items():
            arr = np.asarray(v, dtype=float)
            copied[k] = float(arr.reshape(-1)[0]) if arr.size == 1 else arr.copy()
        return copied

    @staticmethod
    def update_funcs(ap: AeroProblem, ml_funcs, cfd_funcs):
        
        for func in ap.evalFuncs:
            ml_funcs[f"{ap.name}_{func}"] = cfd_funcs[f"{ap.name}_{func}"]
        if "fail" in cfd_funcs:
            ml_funcs["fail"] = cfd_funcs["fail"]

        return ml_funcs

    def wrap_cruiseFuncs(self, _ml_solver: Callable, _cfd_solver: Callable, _pre: Optional[Callable] = None, _post: Optional[Callable] = None):
        '''
        This function will return a function that is called by the optimizer at each iteration with the current design variables `x`.
        It combines the ML solver and the CFD solver according to the specified function keys and the optimization iteration number.
        
        '''
        def cruiseFuncs(x):
 

            opt_iter = self.opt.iterCounter
            major_opt_iter = self.opt.majorIterCounter

            with _rank0_guard(self.comm) as is_rank0:
                if is_rank0:
                    print(f'opt iter: {opt_iter}')

            self.ap.setDesignVars(x)

            if _pre is not None:
                _pre(self.ap, x)

            funcs = _ml_solver(self.ap)

            trigger_portion = (major_opt_iter + 1) % (self.cfd_frequency + self.cfd_iter)
            self.enter_CFD = ((self.cfd_iter <= 0) and (major_opt_iter+1 >= self.cfd_frequency)) or \
                             (self.cfd_iter > 0 and self.cfd_frequency > 0 and trigger_portion >= self.cfd_frequency)

            self.enter_CFD = self.comm.bcast(self.enter_CFD, root=0)

            if self.enter_CFD:
                cfd_funcs = {}
                with _rank0_guard(self.comm) as is_rank0:
                    if is_rank0:
                        print(f"")
                        print(f">>>>>>>>>>>>>>>>>>>>>>>>")
                        print(f"Running Solver (evaluation {opt_iter} / major iteration {major_opt_iter})")

                # print(f'Rank {comm.rank}: Starting CFD solve...')
                cfd_funcs = _cfd_solver(self.ap)

                with _rank0_guard(self.comm) as is_rank0:
                    if is_rank0:
                        row = {"iter": opt_iter, "time": time.time() - self.opt.startTime, "major_iter": major_opt_iter, "ml_fail": funcs.get("fail", 0.0), "cfd_fail": cfd_funcs.get("fail", 0.0)}
                        # update ML and CFD predictions to the writting row
                        for func in self.ap.evalFuncs:
                            row[f"ml_{self.ap.name}_{func}"] = funcs[f"{self.ap.name}_{func}"]
                            row[f"cfd_{self.ap.name}_{func}"] = cfd_funcs[f"{self.ap.name}_{func}"]

                        # Store xuser variables into cfd.hst with a consistent prefix
                        for k, v in x.items():
                            row[f"xuser_{k}"] = v

                        _append_cfd_hist(os.path.join(self.output_dir, "cfd.hst"), row)
                        # simluated_iters.append(opt_iter)
                        print('current cfd funcs', row)

                if self.cfd_include_mode == 0:
                    self.ap.solveFailed = False
                    self.ap.fatalFail = False
                elif self.cfd_include_mode == 1:
                    pass # to update fail information
                elif self.cfd_include_mode == 2:
                    # use CFD solver results to replace ML's
                    # print(f"rank {self.comm.rank} enter 2")
                    funcs = self.update_funcs(self.ap, funcs, cfd_funcs)
                elif self.cfd_include_mode in [10, 11, 12]:
                    # GEK modes: keep ML and CFD function values at the same CFD point.
                    # The paired sensitivities are added in wrap_cruiseFuncsSens.
                    # For GEK training, keep both ML and CFD values at the CFD point.
                    self._pending_gek = {
                        "x": self._copy_design_vars(x),
                        "ml_funcs": OrderedDict((f"{self.ap.name}_{func}", funcs[f"{self.ap.name}_{func}"]) for func in self.ap.evalFuncs),
                        "cfd_funcs": OrderedDict((f"{self.ap.name}_{func}", cfd_funcs[f"{self.ap.name}_{func}"]) for func in self.ap.evalFuncs),
                    }
                    funcs = self.update_funcs(self.ap, funcs, cfd_funcs)
                elif self.cfd_include_mode in [3, 4]:
                    # build first-order correction model and return CFD results
                    self._correction = {
                        "x": x,
                        "funcs": {},
                        "funcSens": OrderedDict()
                    }
                    for func in self.ap.evalFuncs:
                        key = f"{self.ap.name}_{func}"
                        if key in cfd_funcs and key in funcs:
                            self._correction["funcs"][key] = cfd_funcs[key] - funcs[key]
                    funcs = self.update_funcs(self.ap, funcs, cfd_funcs)

            else:
                # print(f'Mode: {self.cfd_include_mode}, _correction is None?: {bool(self._correction)}')
                if self.cfd_include_mode in [3, 4] and self._correction:
                    # apply correction model to ML results between CFD updates
                    for func in self.ap.evalFuncs:
                        key = f"{self.ap.name}_{func}"
                        funcs[key] += self._correction["funcs"][key]

                        if self.cfd_include_mode == 3:
                            for dv, delta in self._correction["funcSens"][key].items():
                                # print(dv, x[dv].shape, self._correction["x"][dv].shape, delta.shape)
                                contrib = np.dot(delta, (x[dv] - self._correction["x"][dv]))
                                if isinstance(contrib, np.ndarray) and contrib.shape in [(), (1,)]:
                                    contrib = float(contrib.reshape(-1)[0])
                                funcs[key] += contrib
                elif self.cfd_include_mode in [10, 11, 12]:
                    # Non-CFD iteration: use GEK to predict correction delta and add it to ML output.
                    if self.comm is None or self.comm.rank == 0:
                        pred_funcs, _ = self._gek.predict(x, log_kind="func", tag=self._gek_tag)
                    else:
                        pred_funcs = None
                    if self.comm is not None:
                        pred_funcs = self.comm.bcast(pred_funcs, root=0)
                    if pred_funcs is not None:
                        for func in self.ap.evalFuncs:
                            key = f"{self.ap.name}_{func}"
                            if key in pred_funcs and key in funcs:
                                funcs[key] += pred_funcs[key]

            # input()
            return funcs
        
        return cruiseFuncs
    
    def wrap_cruiseFuncsSens(self, _ml_solver: Callable, _cfd_solver: Callable, _pre: Optional[Callable] = None, _post: Optional[Callable] = None):
        '''
        This function will return a function that is called by the optimizer at each iteration with the current design variables `x`.
        It combines the ML solver and the CFD solver according to the specified function keys and the optimization iteration number.
        
        '''

        def cruiseFuncsSens(x, funcs):
            
            if _pre is not None:
                _pre(self.ap, x, funcs)

            funcsSens = _ml_solver(self.ap, x=x, funcs=funcs)

            if self.cfd_include_mode in [0, 1]:
                pass
            else:
                if self.enter_CFD:

                    with _rank0_guard(self.comm) as is_rank0:
                        if is_rank0:
                            print(f"")
                            print(f">>>>>>>>>>>>>>>>>>>>>>>>")
                            print(f"Running Solver Adjoint")

                    cfd_funcsSens = _cfd_solver(self.ap, x=x, funcs=funcs)

                    with _rank0_guard(self.comm) as is_rank0:
                        if is_rank0:
                            row = {"ml_sen_fail": funcsSens.get("fail", 0.0), "cfd_sen_fail": cfd_funcsSens.get("fail", 0.0)}
                            for func in self.ap.evalFuncs:
                                row[f"ml_sen_{self.ap.name}_{func}"] = funcsSens[f"{self.ap.name}_{func}"]
                                row[f"cfd_sen_{self.ap.name}_{func}"] = cfd_funcsSens[f"{self.ap.name}_{func}"]
                            
                            print('current cfd recording funcsSens', row)
                            _append_cfd_hist(os.path.join(self.output_dir, "cfd.hst"), row)

                    if self.cfd_include_mode == 2:
                        # Update funcsSens with CFD sensitivities
                        funcsSens = self.update_funcs(self.ap, funcsSens, cfd_funcsSens)
                    elif self.cfd_include_mode in [10, 11, 12]:
                        # CFD iteration: build one GEK sample from (CFD-ML) value and gradient deltas.
                        if self._pending_gek is not None:
                            if self.comm is None or self.comm.rank == 0:
                                func_delta = OrderedDict()
                                sens_delta = OrderedDict()
                                for func in self.ap.evalFuncs:
                                    key = f"{self.ap.name}_{func}"
                                    func_delta[key] = self._pending_gek["cfd_funcs"][key] - self._pending_gek["ml_funcs"][key]
                                    sens_delta[key] = OrderedDict()
                                    for dv, cfd_val in cfd_funcsSens[key].items():
                                        sens_delta[key][dv] = cfd_val - funcsSens[key][dv]
                                self._gek.add_sample_logged(self._pending_gek["x"], func_delta, sens_delta, tag=self._gek_tag)
                        self._pending_gek = None
                        funcsSens = self.update_funcs(self.ap, funcsSens, cfd_funcsSens)

                    elif self.cfd_include_mode >= 3:

                        for func in self.ap.evalFuncs:
                            key = f"{self.ap.name}_{func}"
                            # for initialization
                            if key not in self._correction["funcSens"]:
                                self._correction["funcSens"][key] = OrderedDict()
                            for dv, cfd_val in cfd_funcsSens[key].items():
                                self._correction["funcSens"][key][dv] = cfd_val - funcsSens[key][dv]
                        # Ensure funcsSens at CFD point matches high-fidelity results
                        funcsSens = self.update_funcs(self.ap, funcsSens, cfd_funcsSens)

                else:
                    if self.cfd_include_mode in [3,4] and self._correction:
                        for func in self.ap.evalFuncs:
                            key = f"{self.ap.name}_{func}"
                            for dv, delta in self._correction["funcSens"][key].items():
                                funcsSens[key][dv] += delta
                    elif self.cfd_include_mode in [10, 11, 12]:
                        # Non-CFD iteration: correct ML sensitivities with GEK-predicted gradient deltas.
                        if self.comm is None or self.comm.rank == 0:
                            _, pred_sens = self._gek.predict(x, log_kind="grad", tag=self._gek_tag)
                        else:
                            pred_sens = None
                        if self.comm is not None:
                            pred_sens = self.comm.bcast(pred_sens, root=0)
                        if pred_sens is not None:
                            for func in self.ap.evalFuncs:
                                key = f"{self.ap.name}_{func}"
                                if key not in pred_sens or key not in funcsSens:
                                    continue
                                for dv, dval in pred_sens[key].items():
                                    if dv in funcsSens[key]:
                                        funcsSens[key][dv] += dval

            # if comm.rank == 0:
            #     print('current cfd funcs sen', funcsSens)
            return funcsSens
        
        return cruiseFuncsSens
