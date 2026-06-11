"""Load a GlennOPT optimization configuration from a YAML file.

Three entry points, in order of typical use:

1. :func:`optimizer_from_yaml(path, optimization_folder)` — fully YAML-driven
   construction. The optimizer class (NSGA3 / SODE / ...) is selected by the
   YAML's ``optimizer.type`` field.

2. :meth:`Optimizer.add_yaml(path)` — augment an already-constructed optimizer
   with eval params, objectives, parallel/mutation settings, and DOE from a
   YAML. Useful when you want to keep manual control of construction (e.g. to
   pre-set ``optimization_folder`` from a runtime argument) but YAML-drive the
   rest.

3. :func:`load_optimization_yaml(path)` — low-level: parse the YAML into an
   :class:`OptimizationConfig` you can inspect or post-process.

YAML schema:

.. code-block:: yaml

    optimizer:
      type: NSGA3                  # NSGA3 | SODE
      pop_size: 12
      pareto_resolution: 4         # NSGA3-only
      eval_command: "python evaluate.py"
      eval_folder: Evaluation
      single_folder_eval: false
      overwrite_input_file: false

    parallel:
      concurrent_executions: 3
      cores_per_execution: 1
      execution_timeout: 30        # minutes

    mutation:
      type: de_rand_1_bin          # member of de_mutation_type
      mu: 0.02
      sigma: 0.2
      F: 0.6
      C: 0.7

    doe:
      type: LatinHyperCube         # LatinHyperCube | Default | CCD | BoxBehnken | FullFactorial
      samples: 64
      levels: 4

    eval_parameters:
      x1: { min: 22.0, max: 50.0, test: 30.0 }
      stagger: { min: 30.0, max: 60.0, test: 45.0, value_if_failed: 9999.0 }

    objectives:
      omega: { min: 0.0, max: 1.0 }
      mass_imbalance: { min: 0.0, max: 1.0 }

    performance_parameters:
      Pt_in: { min: 0.0, max: 200000.0 }

The ``test`` field is optional and stored on each Parameter as
``parameter.test_value`` for downstream tooling (e.g. running a smoke-test
evaluation at the nominal design point). The optimizer itself ignores it.

Examples::

    from glennopt.helpers import optimizer_from_yaml
    opt = optimizer_from_yaml("opt.yaml", optimization_folder=os.getcwd())
    opt.start_doe(opt.doe.generate_doe())
    opt.optimize_from_population(pop_start=-1, n_generations=10)

    # Or, augment an existing optimizer:
    from glennopt.optimizers.sode import SODE
    sode = SODE(optimization_folder=os.getcwd(), eval_folder=str(eval_dir))
    sode.add_yaml("opt.yaml")
    sode.start_doe(sode.doe.generate_doe())
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..base.parameter import Parameter
from ..DOE.Experiment import (
    BoxBehnken,
    CCD,
    Default,
    FullFactorial,
    LatinHyperCube,
)
from .mutate import de_mutation_type, mutation_parameters
from .parallel_settings import parallel_settings


_DOE_CLASSES = {
    "Default": Default,
    "LatinHyperCube": LatinHyperCube,
    "CCD": CCD,
    "BoxBehnken": BoxBehnken,
    "FullFactorial": FullFactorial,
}


def _optimizer_classes() -> dict:
    """Lazily build the optimizer class registry to avoid circular imports."""
    from ..optimizers.nsga3 import NSGA3
    from ..optimizers.sode import SODE

    classes = {"NSGA3": NSGA3, "SODE": SODE}
    try:
        from ..optimizers.nsga3_ml import NSGA3_ML
        classes["NSGA3_ML"] = NSGA3_ML
    except Exception:
        pass
    try:
        from ..optimizers.nsopt import NSGAOpt
        classes["NSGAOpt"] = NSGAOpt
    except Exception:
        pass
    return classes


def _make_parameter(name: str, spec: dict) -> Parameter:
    """Build a single Parameter from a YAML mapping."""
    if "min" not in spec or "max" not in spec:
        raise ValueError(
            f"Parameter '{name}' must have both 'min' and 'max' keys; got {spec}"
        )
    p = Parameter(
        name=name,
        min_value=float(spec["min"]),
        max_value=float(spec["max"]),
        value_if_failed=float(spec.get("value_if_failed", 10000.0)),
        constraint_greater_than=spec.get("constraint_greater_than"),
        constraint_less_than=spec.get("constraint_less_than"),
    )
    if "test" in spec:
        p.test_value = float(spec["test"])
    return p


def _params_from_block(block: dict) -> list[Parameter]:
    if not block:
        return []
    return [_make_parameter(name, spec) for name, spec in block.items()]


@dataclass
class OptimizationConfig:
    """Resolved YAML configuration."""

    raw: dict
    optimizer_type: str
    optimizer_kwargs: dict
    parallel: parallel_settings
    mutation: mutation_parameters
    eval_parameters: list[Parameter]
    objectives: list[Parameter]
    performance_parameters: list[Parameter]
    doe: object  # one of the DOE classes with parameters/objectives already added

    def apply_to(self, optimizer):
        """Apply parsed YAML to an already-constructed optimizer.

        Sets:
          - eval_parameters / objectives / performance_parameters (whichever
            apply on the optimizer subclass)
          - parallel_settings, mutation_params
          - optimizer-level kwargs that are settable post-construction
            (pop_size, pareto_resolution, eval_command); skips eval_folder /
            single_folder_eval / overwrite_input_file because those are
            validated at __init__.
          - caches the DOE on ``optimizer.doe`` for later
            ``optimizer.start_doe(optimizer.doe.generate_doe())``
        """
        for k, v in self.optimizer_kwargs.items():
            if k == "eval_command":
                optimizer.evaluation_command = v
            elif k == "pop_size" and hasattr(optimizer, "pop_size"):
                optimizer.pop_size = int(v)
            elif k == "pareto_resolution" and hasattr(optimizer, "pareto_resolution"):
                optimizer.pareto_resolution = int(v)
            # eval_folder / single_folder_eval / overwrite_input_file are
            # validated at __init__; silently ignore here.

        if self.eval_parameters:
            optimizer.add_eval_parameters(self.eval_parameters)
        if self.objectives:
            optimizer.add_objectives(self.objectives)
        if self.performance_parameters and hasattr(optimizer, "add_performance_parameters"):
            optimizer.add_performance_parameters(self.performance_parameters)

        optimizer.parallel_settings = self.parallel
        optimizer.mutation_params = self.mutation
        optimizer.doe = self.doe
        return optimizer

    def build_optimizer(self, optimization_folder: str):
        """Construct a fresh optimizer of the type declared in the YAML."""
        classes = _optimizer_classes()
        cls = classes.get(self.optimizer_type)
        if cls is None:
            raise ValueError(
                f"Unknown optimizer.type {self.optimizer_type!r}; "
                f"expected one of {sorted(classes)}"
            )

        # Filter optimizer_kwargs to only what this class's __init__ accepts.
        sig = inspect.signature(cls.__init__)
        accepted = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k, v in self.optimizer_kwargs.items() if k in accepted}
        opt = cls(optimization_folder=optimization_folder, **filtered)
        return self.apply_to(opt)


def _build_doe(doe_block: dict, eval_params, objectives, perf_params):
    """Build the DOE instance and attach parameters/objectives to it."""
    doe_type = doe_block.get("type", "LatinHyperCube")
    cls = _DOE_CLASSES.get(doe_type)
    if cls is None:
        raise ValueError(
            f"Unknown DOE type {doe_type!r}; expected one of {list(_DOE_CLASSES)}"
        )
    kwargs = {k: v for k, v in doe_block.items() if k != "type"}
    doe = cls(**kwargs)
    for p in eval_params:
        doe.add_parameter(
            name=p.name,
            min_value=p.min_value,
            max_value=p.max_value,
            value_if_failed=p.value_if_failed,
            constr_less_than=p.constraint_less_than,
            constr_greater_than=p.constraint_greater_than,
        )
    for o in objectives:
        doe.add_objectives(name=o.name)
    for pp in perf_params:
        doe.add_perf_parameter(name=pp.name)
    return doe


def load_optimization_yaml(path) -> OptimizationConfig:
    """Parse a GlennOPT YAML config file into an :class:`OptimizationConfig`."""
    with open(Path(path)) as f:
        raw = yaml.safe_load(f)

    opt_block = raw.get("optimizer", {})
    optimizer_type = opt_block.get("type", "NSGA3")
    optimizer_kwargs = {k: v for k, v in opt_block.items() if k != "type"}

    par_block = raw.get("parallel", {})
    par = parallel_settings(
        concurrent_executions=int(par_block.get("concurrent_executions", 1)),
        cores_per_execution=int(par_block.get("cores_per_execution", 1)),
        execution_timeout=int(par_block.get("execution_timeout", 10)),
    )

    mut_block = raw.get("mutation", {})
    mut = mutation_parameters()
    if "type" in mut_block:
        try:
            mut.mutation_type = de_mutation_type[mut_block["type"]]
        except KeyError as e:
            raise ValueError(
                f"Unknown mutation type {mut_block['type']!r}; "
                f"expected one of {[m.name for m in de_mutation_type]}"
            ) from e
    for key in ("mu", "sigma", "F", "C"):
        if key in mut_block:
            setattr(mut, key, float(mut_block[key]))

    eval_params = _params_from_block(raw.get("eval_parameters", {}))
    objectives = _params_from_block(raw.get("objectives", {}))
    perf_params = _params_from_block(raw.get("performance_parameters", {}))

    doe_block = raw.get("doe", {"type": "LatinHyperCube", "samples": 64, "levels": 4})
    doe = _build_doe(doe_block, eval_params, objectives, perf_params)

    return OptimizationConfig(
        raw=raw,
        optimizer_type=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        parallel=par,
        mutation=mut,
        eval_parameters=eval_params,
        objectives=objectives,
        performance_parameters=perf_params,
        doe=doe,
    )


def optimizer_from_yaml(path, optimization_folder: str):
    """Build an optimizer entirely from a YAML file.

    The optimizer class is chosen by ``optimizer.type`` in the YAML
    (NSGA3 | SODE | NSGA3_ML | NSGAOpt).
    """
    cfg = load_optimization_yaml(path)
    return cfg.build_optimizer(optimization_folder)
