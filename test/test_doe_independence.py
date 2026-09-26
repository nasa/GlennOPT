"""Generated designs must own their result parameters independently."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from glennopt.DOE import BoxBehnken, CCD, Default, FullFactorial, LatinHyperCube
from glennopt.base import Optimizer, Parameter


class TestDOEIndependence(unittest.TestCase):
    factories = (
        lambda: Default(4),
        lambda: FullFactorial(levels=2),
        lambda: LatinHyperCube(samples=4),
        CCD,
        BoxBehnken,
    )

    def design(self, factory=FullFactorial):
        doe = factory()
        for name in ("x", "y", "z"):
            doe.add_parameter(name, min_value=0, max_value=1)
        doe.objectives = [
            Parameter("cost", value=-3, test_value=12),
            Parameter("drag", value=2, value_if_failed=999),
        ]
        doe.perf_parameters = [
            Parameter("mass", value=5, constraint_less_than=20),
            Parameter("stress", value=6, constraint_greater_than=0),
        ]
        return doe

    def test_each_design_owns_its_objectives_and_performance_parameters(self):
        for factory in self.factories:
            with self.subTest(factory=factory):
                doe = self.design(factory)
                templates = copy.deepcopy([doe.objectives, doe.perf_parameters])
                individuals = doe.generate_doe()
                self.assertGreater(len(individuals), 1)
                individuals[0].set_objective("COST", 101)
                individuals[0].set_performance_parameter("MASS", 202)
                self.assertEqual(individuals[0].get_objective("cost"), 101)
                self.assertEqual(individuals[0].get_performance_parameter("mass"), 202)
                for other in individuals[1:]:
                    self.assertEqual(other.get_objectives_list(), templates[0])
                    self.assertEqual(
                        other.get_performance_parameters_list(), templates[1]
                    )
                self.assertEqual([doe.objectives, doe.perf_parameters], templates)

    def test_all_parameter_lists_and_objects_are_distinct(self):
        doe = self.design(lambda: FullFactorial(levels=2))
        individuals = doe.generate_doe()
        roles = [
            ("get_eval_parameter_list", doe.eval_parameters),
            ("get_objectives_list", doe.objectives),
            ("get_performance_parameters_list", doe.perf_parameters),
        ]
        for method, template in roles:
            with self.subTest(method=method):
                lists = [getattr(ind, method)() for ind in individuals] + [template]
                self.assertEqual(len({id(values) for values in lists}), len(lists))
                for column in range(len(template)):
                    self.assertEqual(
                        len({id(values[column]) for values in lists}), len(lists)
                    )

    def test_parameter_metadata_and_list_edits_do_not_leak(self):
        doe = self.design(lambda: Default(3))
        first, second, third = doe.generate_doe()
        original = copy.deepcopy([doe.objectives, doe.perf_parameters])
        first.get_objectives_list()[0].value_if_failed = -900
        first.get_objectives_list().append(Parameter("extra"))
        first.get_performance_parameters_list()[0].constraint_less_than = -1
        first.get_performance_parameters_list().pop()
        for other in (second, third):
            self.assertEqual(other.get_objectives_list(), original[0])
            self.assertEqual(other.get_performance_parameters_list(), original[1])
            self.assertEqual(other.get_objectives_list()[0].test_value, 12)

    def test_repeated_generations_do_not_share_result_objects(self):
        doe = self.design(lambda: Default(1))
        first = doe.generate_doe()[0]
        second = doe.generate_doe()[0]
        second.set_objective("cost", 33)
        second.set_performance_parameter("mass", 44)
        self.assertEqual(first.get_objective("cost"), -3)
        self.assertEqual(first.get_performance_parameter("mass"), 5)
        self.assertEqual(doe.objectives[0].value, -3)
        self.assertEqual(doe.perf_parameters[0].value, 5)

    def test_empty_result_lists_can_be_changed_independently(self):
        doe = Default(2)
        doe.add_parameter("x", 0, 1)
        first, second = doe.generate_doe()
        first.get_objectives_list().append(Parameter("new"))
        first.get_performance_parameters_list().append(Parameter("new"))
        self.assertEqual(second.get_objectives_list(), [])
        self.assertEqual(second.get_performance_parameters_list(), [])
        self.assertEqual(doe.objectives, [])
        self.assertEqual(doe.perf_parameters, [])

    def test_reading_each_evaluation_output_preserves_previous_results(self):
        doe = self.design(lambda: Default(3))
        individuals = doe.generate_doe()
        with TemporaryDirectory() as directory:
            optimizer = Optimizer(
                "test", "unused", eval_folder=directory, opt_folder=directory
            )
            optimizer.output_file = str(Path(directory) / "output.txt")
            for index, individual in enumerate(individuals):
                Path(optimizer.output_file).write_text(
                    f"# Simulation results\ncost = {index + 1}\nmass = {index + 11}\n"
                )
                returned = optimizer.__read_output_file__(individual)
                self.assertIs(returned, individual)
            self.assertEqual([i.get_objective("cost") for i in individuals], [1, 2, 3])
            self.assertEqual(
                [i.get_performance_parameter("mass") for i in individuals], [11, 12, 13]
            )
            self.assertEqual(doe.objectives[0].value, -3)
            self.assertEqual(doe.perf_parameters[0].value, 5)

    def test_factorial_design_values_and_evaluation_independence_are_preserved(self):
        doe = self.design(lambda: FullFactorial(levels=2))
        expected = doe.create_design().to_numpy()
        original = copy.deepcopy(doe.eval_parameters)
        individuals = doe.generate_doe()
        np.testing.assert_array_equal(
            [ind.eval_parameters for ind in individuals], expected
        )
        individuals[0].set_eval_parameter_at_indx(0, 123)
        np.testing.assert_array_equal(individuals[1].eval_parameters, expected[1])
        self.assertEqual(doe.eval_parameters, original)


if __name__ == "__main__":
    unittest.main()
