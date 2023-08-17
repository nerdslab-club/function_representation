import inspect
import re
import types
from typing import get_type_hints
from collections import defaultdict

from .math_functions import MathFunctions as mF


class FunctionManager:
    def __init__(self):
        self.name_to_reference = {}
        self.reference_to_name = defaultdict(str)
        self.positive_negative_function_map = {}
        self.vector_arithmetics = []
        self._set_name_to_reference()
        self._set_reference_to_name()
        self._set_vector_arithmetics()
        self._set_positive_negative_function_map()

    def _set_name_to_reference(self):
        self.name_to_reference = {
            "addition": mF.addition,
            "subtraction": mF.subtraction,
            "multiplication": mF.multiplication,
            "division": mF.division,
            "exponentiation": mF.exponentiation,
            "square_root": mF.square_root,
            "floor_division": mF.floor_division,
            "modulus": mF.modulus,
            "logarithm": mF.logarithm,
            "sine": mF.sine,
            "cosine": mF.cosine,
            "tangent": mF.tangent,
            "arcsine": mF.arcsine,
            "arccosine": mF.arccosine,
            "arctangent": mF.arctangent,
            "hyperbolic_sine": mF.hyperbolic_sine,
            "hyperbolic_cosine": mF.hyperbolic_cosine,
            "hyperbolic_tangent": mF.hyperbolic_tangent,
            "logarithm_base_10": mF.logarithm_base_10,
            "logarithm_base_2": mF.logarithm_base_2,
            "degrees_to_radians": mF.degrees_to_radians,
            "radians_to_degrees": mF.radians_to_degrees,
            "gcd": mF.gcd,
            "lcm": mF.lcm,
            "isqrt": mF.isqrt,
            "pow_mod": mF.pow_mod,
            "ceil": mF.ceil,
            "floor": mF.floor,
            "round": mF.round,
            "absolute_difference": mF.absolute_difference,
            "greatest_value": mF.greatest_value,
            "smallest_value": mF.smallest_value,
            "product": mF.product,
            "factorial": mF.factorial,
            "is_prime": mF.is_prime,
            "prime_factors": mF.prime_factors,
            "is_perfect_square": mF.is_perfect_square,
            "is_perfect_cube": mF.is_perfect_cube,
            "mean": mF.mean,
            "median": mF.median,
            "relu": mF.relu,
            "ascending_sort": mF.ascending_sort,
            "descending_sort": mF.descending_sort,
            "square_int": mF.square_int,
            "square": mF.square,
            "absolute": mF.absolute,
            "power_of_ten": mF.power_of_ten,
            "cube": mF.cube,
            "cube_root": mF.cube_root,
            "is_even": mF.is_even,
            "is_odd": mF.is_odd,
            "max_value": mF.max_value,
            "min_value": mF.min_value,
            "nth_root": mF.nth_root,
            "geometric_mean": mF.geometric_mean,
            "is_power_of_two": mF.is_power_of_two,
            "binary_to_decimal": mF.binary_to_decimal,
            "decimal_to_binary": mF.decimal_to_binary,
            "is_palindrome": mF.is_palindrome,
            "sum_of_digits": mF.sum_of_digits,
            "hypotenuse": mF.hypotenuse,
            "circle_area": mF.circle_area,
            "permutation": mF.permutation,
            "combination": mF.combination,
            "geometric_series_sum": mF.geometric_series_sum,
            "sigmoid": mF.sigmoid,
            "cosine_similarity": mF.cosine_similarity,
            "euler_totient": mF.euler_totient,
            "l1_norm": mF.l1_norm,
            "l2_norm": mF.l2_norm,
            "average": mF.average,
            "sum": mF.sum,
            "length": mF.length,
            "a_plus_b_whole_square": mF.a_plus_b_whole_square,
            "a_squared_plus_2ab_plus_b_squared": mF.a_squared_plus_2ab_plus_b_squared,
            "a_minus_b_whole_squared_plus_4ab": mF.a_minus_b_whole_squared_plus_4ab,
            "a_minus_b_whole_squared": mF.a_minus_b_whole_squared,
            "a_squared_minus_2ab_plus_b_squared": mF.a_squared_minus_2ab_plus_b_squared,
            "a_plus_b_whole_squared_minus_4ab": mF.a_plus_b_whole_squared_minus_4ab,
            "a_squared_plus_b_squared": mF.a_squared_plus_b_squared,
            "negative_2ab": mF.negative_2ab,
            "positive_2ab": mF.positive_2ab,
            "x_plus_a_times_x_plus_b": mF.x_plus_a_times_x_plus_b,
            "x_squared_plus_a_plus_b_times_x_plus_ab": mF.x_squared_plus_a_plus_b_times_x_plus_ab,
            "a_cubed_plus_b_cubed": mF.a_cubed_plus_b_cubed,
            "a_plus_b_whole_cubed_minus_3ab_times_a_plus_b": mF.a_plus_b_whole_cubed_minus_3ab_times_a_plus_b,
            "a_plus_b_times_a_squared_minus_ab_plus_b_squared": mF.a_plus_b_times_a_squared_minus_ab_plus_b_squared,
            "a_cubed_minus_b_cubed": mF.a_cubed_minus_b_cubed,
            "a_minus_b_whole_cubed_plus_3ab_times_a_minus_b": mF.a_minus_b_whole_cubed_plus_3ab_times_a_minus_b,
            "a_minus_b_times_a_squared_plus_ab_plus_b_squared": mF.a_minus_b_times_a_squared_plus_ab_plus_b_squared,
            "invert_number": mF.invert_number,
            "float_to_int": mF.float_to_int,
            "int_to_float": mF.int_to_float,
            "check_same_string": mF.check_same_string,
            "reverse_string": mF.reverse_string,
            "get_pi": mF.get_pi,
            "get_e": mF.get_e,
            "calculate_dot_product": mF.calculate_dot_product,
        }

    def _set_reference_to_name(self):
        self.reference_to_name = {
            mF.addition: "addition",
            mF.subtraction: "subtraction",
            mF.multiplication: "multiplication",
            mF.division: "division",
            mF.exponentiation: "exponentiation",
            mF.square_root: "square_root",
            mF.floor_division: "floor_division",
            mF.modulus: "modulus",
            mF.logarithm: "logarithm",
            mF.sine: "sine",
            mF.cosine: "cosine",
            mF.tangent: "tangent",
            mF.arcsine: "arcsine",
            mF.arccosine: "arccosine",
            mF.arctangent: "arctangent",
            mF.hyperbolic_sine: "hyperbolic_sine",
            mF.hyperbolic_cosine: "hyperbolic_cosine",
            mF.hyperbolic_tangent: "hyperbolic_tangent",
            mF.logarithm_base_10: "logarithm_base_10",
            mF.logarithm_base_2: "logarithm_base_2",
            mF.degrees_to_radians: "degrees_to_radians",
            mF.radians_to_degrees: "radians_to_degrees",
            mF.gcd: "gcd",
            mF.lcm: "lcm",
            mF.isqrt: "isqrt",
            mF.pow_mod: "pow_mod",
            mF.ceil: "ceil",
            mF.floor: "floor",
            mF.round: "round",
            mF.absolute_difference: "absolute_difference",
            mF.greatest_value: "greatest_value",
            mF.smallest_value: "smallest_value",
            mF.product: "product",
            mF.factorial: "factorial",
            mF.is_prime: "is_prime",
            mF.prime_factors: "prime_factors",
            mF.is_perfect_square: "is_perfect_square",
            mF.is_perfect_cube: "is_perfect_cube",
            mF.mean: "mean",
            mF.median: "median",
            mF.relu: "relu",
            mF.ascending_sort: "ascending_sort",
            mF.descending_sort: "descending_sort",
            mF.square_int: "square_int",
            mF.square: "square",
            mF.absolute: "absolute",
            mF.power_of_ten: "power_of_ten",
            mF.cube: "cube",
            mF.cube_root: "cube_root",
            mF.is_even: "is_even",
            mF.is_odd: "is_odd",
            mF.max_value: "max_value",
            mF.min_value: "min_value",
            mF.nth_root: "nth_root",
            mF.geometric_mean: "geometric_mean",
            mF.is_power_of_two: "is_power_of_two",
            mF.binary_to_decimal: "binary_to_decimal",
            mF.decimal_to_binary: "decimal_to_binary",
            mF.is_palindrome: "is_palindrome",
            mF.sum_of_digits: "sum_of_digits",
            mF.hypotenuse: "hypotenuse",
            mF.circle_area: "circle_area",
            mF.permutation: "permutation",
            mF.combination: "combination",
            mF.geometric_series_sum: "geometric_series_sum",
            mF.sigmoid: "sigmoid",
            mF.cosine_similarity: "cosine_similarity",
            mF.euler_totient: "euler_totient",
            mF.l1_norm: "l1_norm",
            mF.l2_norm: "l2_norm",
            mF.average: "average",
            mF.sum: "sum",
            mF.length: "length",
            mF.a_plus_b_whole_square: "a_plus_b_whole_square",
            mF.a_squared_plus_2ab_plus_b_squared: "a_squared_plus_2ab_plus_b_squared",
            mF.a_minus_b_whole_squared_plus_4ab: "a_minus_b_whole_squared_plus_4ab",
            mF.a_minus_b_whole_squared: "a_minus_b_whole_squared",
            mF.a_squared_minus_2ab_plus_b_squared: "a_squared_minus_2ab_plus_b_squared",
            mF.a_plus_b_whole_squared_minus_4ab: "a_plus_b_whole_squared_minus_4ab",
            mF.a_squared_plus_b_squared: "a_squared_plus_b_squared",
            mF.negative_2ab: "negative_2ab",
            mF.positive_2ab: "positive_2ab",
            mF.x_plus_a_times_x_plus_b: "x_plus_a_times_x_plus_b",
            mF.x_squared_plus_a_plus_b_times_x_plus_ab: "x_squared_plus_a_plus_b_times_x_plus_ab",
            mF.a_cubed_plus_b_cubed: "a_cubed_plus_b_cubed",
            mF.a_plus_b_whole_cubed_minus_3ab_times_a_plus_b: "a_plus_b_whole_cubed_minus_3ab_times_a_plus_b",
            mF.a_plus_b_times_a_squared_minus_ab_plus_b_squared: "a_plus_b_times_a_squared_minus_ab_plus_b_squared",
            mF.a_cubed_minus_b_cubed: "a_cubed_minus_b_cubed",
            mF.a_minus_b_whole_cubed_plus_3ab_times_a_minus_b: "a_minus_b_whole_cubed_plus_3ab_times_a_minus_b",
            mF.a_minus_b_times_a_squared_plus_ab_plus_b_squared: "a_minus_b_times_a_squared_plus_ab_plus_b_squared",
            mF.invert_number: "invert_number",
            mF.float_to_int: "float_to_int",
            mF.int_to_float: "int_to_float",
            mF.check_same_string: "check_same_string",
            mF.reverse_string: "reverse_string",
            mF.get_pi: "get_pi",
            mF.get_e: "get_e",
            mF.calculate_dot_product: "calculate_dot_product",
        }

    def _set_positive_negative_function_map(self):
        for key, value in self.name_to_reference.items():
            all_values = list(self.name_to_reference.values())
            first_list = [value] + [
                self.name_to_reference[item]
                for sublist in self.vector_arithmetics
                if key in sublist
                for item in sublist
                if item != key
            ]
            second_list = [v for v in all_values if v not in first_list]
            self.positive_negative_function_map[key] = (first_list, second_list)

    def _set_vector_arithmetics(self):
        self.vector_arithmetics = [
            ["mean", "average"],
            [
                "a_plus_b_whole_square",
                "a_squared_plus_2ab_plus_b_squared",
                "a_minus_b_whole_squared_plus_4ab",
            ],
            [
                "a_minus_b_whole_squared",
                "a_squared_minus_2ab_plus_b_squared",
                "a_plus_b_whole_squared_minus_4ab",
            ],
            [
                "a_cubed_minus_b_cubed",
                "a_minus_b_whole_cubed_plus_3ab_times_a_minus_b",
                "a_minus_b_times_a_squared_plus_ab_plus_b_squared",
            ],
            [
                "a_cubed_plus_b_cubed",
                "a_plus_b_whole_cubed_minus_3ab_times_a_plus_b",
                "a_plus_b_times_a_squared_minus_ab_plus_b_squared",
            ],
            ["x_plus_a_times_x_plus_b", "x_squared_plus_a_plus_b_times_x_plus_ab"],
        ]
        # "a_squared_plus_b_squared" == "a_plus_b_whole_square" + "negative_2ab" == "a_minus_b_whole_squared" + "positive_2ab"

    def get_positive_negative_function_map(self):
        return self.positive_negative_function_map

    def get_name_to_reference(self):
        return self.name_to_reference

    def get_reference_to_name(self):
        return self.reference_to_name

    @staticmethod
    def get_function_as_string(function_name: types):
        return inspect.getsource(function_name)

    @staticmethod
    def get_doc_string_of_function(function_name: types):
        return function_name.__doc__

    @staticmethod
    def get_function_as_string_without_doc_string(function_name: types):
        # Get the source code of the function
        source_lines = inspect.getsource(function_name)
        source_lines = re.sub(r'"{3}([\s\S]*?"{3})', "", source_lines)
        return source_lines

    @staticmethod
    def get_function_return_type(function_name: types):
        return get_type_hints(function_name)["return"]

    @staticmethod
    def get_function_param_types(function_name: types) -> dict:
        parameter_types = get_type_hints(function_name, include_extras=True)
        return {k: v for k, v in parameter_types.items() if k != "return"}


if __name__ == "__main__":
    print(FunctionManager.get_function_as_string_without_doc_string(mF.average))
    print(FunctionManager.get_function_return_type(mF.average))
    print(FunctionManager.get_function_param_types(mF.average))
