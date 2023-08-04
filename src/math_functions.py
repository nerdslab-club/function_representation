import math

import numpy as np


class MathFunctions:
    def __init__(self):
        pass

    @staticmethod
    def addition(x: int, y: int) -> int:
        """This function adds two integers and returns the result as an integer.

        Params:
            x (int): First integer of the addition operation.
            y (int): Second integer of the addition operation.

        Returns:
            int: The result of the addition operation of the given params.
        """
        return x + y

    @staticmethod
    def subtraction(x: int, y: int) -> int:
        """This function subtracts the second integer from the first integer and returns the result as an integer.

        Params:
            x (int): The integer from which the other integer will be subtracted.
            y (int): The integer to be subtracted from the first integer.

        Returns:
            int: The result of the subtraction operation of the given params.
        """
        return x - y

    @staticmethod
    def multiplication(x: float, y: float) -> float:
        """This function multiplies two float and returns the result as a float.

        Params:
            x (float): First float of the multiplication operation.
            y (float): Second float of the multiplication operation.

        Returns:
            float: The result of the multiplication operation of the given params.
        """
        return x * y

    @staticmethod
    def division(x: float, y: float) -> float:
        """This function divides the first integer by the second integer and returns the result as a float.

        Params:
            x (float): The dividend.
            y (float): The divisor.

        Returns:
            float: The result of the division operation of the given params.
        """
        return x / y

    @staticmethod
    def exponentiation(x: float, y: float) -> float:
        """This function raises the first float to the power of the second float and returns the result as a float.

        Params:
            x (float): The base.
            y (float): The exponent.

        Returns:
            float: The result of the exponentiation operation of the given params.
        """
        return x ** y

    @staticmethod
    def square_root(x: float) -> float:
        """This function calculates the square root of the given number and returns the result as a float.

        Params:
            x (float): The number for which the square root will be calculated.

        Returns:
            float: The square root of the given number.
        """
        return math.sqrt(x)

    @staticmethod
    def floor_division(x: int, y: int) -> int:
        """This function performs floor division on the first integer by the second integer and returns the result as an integer.

        Params:
            x (int): The dividend.
            y (int): The divisor.

        Returns:
            int: The result of the floor division operation of the given params.
        """
        return x // y

    @staticmethod
    def modulus(x: int, y: int) -> int:
        """This function calculates the modulus of the first integer with respect to the second integer and returns the result as an integer.

        Params:
            x (int): The dividend.
            y (int): The divisor.

        Returns:
            int: The modulus of the first integer with respect to the second integer.
        """
        return x % y

    @staticmethod
    def logarithm(x: float, base: float) -> float:
        """This function calculates the logarithm of the given number with respect to the given base and returns the result as a float.

        Params:
            x (float): The number for which the logarithm will be calculated.
            base (float): The base of the logarithm.

        Returns:
            float: The logarithm of the given number with respect to the given base.
        """
        return math.log(x, base)

    # kothin
    @staticmethod
    def sine(x: float) -> float:
        """This function calculates the sine of the given angle in radians and returns the result as a float.

        Params:
            x (float): The angle in radians.

        Returns:
            float: The sine of the given angle.
        """
        return math.sin(x)

    # kothin
    @staticmethod
    def cosine(x: float) -> float:
        """This function calculates the cosine of the given angle in radians and returns the result as a float.

        Params:
            x (float): The angle in radians.

        Returns:
            float: The cosine of the given angle.
        """
        return math.cos(x)

    # kothin
    @staticmethod
    def tangent(x: float) -> float:
        """This function calculates the tangent of the given angle in radians and returns the result as a float.

        Params:
            x (float): The angle in radians.

        Returns:
            float: The tangent of the given angle.
        """
        return math.tan(x)

    # kothin
    @staticmethod
    def arcsine(x: float) -> float:
        """This function calculates the arcsine of the given value and returns the result as a float.

        Params:
            x (float): The value for which the arcsine will be calculated.

        Returns:
            float: The arcsine of the given value.
        """
        return math.asin(x)

    # kothin
    @staticmethod
    def arccosine(x: float) -> float:
        """This function calculates the arccosine of the given value and returns the result as a float.

        Params:
            x (float): The value for which the arccosine will be calculated.

        Returns:
            float: The arccosine of the given value.
        """
        return math.acos(x)

    # kothin
    @staticmethod
    def arctangent(x: float) -> float:
        """This function calculates the arctangent of the given value and returns the result as a float.

        Params:
            x (float): The value for which the arctangent will be calculated.

        Returns:
            float: The arctangent of the given value.
        """
        return math.atan(x)

    # kothin
    @staticmethod
    def hyperbolic_sine(x: float) -> float:
        """This function calculates the hyperbolic sine of the given value and returns the result as a float.

        Params:
            x (float): The value for which the hyperbolic sine will be calculated.

        Returns:
            float: The hyperbolic sine of the given value.
        """
        return math.sinh(x)

    # kothin
    @staticmethod
    def hyperbolic_cosine(x: float) -> float:
        """This function calculates the hyperbolic cosine of the given value and returns the result as a float.

        Params:
            x (float): The value for which the hyperbolic cosine will be calculated.

        Returns:
            float: The hyperbolic cosine of the given value.
        """
        return math.cosh(x)

    # kothin
    @staticmethod
    def hyperbolic_tangent(x: float) -> float:
        """This function calculates the hyperbolic tangent of the given value and returns the result as a float.

        Params:
            x (float): The value for which the hyperbolic tangent will be calculated.

        Returns:
            float: The hyperbolic tangent of the given value.
        """
        return math.tanh(x)

    @staticmethod
    def logarithm_base_10(x: float) -> float:
        """This function calculates the logarithm base 10 of the given number and returns the result as a float.

        Params:
            x (float): The number for which the logarithm base 10 will be calculated.

        Returns:
            float: The logarithm base 10 of the given number.
        """
        return math.log10(x)

    @staticmethod
    def logarithm_base_2(x: float) -> float:
        """This function calculates the logarithm base 2 of the given number and returns the result as a float.

        Params:
            x (float): The number for which the logarithm base 2 will be calculated.

        Returns:
            float: The logarithm base 2 of the given number.
        """
        return math.log2(x)

    # kothin
    @staticmethod
    def degrees_to_radians(x: float) -> float:
        """This function converts the given angle from degrees to radians and returns the result as a float.

        Params:
            x (float): The angle in degrees.

        Returns:
            float: The angle converted to radians.
        """
        return math.radians(x)

    # kothin
    @staticmethod
    def radians_to_degrees(x: float) -> float:
        """This function converts the given angle from radians to degrees and returns the result as a float.

        Params:
            x (float): The angle in radians.

        Returns:
            float: The angle converted to degrees.
        """
        return math.degrees(x)

    @staticmethod
    def gcd(x: int, y: int) -> int:
        """This function calculates the greatest common divisor (GCD) of the two given integers and returns the result as an integer.

        Params:
            x (int): The first integer.
            y (int): The second integer.

        Returns:
            int: The greatest common divisor (GCD) of the two given integers.
        """
        return math.gcd(x, y)

    @staticmethod
    def lcm(x: int, y: int) -> int:
        """This function calculates the least common multiple (LCM) of the two given integers and returns the result as an integer.

        Params:
            x (int): The first integer.
            y (int): The second integer.

        Returns:
            int: The least common multiple (LCM) of the two given integers.
        """
        return abs(x * y) // math.gcd(x, y)

    @staticmethod
    def isqrt(x: int) -> int:
        """This function calculates the integer square root of the given integer and returns the result as an integer.

        Params:
            x (int): The integer for which the integer square root will be calculated.

        Returns:
            int: The integer square root of the given integer.
        """
        return math.isqrt(x)

    @staticmethod
    def pow_mod(x: int, y: int, mod: int) -> int:
        """This function calculates the modular exponentiation of the first integer raised to the power of the second integer with respect to the third integer and returns the result as an integer.

        Params:
            x (int): The base.
            y (int): The exponent.
            mod (int): The modulus.

        Returns:
            int: The result of the modular exponentiation operation of the given params.
        """
        return pow(x, y, mod)

    @staticmethod
    def ceil(x: float) -> int:
        """This function rounds up the given number to the nearest integer greater than or equal to the given number and returns the result as an integer.

        Params:
            x (float): The number to be rounded up.

        Returns:
            int: The rounded up value of the given number.
        """
        return math.ceil(x)

    @staticmethod
    def floor(x: float) -> int:
        """This function rounds down the given number to the nearest integer less than or equal to the given number and returns the result as an integer.

        Params:
            x (float): The number to be rounded down.

        Returns:
            int: The rounded down value of the given number.
        """
        return math.floor(x)

    @staticmethod
    def round(x: float) -> int:
        """This function rounds the given number to the nearest integer and returns the result as an integer.

        Params:
            x (float): The number to be rounded.

        Returns:
            int: The rounded value of the given number.
        """
        return round(x)

    @staticmethod
    def absolute_difference(x: float, y: float) -> float:
        """This function calculates the absolute difference between the two given numbers and returns the result as a float.

        Params:
            x (float): The first number.
            y (float): The second number.

        Returns:
            float: The absolute difference between the two given numbers.
        """
        return abs(x - y)

    @staticmethod
    def greatest_value(x: float, y: float) -> float:
        """This function returns the greater value among the two given numbers.

        Params:
            x (float): The first number.
            y (float): The second number.

        Returns:
            float: The greater value among the two given numbers.
        """
        return max(x, y)

    @staticmethod
    def smallest_value(x: float, y: float) -> float:
        """This function returns the smaller value among the two given numbers.

        Params:
            x (float): The first number.
            y (float): The second number.

        Returns:
            float: The smaller value among the two given numbers.
        """
        return min(x, y)

    @staticmethod
    def product(numbers: list) -> float:
        """This function calculates the product of the given list of numbers and returns the result as a float.

        Params:
            numbers (list): A list of numbers.

        Returns:
            float: The product of the given list of numbers.
        """
        result = 1
        for num in numbers:
            result *= num
        return result

    @staticmethod
    def factorial(x: int) -> int:
        """This function calculates the factorial of the given integer and returns the result as an integer.

        Params:
            x (int): The integer for which the factorial will be calculated.

        Returns:
            int: The factorial of the given integer.
        """
        return math.factorial(x)

    @staticmethod
    def is_prime(x: int) -> bool:
        """This function checks if the given number is prime.

        Params:
            x (int): The number to be checked.

        Returns:
            bool: True if the number is prime, False otherwise.
        """
        if x < 2:
            return False
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True

    @staticmethod
    def prime_factors(x: int) -> list:
        """This function calculates the prime factors of the given number and returns them as a list.

        Params:
            x (int): The number for which the prime factors will be calculated.

        Returns:
            list: The list of prime factors of the given number.
        """
        factors = []
        while x % 2 == 0:
            factors.append(2)
            x //= 2
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            while x % i == 0:
                factors.append(i)
                x //= i
        if x > 2:
            factors.append(x)
        return factors

    @staticmethod
    def is_perfect_square(x: int) -> bool:
        """This function checks if the given number is a perfect square.

        Params:
            x (int): The number to be checked.

        Returns:
            bool: True if the number is a perfect square, False otherwise.
        """
        sqrt = math.isqrt(x)
        return sqrt * sqrt == x

    @staticmethod
    def is_perfect_cube(x: int) -> bool:
        """This function checks if the given number is a perfect cube.

        Params:
            x (int): The number to be checked.

        Returns:
            bool: True if the number is a perfect cube, False otherwise.
        """
        cbrt = round(x ** (1 / 3))
        return cbrt * cbrt * cbrt == x

    @staticmethod
    def mean(numbers: list) -> float:
        """This function calculates the mean (average) of the given list of numbers and returns the result as a float.

        Params:
            numbers (list): A list of numbers.

        Returns:
            float: The mean (average) of the given list of numbers.
        """
        return sum(numbers) / len(numbers)

    @staticmethod
    def median(numbers: list) -> float:
        """This function calculates the median of the given list of numbers and returns the result as a float.

        Params:
            numbers (list): A list of numbers.

        Returns:
            float: The median of the given list of numbers.
        """
        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)
        if n % 2 == 0:
            return (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
        else:
            return sorted_numbers[n // 2]

    @staticmethod
    def relu(x: float) -> float:
        """Rectified Linear Unit (ReLU) activation function.

        Params:
            x (float): Input value.

        Returns:
            The output of the ReLU activation function.
        """
        return max(0.0, x)

    @staticmethod
    def ascending_sort(lst: list[int]) -> list[int]:
        """Sorts a list of int in ascending order

        Params:
            lst (list[int]): List of values to be sorted.

        Returns:
            list[int]: Sorted list in ascending order.
        """
        return sorted(lst)

    @staticmethod
    def descending_sort(lst: list[int]) -> list[int]:
        """Sorts a list of values in descending order.

        Params:
            lst (list[int]): List of values to be sorted.

        Returns:
            list[int]: Sorted list in descending order.
        """
        return sorted(lst, reverse=True)

    @staticmethod
    def square_int(x: int) -> int:
        """Calculates the square of an int.

        Params:
            x (int): Number to be squared.

        Returns:
            int: The square of the input number.
        """
        return x ** 2

    @staticmethod
    def square(x: float) -> float:
        """Calculates the square of a number.

        Params:
            x (float): Number to be squared.

        Returns:
            float: The square of the input number.
        """
        return x ** 2

    @staticmethod
    def absolute(x: float) -> float:
        """Calculates the absolute value of a number.

        Params:
            x: Number for which absolute value is calculated.

        Returns:
            The absolute value of the input number.
        """
        return abs(x)

    @staticmethod
    def power_of_ten(x: float) -> float:
        """Calculates the power of ten for a given exponent.

        Params:
            x: Exponent value.

        Returns:
            The result of 10 raised to the power of the input exponent.
        """
        return 10 ** x

    @staticmethod
    def cube(x: float) -> float:
        """Calculates the cube of a number.

        Params:
            x: Number to be cubed.

        Returns:
            The cube of the input number.
        """
        return x ** 3

    @staticmethod
    def cube_root(x: float) -> float:
        """Calculates the cube root of a number.

        Params:
            x: Number for which cube root is calculated.

        Returns:
            The cube root of the input number.
        """
        return x ** (1 / 3)

    @staticmethod
    def is_even(x: int) -> bool:
        """Checks if a number is even.

        Params:
            x: Number to be checked.

        Returns:
            bool: True if the number is even, False otherwise.
        """
        return x % 2 == 0

    @staticmethod
    def is_odd(x: int) -> bool:
        """Checks if a number is odd.

        Params:
            x: Number to be checked.

        Returns:
            bool: True if the number is odd, False otherwise.
        """
        return x % 2 != 0

    @staticmethod
    def max_value(lst: list[int]) -> int:
        """Finds the maximum value in a list of integer

        Params:
            lst (list[int]): List of values.

        Returns:
            The maximum value in the list as int
        """
        return max(lst)

    @staticmethod
    def min_value(lst: list[int]) -> int:
        """Finds the minimum value in a list of integer

        Params:
            lst (list[int]): List of values.

        Returns:
            The minimum value in the list as int.
        """
        return min(lst)

    @staticmethod
    def nth_root(x: float, n: int) -> float:
        """Calculates the nth root of a number.

        Params:
            x (float): Number for which nth root is calculated.
            n (int): Root degree.

        Returns:
            The nth root of the input number.
        """
        return x ** (1 / n)

    @staticmethod
    def geometric_mean(lst: list[float]) -> float:
        """Calculates the geometric mean of a list of numbers.

        Params:
            lst (list[float]): List of numbers.

        Returns:
            float: The geometric mean of the input list.
        """
        product = 1
        for num in lst:
            product *= num
        return product ** (1 / len(lst))

    @staticmethod
    def is_power_of_two(x: int) -> bool:
        """Checks if a number is a power of two.

        Params:
            x (int): Number to be checked.

        Returns:
            bool: True if the number is a power of two, False otherwise.
        """
        return x > 0 and (x & (x - 1)) == 0

    @staticmethod
    def binary_to_decimal(binary):
        """Converts a binary number to decimal.

        Params:
            binary (str): Binary number as a string.

        Returns:
            int: Decimal representation of the input binary number.
        """
        return int(binary, 2)

    @staticmethod
    def decimal_to_binary(decimal):
        """Converts a decimal number to binary.

        Params:
            decimal: Decimal number.

        Returns:
            str: Binary representation of the input decimal number.
        """
        return bin(decimal)[2:]

    @staticmethod
    def is_palindrome(x) -> bool:
        """Checks if a number is a palindrome.

        Params:
            x (int): Number to be checked.

        Returns:
            bool: True if the number is a palindrome, False otherwise.
        """
        return str(x) == str(x)[::-1]

    @staticmethod
    def sum_of_digits(x: int) -> int:
        """Calculates the sum of digits in an integer.

        Params:
            x (int): Input integer.

        Returns:
            int: Sum of the digits in the input integer.
        """
        return sum(int(digit) for digit in str(abs(x)))

    @staticmethod
    def hypotenuse(a: float, b: float) -> float:
        """Calculates the length of the hypotenuse of a right-angled triangle.

        Params:
            a (float): Length of one side of the triangle.
            b (float): Length of the other side of the triangle.

        Returns:
            float: Length of the hypotenuse of the triangle.
        """
        return (a ** 2 + b ** 2) ** 0.5

    # TODO can be done
    @staticmethod
    def circle_area(radius: float) -> float:
        """Calculates the area of a circle.

        Params:
            radius (float): Radius of the circle.

        Returns:
            float: Area of the circle.
        """
        return 3.14159 * radius ** 2

    @staticmethod
    def permutation(n: int, r: int) -> int:
        """Calculates the number of permutations of choosing r items from n items.

        Params:
            n (int): Total number of items.
            r (int): Number of items to be chosen.

        Returns:
            int: Number of permutations.
        """
        if n < r:
            return 0
        numerator = 1
        for i in range(n, n - r, -1):
            numerator *= i
        return numerator

    @staticmethod
    def combination(n: int, r: int) -> int:
        """Calculates the number of combinations of choosing r items from n items.

        Params:
            n (int): Total number of items.
            r (int): Number of items to be chosen.

        Returns:
            int: Number of combinations.
        """
        if n < r:
            return 0
        numerator = 1
        denominator = 1
        for i in range(n, n - r, -1):
            numerator *= i
        for i in range(1, r + 1):
            denominator *= i
        return numerator // denominator

    @staticmethod
    def invert_number(number: float) -> float:
        """This function calculates the reciprocal (inverse) of a given number.
        The reciprocal of a number 'n' is 1/n.

        Params:
            number (float): The number for which the reciprocal will be calculated.

        Returns:
            float: The reciprocal of the given number.
        """
        if number == 0:
            raise ValueError("Cannot invert the number 0 (division by zero).")

        return 1 / number

    @staticmethod
    def float_to_int(value: float) -> int:
        """This function converts a floating-point number to an integer by truncating the decimal part.

        Params:
            value (float): The floating-point number to be converted to an integer.

        Returns:
            int: The integer representation of the given floating-point number.
        """
        return int(value)

    @staticmethod
    def int_to_float(value: int) -> float:
        """This function converts an integer to a floating-point number.

        Params:
            value (int): The integer to be converted to a floating-point number.

        Returns:
            float: The floating-point representation of the given integer.
        """
        return float(value)

    # TODO can be done
    @staticmethod
    def geometric_series_sum(a: float, r: float, n: int) -> float:
        """Calculates the sum of a geometric series.

        Params:
            a (float): First term of the series.
            r (float): Common ratio of the series.
            n (int): Number of terms in the series.

        Returns:
            float: Sum of the geometric series.
        """
        if r == 1:
            return a * n
        else:
            return a * (1 - r ** n) / (1 - r)

    # TODO can be done
    @staticmethod
    def sigmoid(x: float) -> float:
        """Calculates the sigmoid function value.

        Params:
            x (float): Input value.

        Returns:
            float: Sigmoid function value.
        """
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def cosine_similarity(vector1: list, vector2: list) -> float:
        """Calculates the cosine similarity between two vectors.

        Params:
            vector1 (list): First vector.
            vector2 (list): Second vector.

        Returns:
            float: Cosine similarity between the two vectors.
        """
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        return dot_product / (norm_vector1 * norm_vector2)

    @staticmethod
    def euler_totient(n: int) -> int:
        """Calculates the Euler's totient function value for a given number.

        Params:
            n (int): Input number.

        Returns:
            int: Euler's totient function value.
        """
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    @staticmethod
    def l1_norm(vector: list) -> float:
        """Calculates the L1 norm (Manhattan norm) of a vector.

        Params:
            vector (list): Input vector.

        Returns:
            float: L1 norm of the vector.
        """
        return np.linalg.norm(vector, ord=1)

    @staticmethod
    def l2_norm(vector: list) -> float:
        """Calculates the L2 norm (Euclidean norm) of a vector.

        Params:
            vector (list): Input vector.

        Returns:
            float: L2 norm of the vector.
        """
        return np.linalg.norm(vector)

    @staticmethod
    def average(numbers: list) -> float:
        """This function calculates the average of the given list of numbers and returns the result as a float.

        Params:
            numbers (list): A list of numbers.

        Returns:
            float: The average of the given list of numbers.
        """
        return sum(numbers) / len(numbers)

    @staticmethod
    def sum(numbers: list) -> float:
        """This function calculates the sum of the given list of numbers and returns the result as a float.

        Params:
            numbers (list): A list of numbers.

        Returns:
            float: The sum of the given list of numbers.
        """
        return sum(numbers)

    @staticmethod
    def length(numbers: list) -> int:
        """This function calculates the length of the given list of numbers and returns the result as a int.

        Params:
            numbers (list): A list of items.

        Returns:
            float: The length of the given list of items.
        """
        return len(numbers)

    @staticmethod
    def a_plus_b_whole_square(a: int, b: int) -> int:
        """This function calculates a plus b whole square formula and returns the result as an int.

        Params:
            a (int): First value for which (a+b)^2 is going to be calculated
            b (int): Second value for which (a+b)^2 is going to be calculated

        Returns:
            int: The result of (a+b)^2 after calculating.
        """
        return (a + b) ** 2

    @staticmethod
    def a_squared_plus_2ab_plus_b_squared(a: int, b: int) -> int:
        """
        This function calculates the expression a^2 + 2ab + b^2 and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression a^2 + 2ab + b^2.
        """
        return a ** 2 + 2 * a * b + b ** 2

    @staticmethod
    def a_minus_b_whole_squared_plus_4ab(a: int, b: int) -> int:
        """
        This function calculates the expression (a - b)^2 + 4ab and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression (a - b)^2 + 4ab.
        """
        return (a - b) ** 2 + 4 * a * b

    @staticmethod
    def a_minus_b_whole_squared(a: int, b: int) -> int:
        """
        This function calculates the expression (a - b)^2 and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression (a - b)^2.
        """
        return (a - b) ** 2

    @staticmethod
    def a_squared_minus_2ab_plus_b_squared(a: int, b: int) -> int:
        """
        This function calculates the expression a^2 - 2ab + b^2 and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression a^2 - 2ab + b^2.
        """
        return a ** 2 - 2 * a * b + b ** 2

    @staticmethod
    def a_plus_b_whole_squared_minus_4ab(a: int, b: int) -> int:
        """
        This function calculates the expression (a + b)^2 - 4ab and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression (a + b)^2 - 4ab.
        """
        return (a + b) ** 2 - 4 * a * b

    @staticmethod
    def a_squared_plus_b_squared(a: int, b: int) -> int:
        """
        This function calculates the expression a^2 + b^2 and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression a^2 + b^2.
        """
        return a ** 2 + b ** 2

    @staticmethod
    def negative_2ab(a: int, b: int) -> int:
        """
        This function calculates the expression -2ab and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression -2ab.
        """
        return -2 * a * b

    @staticmethod
    def positive_2ab(a: int, b: int) -> int:
        """
        This function calculates the expression 2ab and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression 2ab.
        """
        return 2 * a * b

    @staticmethod
    def x_plus_a_times_x_plus_b(x: int, a: int, b: int) -> int:
        """
        This function calculates the expression (x + a)(x + b) and returns the result as an int.

        Params:
            x (int): The value of x for which the expression is going to be calculated.
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression (x + a)(x + b).
        """
        return (x + a) * (x + b)

    @staticmethod
    def x_squared_plus_a_plus_b_times_x_plus_ab(x: int, a: int, b: int) -> int:
        """
        This function calculates the expression x^2 + (a + b)x + ab and returns the result as an int.

        Params:
            x (int): The value of x for which the expression is going to be calculated.
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression x^2 + (a + b)x + ab.
        """
        return x ** 2 + (a + b) * x + a * b

    @staticmethod
    def a_cubed_plus_b_cubed(a: int, b: int) -> int:
        """
        This function calculates the expression a^3 + b^3 and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression a^3 + b^3.
        """
        return a ** 3 + b ** 3

    @staticmethod
    def a_plus_b_whole_cubed_minus_3ab_times_a_plus_b(a: int, b: int) -> int:
        """
        This function calculates the expression (a + b)^3 - 3ab(a + b) and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression (a + b)^3 - 3ab(a + b).
        """
        return (a + b) ** 3 - 3 * a * b * (a + b)

    @staticmethod
    def a_plus_b_times_a_squared_minus_ab_plus_b_squared(a: int, b: int) -> int:
        """
        This function calculates the expression (a + b)(a^2 - ab + b^2) and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression (a + b)(a^2 - ab + b^2).
        """
        return (a + b) * (a ** 2 - a * b + b ** 2)

    @staticmethod
    def a_cubed_minus_b_cubed(a: int, b: int) -> int:
        """
        This function calculates the expression a^3 - b^3 and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression a^3 - b^3.
        """
        return a ** 3 - b ** 3

    @staticmethod
    def a_minus_b_whole_cubed_minus_3ab_times_a_minus_b(a: int, b: int) -> int:
        """
        This function calculates the expression (a - b)^3 - 3ab(a - b) and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression (a - b)^3 - 3ab(a - b).
        """
        return (a - b) ** 3 - 3 * a * b * (a - b)

    @staticmethod
    def a_minus_b_times_a_squared_plus_ab_plus_b_squared(a: int, b: int) -> int:
        """
        This function calculates the expression (a - b)(a^2 + ab + b^2) and returns the result as an int.

        Params:
            a (int): First value for which the expression is going to be calculated.
            b (int): Second value for which the expression is going to be calculated.

        Returns:
            int: The result of the expression (a - b)(a^2 + ab + b^2).
        """
        return (a - b) * (a ** 2 + a * b + b ** 2)
