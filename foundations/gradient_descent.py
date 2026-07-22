class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        def func(init):
            return init**2
        def derivative(init):
            return 2*init
        def error(new_value,old_value):
            return abs(func(new_value)-func(old_value))

        old_value = init
        new_value = init

        for i in range(iterations):
            new_value = old_value - learning_rate*derivative(old_value)
            if error(new_value,old_value)<1e-5:
                break
            old_value = new_value

        return round(new_value,5)