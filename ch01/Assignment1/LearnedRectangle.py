"""
ECE 457B, Fall 2025
University of Waterloo
Mahesh Tripunitara, tripunit@uwaterloo.ca
"""

from Assignment1_01.generate_example import generate_example

"""
You are not allowed to import anything else.

generate_example() is a generator. If you invoke
next(generate_example()), it will give you an example.
An example is a pair (point, b) where b is True if point is a positive
example, and False if it is a negative example. The component point is
a list of length >= 1. You will discover the number of dimensions of
the target rectangle only after your first invocation to this
generator.

E.g., a return value from next(generate_example()) may be
([1,12,3], True). This means that the target rectangle is in
3-dimensions, and the point <1,12,3> is within that rectangle.
"""

class LearnedRectangle:
    def __init__(self):
        self.min = None
        self.max = None
        self.dimensions = None

    def learn(self, m):
        # Your code here to learn the target rectangle. You should
        # get the m examples by invoking next(generate_example()).
        # You can assume that m is an integer >= 1.
        positive_points = []

        for i in range(m):
            point, valid_point = next(generate_example())

            if self.dimensions == None:
                self.dimensions = len(point)
                self.max = [float('-inf')]*self.dimensions
                self.min = [float('inf')]*self.dimensions

            if valid_point:
                positive_points.append(point)

        if positive_points:
            for point in positive_points:
                for d_iter in range(self.dimensions):
                    self.min[d_iter] = min(self.min[d_iter],point[d_iter])
                    self.max[d_iter] = max(self.max[d_iter],point[d_iter])

        else:
            self.min = [0]*self.dimensions
            self.max = [0]*self.dimensions



    def checkgoodness(self, n, k, epsilon):
        # Your code here for the following, whose intent is to check
        # the goodness of your learned rectangle.

        # Initialize a counter, and perform the following n times.

        # For k examples, check whether the proportion of those k
        # that are misclassified by your learned rectangle is > epsilon.
        # If yes, increase your counter by 1.
        # Return the value of your counter.
        #
        # E.g., suppose n = 2, k = 5 and epsilon = 0.2. This means
        # you will consider 2 sets of 5 examples each. Suppose for the
        # first # set of 5 examples, your learned rectangle has
        # misclassified 1 of those 5 examples. As # 1 <= 5 x 0.2, you
        # do not increase your counter. Suppose for the second set of
        # 5 examples, your learned rectangle misclassied 3 out of 5.
        # As 3 > 5 x 0.2, you will increase your counter by 1. And you
        # will return 1, i.e., the value of the counter, as your result.
        count = 0
        for _ in range(n):
            errors = 0
            for _ in range(k):
                pt, actual = next(generate_example())
                inside = True
                for idx in range(len(pt)):
                    if pt[idx] < self.min[idx] or pt[idx] > self.max[idx]:
                        inside = False
                        break
                guess = inside
                if guess != actual:
                    errors += 1
            if errors > k * epsilon:
                count += 1
        return count


if __name__ == "__main__":
    m = 50
    n = 100
    k = 20
    epsilon = 0.2

    learner = LearnedRectangle()
    learner.learn(m)
    bad_trials = learner.checkgoodness(n, k, epsilon)
    print(f"Number of bad trials: {bad_trials} out of {n}")


#  E.g., “Credit: ChatGPT.”