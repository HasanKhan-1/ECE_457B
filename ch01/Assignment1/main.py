from LearnedRectangle import LearnedRectangle

m = 50
n = 100
k = 20
epsilon = 0.1

learner = LearnedRectangle()
learner.learn(m)
bad_trials = learner.checkgoodness(n, k, epsilon)
print(f"Number of bad trials: {bad_trials} out of {n}")
