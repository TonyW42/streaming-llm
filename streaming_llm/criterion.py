from statistics import NormalDist

class bayesian:
    def __init__(self, mu_0, sigma_0):
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0 
        self.n = 0
        self.x_sum = 0
        self.x_sq_sum = 0
    
    def _compute_posterior_mean(self):
        sample_sigma = self._compute_sample_sigma(self)
        return (
            self.mu_0 / (self.sigma_0 ** 2) + \
            self.x_sum / (sample_sigma ** 2) 
            ) / \
            (
                1 / (self.sigma_0 ** 2) + \
                self.n / (sample_sigma ** 2) 
            )
    def _compute_posterior_sigma(self):
        sample_sigma = self._compute_sample_sigma(self)
        return (
            1 / (self.sigma_0 ** 2) + \
            self.n / (sample_sigma ** 2) 
        ) ** -1


    def _compute_sample_sigma(self):
        return ((self.x_sq_sum - (self.x_sq_sum) ** 2 / self.n) / self.n) ** 0.5
    
    def _compute_posterior_params(self):
        return self._compute_posterior_mean(), self._compute_posterior_sigma()
    
    def bayesian_update(self, x):
        self.n += 1
        self.x_sum += x
        self.x_sq_sum += x ** 2
    
    def posterior_cdf(self, x):
        mu, sigma = self._compute_posterior_params()
        return NormalDist(mu=mu, sigma=sigma).cdf(x)
    





