import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer


class FractalMomentAdam(Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        beta_1_slow=0.99,
        beta_2_slow=0.9999,
        gamma=0.3,
        epsilon=1e-7,
        name="FractalMomentAdam",
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_1_slow = beta_1_slow
        self.beta_2_slow = beta_2_slow
        self.gamma = gamma
        self.epsilon = epsilon

        self._m_fast = {}
        self._v_fast = {}
        self._m_slow = {}
        self._v_slow = {}

    def build(self, var_list):
        for var in var_list:
            name = var.name
            self._m_fast[name] = self.add_variable_from_reference(var, "m_fast", initializer="zeros")
            self._v_fast[name] = self.add_variable_from_reference(var, "v_fast", initializer="zeros")
            self._m_slow[name] = self.add_variable_from_reference(var, "m_slow", initializer="zeros")
            self._v_slow[name] = self.add_variable_from_reference(var, "v_slow", initializer="zeros")

        self._step = self.add_variable((), dtype=tf.int64, initializer="zeros", name="iter")
        super().build(var_list)

    def update_step(self, grad, var, learning_rate=None):
        name = var.name

        if name not in self._m_fast:
            self._m_fast[name] = self.add_variable_from_reference(var, "m_fast", initializer="zeros")
            self._v_fast[name] = self.add_variable_from_reference(var, "v_fast", initializer="zeros")
            self._m_slow[name] = self.add_variable_from_reference(var, "m_slow", initializer="zeros")
            self._v_slow[name] = self.add_variable_from_reference(var, "v_slow", initializer="zeros")

        m_f, v_f = self._m_fast[name], self._v_fast[name]
        m_s, v_s = self._m_slow[name], self._v_slow[name]

        lr = tf.cast(learning_rate if learning_rate is not None else self.learning_rate, var.dtype)
        eps = tf.cast(self.epsilon, var.dtype)
        beta_1 = tf.cast(self.beta_1, var.dtype)
        beta_2 = tf.cast(self.beta_2, var.dtype)
        beta_1_slow = tf.cast(self.beta_1_slow, var.dtype)
        beta_2_slow = tf.cast(self.beta_2_slow, var.dtype)
        gamma = tf.cast(self.gamma, var.dtype)

        # Step
        self._step.assign_add(1)
        t = tf.cast(self._step, var.dtype)

        # Fast moments
        m_fast = beta_1 * m_f + (1 - beta_1) * grad
        v_fast = beta_2 * v_f + (1 - beta_2) * tf.square(grad)

        # Slow moments
        m_slow = beta_1_slow * m_s + (1 - beta_1_slow) * grad
        v_slow = beta_2_slow * v_s + (1 - beta_2_slow) * tf.square(grad)

        # Merge by scale
        m_eff = gamma * m_fast + (1 - gamma) * m_slow
        v_eff = gamma * v_fast + (1 - gamma) * v_slow

        # bias correction
        m_eff_hat = m_eff / (1 - tf.pow(beta_1, t))
        v_eff_hat = v_eff / (1 - tf.pow(beta_2, t))

        var.assign_sub(lr * m_eff_hat / (tf.sqrt(v_eff_hat) + eps))

        # Renew slots
        m_f.assign(m_fast)
        v_f.assign(v_fast)
        m_s.assign(m_slow)
        v_s.assign(v_slow)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "beta_1_slow": self.beta_1_slow,
            "beta_2_slow": self.beta_2_slow,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
        }
