import pytest

from learning.utils import (set_discount_from_horizon,
                            remove_zero_weighted_rewards)


class TestUtils:
    def test_set_discount_from_horizon(self):
        dt = 0.1
        horizon = 1
        assert set_discount_from_horizon(dt, horizon) == 0.9
        assert set_discount_from_horizon(dt, 0.) == 0
        with pytest.raises(AssertionError):
            set_discount_from_horizon(dt, dt/2.)
        with pytest.raises(AssertionError):
            set_discount_from_horizon(dt, -1)
        with pytest.raises(AssertionError):
            set_discount_from_horizon(-dt, dt/2.)

    def test_remove_zero_weighted_rewards(self):
        reward_weights = {'a': 0, 'b': 1}
        remove_zero_weighted_rewards(reward_weights)
        assert reward_weights == {'b': 1}
