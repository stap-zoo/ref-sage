from rescue.hash import Rescue
from rescue_prime.params import RescuePrimeParams
from modes import pad_one


class RescuePrime(Rescue):
    def __init__(self, params: RescuePrimeParams):
        super().__init__(params)

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        """Similar to Rescue.permutation, but no initial round constant addition, and SBox/Sbox_inv order switched."""
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")
        
        for r in range(self.R):
            state = self.SBox(state)
            state = self.AffineLayer(state, 2 * r)
            state = self.SBox_inv(state)
            state = self.AffineLayer(state, 2 * r + 1)

        return state

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for r in reversed(range(self.R)):
            state = self.AffineLayer_inv(state, 2 * r + 1)
            state = self.SBox(state)
            state = self.AffineLayer_inv(state, 2 * r)
            state = self.SBox_inv(state)

        return state