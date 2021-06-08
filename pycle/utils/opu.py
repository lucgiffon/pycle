"""temporary module provided by lighton. It should be replaced with the official release at some point."""

from lightonml.encoding.base import preserve_type, BaseTransformer, SeparatedBitPlanEncoder as OldSeparatedBitPlanEncoder
import numexpr as ne
import numpy as np


class SeparatedBitPlanEncoder(BaseTransformer):
    """
    Implements an encoder for floating point input

    Parameters
    ----------
    precision: int, optional
        The number of binary projections that are preformed to reconstruct an unsigned floating point projection.
        if the input contains both positive and negative values, the total number of projections is 2*precision

    Returns
    -------
    X_bits: np.array(dtype = np.unit8)

    """

    def __init__(self, precision=6, **kwargs):
        assert (0 < precision <= 8)
        if "n_bits" in kwargs.keys() or "starting_bit" in kwargs.keys():
            raise RuntimeError("Encoder interface has changed from n_bit to precision")
        self.precision = precision
        self.magnitude = None, None
        self.magnitude_p = None
        self.magnitude_n = None
        self.fitted = False

    def fit(self, X, y=None):
        def get_int_magnitude(X_):
            # separate case for integers to increase precision.
            # ensures X_quantized is just a shift.
            magnitude = X_.max()
            if magnitude < 0:
                return 0
            shift = self.precision - np.ceil(np.log2(X_.max() + 1))
            return (2 ** self.precision - 1) * 0.5 ** shift

        if np.issubdtype(X.dtype, np.signedinteger):
            magnitude_p = get_int_magnitude(+X)
            magnitude_n = get_int_magnitude(-X)
        elif np.issubdtype(X.dtype, np.integer):
            magnitude_p = get_int_magnitude(+X)
            magnitude_n = 0
        else:
            magnitude_p = np.max(+X.max(), 0)
            magnitude_n = np.max(-X.min(), 0)


        dequantization_scale = (1 - 0.5 ** self.precision) * 2
        self.magnitude = magnitude_p / dequantization_scale, magnitude_n / dequantization_scale
        self.magnitude_p = magnitude_p
        self.magnitude_n = magnitude_n
        self.fitted = True

    @preserve_type
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    @preserve_type
    def transform(self, X):
        if not self.fitted:
            raise TypeError("Encoder should be fit before transformation.")
        X = X.astype(np.float)

        # Takes inputs in the range 0-1 ands splits into n (=precision) bitplanes
        def get_bits_unit_positive(X_, precision):
            X_quantized = (X_ * (2 ** precision - 1) + 0.5).astype(np.uint8)
            X_bits = np.unpackbits(X_quantized[:, None], axis=1, bitorder='big')[:, -precision:]
            return X_bits.reshape(-1, *X_bits.shape[2:])

        magnitude_p_safe = self.magnitude_p if self.magnitude_p != 0 else 1
        magnitude_n_safe = self.magnitude_n if self.magnitude_n != 0 else 1

        if self.magnitude_n <= 0:
            return get_bits_unit_positive(np.clip(+X / magnitude_p_safe, 0, 1), self.precision)
        if self.magnitude_p <= 0:
            return get_bits_unit_positive(np.clip(-X / magnitude_n_safe, 0, 1), self.precision)

        Xp_bits = get_bits_unit_positive(np.clip(+X / magnitude_p_safe, 0, 1), self.precision)
        Xn_bits = get_bits_unit_positive(np.clip(-X / magnitude_n_safe, 0, 1), self.precision)

        return np.concatenate((Xp_bits, Xn_bits), 0)

    def get_params(self):
        """
        internal information necessary to undo the transformation,
        must be passed to the SeparatedBitPlanDecoder init.
        """
        return {'precision': self.precision, 'magnitude_p': self.magnitude[0], 'magnitude_n': self.magnitude[1]}


class SeparatedBitPlanDecoder(BaseTransformer):
    def __init__(self, precision, magnitude_p=1, magnitude_n=0, decoding_decay=0.5):
        """Init takes the output of the get_params() method of the SeparatedBitPlanEncoder"""
        self.precision = precision
        self.magnitude_p = magnitude_p
        self.magnitude_n = magnitude_n
        self.decoding_decay = decoding_decay

    @preserve_type
    def transform(self, X):
        n_out, n_features = X.shape
        sides = 2 if (self.magnitude_n > 0 and self.magnitude_p > 0) else 1
        n_dim_0 = int(n_out / (self.precision * sides))

        X = np.reshape(X, (sides, n_dim_0, self.precision, n_features))

        # recombines the bitplanes with the correct weights.
        def decode_unit_positive(X_):
            # compute factors for each bit to weight their significance
            decay_factors = (self.decoding_decay ** np.arange(self.precision)).astype('float32')
            if self.precision < 16:
                d = {'X' + str(i): X_[:, i] for i in range(self.precision)}
                d.update({'decay_factors' + str(i): decay_factors[i] for i in range(self.precision)})
                eval_str = ' + '.join(['X' + str(i) + '*' + 'decay_factors' + str(i) for i in range(self.precision)])
                X_dec = ne.evaluate(eval_str, d)
            else:
                # fallback to slower version if n_bits > 15 because of
                # https://gitlab.lighton.ai/main/lightonml/issues/58
                X_dec = np.einsum('ijk,j->ik', X_, decay_factors).astype('float32')
            return X_dec

        X_transformed = X.astype(np.float)
        Xp_transformed_raw = X_transformed[0]
        Xn_transformed_raw = X_transformed[1 if sides == 2 else 0]

        if self.magnitude_n <= 0:
            return decode_unit_positive(Xp_transformed_raw) * self.magnitude_p
        if self.magnitude_p <= 0:
            return -decode_unit_positive(Xp_transformed_raw) * self.magnitude_n

        Xp_transformed = decode_unit_positive(Xp_transformed_raw) * self.magnitude_p
        Xn_transformed = decode_unit_positive(Xn_transformed_raw) * self.magnitude_n

        return Xp_transformed - Xn_transformed


class QuantizedSeparatedBitPlanEncoder(OldSeparatedBitPlanEncoder):
    def __init__(self, base, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = base
        if self.n_bits <= 8:
            self.dtype_index_quantile = np.uint8
        elif self.n_bits <= 16:
            self.dtype_index_quantile = np.uint16
        elif self.n_bits <= 32:
            self.dtype_index_quantile = np.uint32
        elif self.n_bits <= 64:
            self.dtype_index_quantile = np.uint64
        else:
            raise ValueError(f"`n_bits` attribute can't be greater than 64. It is {self.n_bits}")

    def transform(self, X):
        assert np.max(X) <= 1 and np.min(X) >= 0
        float_index_x = X * (self.base**self.n_bits - 1)
        rounded = np.around(float_index_x)
        x_before_enc = rounded.astype(self.dtype_index_quantile).reshape(-1, X.shape[-1])
        encoded = super().transform(x_before_enc)
        return encoded


class QuantizedMixingBitPlanDecoder(MixingBitPlanDecoder):
    def transform(self, X):
        decoded = super().transform(X)
        x_after_dec = decoded / (self.decoding_decay ** self.n_bits - 1)
        return x_after_dec