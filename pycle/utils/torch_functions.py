"""
This module contains different torch functions to be used to compute the random projection with adapted folded gaussian
distribution.
"""

import torch
from lightonml.encoding.base import SeparatedBitPlanEncoder, SeparatedBitPlanDecoder
from torch.autograd import Function


class MultiSigmaARFrequencyMatrixLinApEncDec(Function):
    """
    This function is able to compute multiple random projection with different scaling factors sigma.

    The base random matrix is a set of directions multiplied by a set of amplitudes (R).
    For each different sigma, this random matrix is scaled by the corresponding sigma.

    It is also possible to use as many set of amplitudes R as there are sigmas.

    All the computation can be wrapped into encoding/decoding functions or the input can be quantified before
    being transformed. These options, however, are here for experimental purpose and I don't see why they could be used
    for production purposes. These encoding/decoding and quantification processes not being differentiable, the backward step
    just ignores them.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, SigFacts: torch.Tensor,
                directions: torch.Tensor, R: torch.Tensor,
                quantif: bool = False, enc_dec: bool = False, encoding_decoding_precision: int = 8) -> torch.Tensor:
        """

        Parameters
        ----------
        ctx
        input:
            tensor to transform.
        SigFacts:
            1D tensor containing the list of sigmas.
        directions:
            2D tensor of the base directions.
        R:
            1D or 2D tensor containing the list of amplitudes for each direction.
        quantif:
            Tells to quantify the input before transformation (experimentation purposes).
        enc_dec:
            Tells to encode the input before transformation and decode after (experimentation purposes).
        encoding_decoding_precision:
            Tells the number of bits used to encode the input (precision is 1/2**precision)
        Returns
        -------
            The input transformed

        """
        assert not (quantif and enc_dec)

        if R.ndim == 1:
            R = R.unsqueeze(-1)

        ctx.save_for_backward(input, SigFacts, directions, R)
        if quantif or enc_dec:
            encoder = SeparatedBitPlanEncoder(precision=encoding_decoding_precision)
            x_enc = encoder.fit_transform(input.data)
            decoder = SeparatedBitPlanDecoder(**encoder.get_params())

        else:
            x_enc = input

        if quantif and not enc_dec:
            # in case only quantification of input is requiered (testing purposes):
            # make quantification/dequantification directly. It is a quantification onto
            # one of the 2**precision possible values in the dynamic range.
            x_enc = decoder.transform(x_enc)

        weight_dtype = torch.promote_types(torch.promote_types(SigFacts.dtype, directions.dtype), R.dtype)
        y_dec = x_enc.to(weight_dtype).mm(directions)
        # y_dec: (B, M)
        # R: (M, NR)

        y_dec = y_dec.unsqueeze(-1) * R
        y_dec = torch.einsum("ijk,h->ikhj", y_dec, SigFacts)

        y_dec = y_dec.reshape((y_dec.shape[0], directions.shape[1] * SigFacts.shape[0] * R.shape[-1]))

        if not quantif and enc_dec:
            # standard scenario: dequantification happens after the transformation
            y_dec = decoder.transform(y_dec)

        return y_dec.to(weight_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, SigFacts, directions, R = ctx.saved_tensors
        grad_output = (grad_output).reshape((grad_output.shape[0], R.shape[-1], SigFacts.shape[0], directions.shape[1]))
        # grad_input = (torch.einsum("ikhj,h->ijk", grad_output, SigFacts) * R).mm(directions.t())
        grad_input = torch.einsum("ikhj,h->ijk", grad_output, SigFacts)
        grad_input = torch.einsum("ijk,jk->ij", grad_input, R).mm(directions.t())
        # grad_input = grad_output.mm(weight.t())

        # the three first Nones are for the weights which have fixed values
        # two last None correspond to `quantif` and `enc_dec` arguments in forward pass
        return grad_input, None, None, None, None, None, None


class LinearFunctionEncDec(Function):
    """
    A simple linear transformation by a weight matrix but with encoding/decoding or input quantification capabilities.

    These encoding/decoding and quantification options are for experimation purposes. Because they are not differentiable,
    the backward step just ignore them.
    """

    @staticmethod
    def forward(ctx, input, weight, quantif=False, enc_dec=False, encoding_decoding_precision=8):
        """

        Parameters
        ----------
        ctx
        input:
            tensor to transform.
        weight:
            1D tensor containing the list of sigmas.
        quantif:
            Tells to quantify the input before transformation (experimentation purposes).
        enc_dec:
            Tells to encode the input before transformation and decode after (experimentation purposes).
        encoding_decoding_precision:
            Tells the number of bits used to encode the input (precision is 1/2**precision)
        Returns
        -------
            The input transformed

        """
        assert not (quantif and enc_dec)

        ctx.save_for_backward(input, weight)
        if quantif or enc_dec:
            encoder = SeparatedBitPlanEncoder(precision=encoding_decoding_precision)
            x_enc = encoder.fit_transform(input.data)
            decoder = SeparatedBitPlanDecoder(**encoder.get_params())
        else:
            x_enc = input

        if quantif and not enc_dec:
            # in case only quantification of input is requiered (testing purposes):
            # make quantification/dequantification directly
            x_enc = decoder.transform(x_enc)

        # if ever using the opu here: careful with the type of x_enc and y_dec
        y_dec = x_enc.to(weight.dtype).mm(weight)

        if not quantif and enc_dec:
            # standard scenario: dequantification happens after the transformation
            y_dec = decoder.transform(y_dec)

        return y_dec.to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_input = grad_output.mm(weight.t())

        # first None is for the weights which have fixed values
        # two last None correspond to `quantif` and `enc_dec` arguments in forward pass
        return grad_input, None, None, None, None


class OPUFunctionEncDec(Function):

    @staticmethod
    def forward(ctx, input, opu_function, calibrated_opu, encoding_decoding_precision=8, nb_iter_linear_transformation=1):
        """
        Parameters
        ----------
        ctx
        input:
            tensor to transform.
        opu_function:
            The OPU function to call in the forward pass.
        calibrated_opu:
            Tells to quantify the input before transformation (experimentation purposes).
        encoding_decoding_precision:
            The OPU needs encoding before use.
            Tells the number of bits used to encode the input (precision is 1/2**precision).
        nb_iter_linear_transformation:
            The number of OPU calls to use for averaging the output. This was intended to reduce
            the noise of the OPU.
        Returns
        -------
            The input transformed
        """
        ctx.save_for_backward(input, calibrated_opu)
        encoder = SeparatedBitPlanEncoder(precision=encoding_decoding_precision)
        x_enc = encoder.fit_transform(input.data)
        decoder = SeparatedBitPlanDecoder(**encoder.get_params())

        y_dec = opu_function(x_enc)
        i_repeat_opu = 1
        while i_repeat_opu < nb_iter_linear_transformation:
            y_dec += opu_function(x_enc)
            i_repeat_opu += 1
        y_dec /= nb_iter_linear_transformation

        # standard scenario: dequantification happens after the transformation
        y_dec = decoder.transform(y_dec)

        return y_dec.to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, calibrated_opu = ctx.saved_tensors
        if calibrated_opu is None:
            raise NotImplementedError("Impossible to compute the gradient for the OPU transformation operation. "
                                      "Because no calibrated matrix was provided.")
        grad_input = grad_output.mm(calibrated_opu.t())

        # first None is for the weights which have fixed values
        # 4 last None correspond to `quantif` and `enc_dec`
        # and `save_outputs` and `nb_iter_linear_transformation` arguments in forward pass
        return grad_input, None, None, None, None
