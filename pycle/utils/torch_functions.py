import torch
from lightonml.encoding.base import SeparatedBitPlanEncoder, SeparatedBitPlanDecoder
from torch.autograd import Function


# cleaning add documentation for what these functions do
# cleaning clean these functions
class MultiSigmaARFrequencyMatrixLinApEncDec(Function):

    @staticmethod
    def forward(ctx, input, SigFacts, directions, R, quantif=False, enc_dec=False, encoding_decoding_precision=8):
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
            # make quantification/dequantification directly
            x_enc = decoder.transform(x_enc)

        # if ever using the opu here: careful with the type of x_enc and y_dec

        # y_dec = x_enc.to(weight.dtype).mm(weight)
        weight_dtype = torch.promote_types(torch.promote_types(SigFacts.dtype, directions.dtype), R.dtype)
        y_dec = x_enc.to(weight_dtype).mm(directions)
        # y_dec: (B, M)
        # R: (M, NR)
        # how to multiply the transformed vector by R:
        # torch.einsum("ij,jk->ijk", xtd, R) == xtd.unsqueeze(-1) * R

        y_dec = y_dec.unsqueeze(-1) * R
        y_dec = torch.einsum("ijk,h->ikhj", y_dec, SigFacts)
        # y_dec = torch.einsum("ij,jkl->kil", SigFacts.unsqueeze(-1), y_dec.unsqueeze(0))

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

    @staticmethod
    # def forward(ctx, input, weight):
    def forward(ctx, input, weight, quantif=False, enc_dec=False, encoding_decoding_precision=8):
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
        # grad_weight = grad_output.t().mm(input)

        # first None is for the weights which have fixed values
        # two last None correspond to `quantif` and `enc_dec` arguments in forward pass
        return grad_input, None, None, None, None


class OPUFunctionEncDec(Function):

    @staticmethod
    # def forward(ctx, input, weight):
    def forward(ctx, input, opu_function, calibrated_opu, encoding_decoding_precision=8, nb_iter_linear_transformation=1):
        from pycle.utils.optim import IntermediateResultStorage

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

        # y_dec = x_enc.to(weight.dtype).mm(weight)

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
        # grad_weight = grad_output.t().mm(input)

        # first None is for the weights which have fixed values
        # 4 last None correspond to `quantif` and `enc_dec` and `save_outputs` and `nb_iter_linear_transformation` arguments in forward pass
        return grad_input, None, None, None, None