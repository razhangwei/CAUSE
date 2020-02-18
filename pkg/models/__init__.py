from .rnn import EventSeqDataset, RecurrentMarkDensityEstimator

__all__ = ["EventSeqDataset", "RecurrentMarkDensityEstimator"]

# def get_model(model, **kwargs):

#     if model == 'N-SCCS':
#         model = NeuralSCCS(**kwargs)
#     elif model == 'MSCCS':
#         kwargs["mlp_config"] = {"n_output": kwargs["n_outcomes"]}
#         model = NeuralSCCS(**kwargs)
#     elif model == "LLH":
#         model = LogLinearHawkes(**kwargs)
#     elif model == "SLLH":
#         model = StochasticLogAdditiveHawkes(include_ddi=False, **kwargs)
#     elif model == "SLA^2H":
#         model = StochasticLogAdditiveHawkes(include_ddi=True, **kwargs)
#     else:
#         raise NotImplementedError()

#     return model
