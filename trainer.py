import torch
from transformers import Trainer as HFTrainer
from transformers.file_utils import is_apex_available

if is_apex_available():
    pass


class Trainer(HFTrainer):
    def __init__(self, label_smoothing: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing

    # override to support label smoothing
    # def training_step(
    #         self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    #         # optimizer: torch.optim.Optimizer
    # ) -> float:
    #     model.train()
    #     for k, v in inputs.items():
    #         if isinstance(v, torch.Tensor):
    #             inputs[k] = v.to(self.args.device)
    #
    #     # Our model outputs do not work with DataParallel, so forcing return tuple.
    #     if isinstance(model, nn.DataParallel):
    #         inputs["return_tuple"] = True
    #
    #     if self.label_smoothing == 0:
    #         outputs = model(**inputs)
    #         loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
    #     else:
    #         labels = inputs.pop("labels")
    #         labels[labels == -100] = model.config.pad_token_id
    #         outputs = model(**inputs)
    #         lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
    #         loss, nll_loss = label_smoothed_nll_loss(
    #             lprobs, labels, self.label_smoothing, ignore_index=model.config.pad_token_id
    #         )
    #
    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #     if self.args.gradient_accumulation_steps > 1:
    #         loss = loss / self.args.gradient_accumulation_steps
    #
    #     if self.args.fp16:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         loss.backward()
    #
    #     return loss.item()

    # def compute_loss(self, model, inputs):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.
    #     Subclass and override for custom behavior.
    #     """
    #
    #     sim_scores = inputs.pop('similarity_score')
    #     labels = inputs.pop('labels')
    #     model_output = model(**inputs)
    #
    #     logits = model_output["logits"] if isinstance(model_output, dict) else model_output[1]
    #     log_probs = torch.nn.functional.log_softmax(logits.view(-1, model.config.vocab_size), dim=-1)
    #     temp_loss = torch.nn.functional.nll_loss(log_probs, labels.view(-1), reduction="none")
    #     temp_loss = temp_loss.view(labels.shape)
    #     size_to_divide = temp_loss[temp_loss != 0].size()[0]
    #
    #     smoothed_loss = (temp_loss.sum(dim=-1) * sim_scores).sum() / size_to_divide
    #     model_loss = temp_loss.sum() / size_to_divide
    #
    #     return (1 - self.args.similarity_factor) * model_loss + self.args.similarity_factor * smoothed_loss
    #
    #     # override to support label smoothing


    # model.train()
    # inputs = self._prepare_inputs(inputs)
    #
    # sim_scores = inputs.pop('sim_scores')
    #
    # model_output = model(**inputs)
    #
    # model_loss = model_output["loss"] if isinstance(model_output, dict) else model_output[0]
    # logits = model_output["logits"] if isinstance(model_output, dict) else model_output[1]
    # log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
    #
    # # torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits.view(-1, 32102), dim=-1),
    # #                              inputs['labels'].view(-1), reduction="none").view(32, 31)[
    # #     torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits.view(-1, 32102), dim=-1),
    # #                                  inputs['labels'].view(-1), reduction="none").view(32, 31) != 0].mean()
    #
    # # Look at the ignored index and mask the corresponding log_probs.
    # padding_mask = inputs['labels'].unsqueeze(-1).eq(model.config.pad_token_id)
    # log_probs.masked_fill_(padding_mask, 0.0)
    #
    # # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    # smoothed_loss = log_probs.mean(dim=-1).sum() / (padding_mask.numel() - padding_mask.long().sum())
    #
    # return (1 - self.epsilon) * model_loss + self.epsilon * smoothed_loss
    #
    #
    # # Look at the ignored index and mask the corresponding log_probs.
    # # padding_mask = inputs['labels'].unsqueeze(-1).eq(model.config.pad_token_id)
    # # log_probs.masked_fill_(padding_mask, 0.0)
    #
    # # smoothed_loss = log_probs.mean(dim=-1).sum() / (padding_mask.numel() - padding_mask.long().sum())
    #
    # # gg = label_similarity_smoothed_nll_loss(log_probs, inputs['labels'], 0.1,model_loss,
    # #                                         ignore_index=model.config.pad_token_id,)
    # # if self.label_smoothing == 0:
    # #
    # #     loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
    # # else:
    # #     labels = inputs.pop("labels")
    # #     labels[labels == -100] = model.config.pad_token_id
    # #     outputs = model(**inputs)
    # #     lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
    # #     loss, nll_loss = label_similarity_smoothed_nll_loss(
    # #         lprobs, labels, sim_scores, ignore_index=model.config.pad_token_id
    # #     )
    #
