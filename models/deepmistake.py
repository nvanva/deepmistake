from models.utils import Example, get_dataloader_and_tensors
from models.xlmr import XLMRModel
from run_model import run_inference


class DeepMistake:
    """
    TODO: This is a sketch of the code. Arshan, please finish and test.
    """

    def __init__(self, ckpt_dir, device):
        """
        Loads a DeepMistake model from the specified checkpoint directory.
        """
        self.device = device
        # Loads the model, similarly to run_model.py, lines 619-623
        self.model = XLMRModel.from_pretrained()  # TODO: fill in the arguments
        self.model.to(device)

    def predict(self, examples: list[Example], batch_size=16):
        """
        Runs inference for the provided list of examples.
        """
        # Runs inference, similarly to run_model.py, lines 624-634
        test_features = self.model.convert_examples_to_features(examples)
        eval_dataloader = get_dataloader_and_tensors(test_features, batch_size)
        metrics, syns_preds, syns_scores_res = run_inference(eval_dataloader, self.model, self.device,
                                                             compute_metrics=False, dump_feature=False, output_dir=None)
        # TODO: syns_scores_res should contain len(examples) or 2*len(examples) scores when test time augmentation
        #  is disabled or enabled correspondingly ("symmetric" argument in run_model.py);
        #  in the second case we would like to reformat the results,
        #  returning array of shape (len(examples), 2) - this will make possible experiments with different aggregation
        #  of scores for symmetric versions of the same usage pairc
        # Same for syns_preds!
        assert syns_scores_res.shape[0] == (len(examples))
        assert syns_preds.shape[0] == (len(examples))
        return syns_scores_res, syns_scores_res

