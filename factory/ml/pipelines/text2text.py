from factory.ml.pipelines.general import PipelineMixin
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class SummarizationPipeline(PipelineMixin):
    output_type = 'text'
    
    def __init__(self, model_name='pszemraj/led-large-book-summary'):
        self.model_name = model_name
        self.device = 'cpu' #cuda' if torch.cuda.is_available() else 'cpu'

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "text-to-text"

    def run_task(self, task):
        if task.task == "text-to-text":
            return self.text_to_text(task.inputs, task.parameters)
        raise ValueError("Invalid task")

    def get_options(self):
        return {
            'task': 'text-to-text',
            'output_type': 'text',
            'parameters': {
                'inputs': 'A text to summarize',
            }
        }

    def summarize_and_score(self, ids, mask, **kwargs):
        ids = ids[None, :]
        mask = mask[None, :]
        
        input_ids = ids.to(self.device)
        attention_mask = mask.to(self.device)
        global_attention_mask = torch.zeros_like(attention_mask)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1

        summary_pred_ids = self.model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask, 
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )
        summary = self.tokenizer.batch_decode(
            summary_pred_ids.sequences, 
            skip_special_tokens=True,
            remove_invalid_values=True,
        )
        score = round(summary_pred_ids.sequences_scores.cpu().numpy()[0], 4)
        
        return summary, score
        
    def summarize_via_tokenbatches(
            self,
            input_text:str,
            batch_length=8192,
            batch_stride=16,
            **kwargs,
        ):
        
        encoded_input = self.tokenizer(
            input_text, 
            padding='max_length', 
            truncation=True,
            max_length=batch_length, 
            stride=batch_stride,
            return_overflowing_tokens=True,
            add_special_tokens =False,
            return_tensors='pt',
        )
        
        in_id_arr, att_arr = encoded_input.input_ids, encoded_input.attention_mask
        gen_summaries = []
        for _id, _mask in zip(in_id_arr, att_arr):

            result, score = self.summarize_and_score(
                ids=_id, 
                mask=_mask, 
                #**kwargs,
            )
            score = round(float(score),4)
            _sum = {
                "input_tokens":_id,
                "summary":result,
                "summary_score":score,
            }
            gen_summaries.append(_sum)
            print(f"\t{result[0]}\nScore:\t{score}")

        return gen_summaries

    def text_to_text(self, text, options):
        return self.summarize_via_tokenbatches(text, **options)


