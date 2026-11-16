"""The VLM: Qwen3.5-2B (4-bit) + our LoRA adapter, loaded with Unsloth."""
import re, torch

class VLM:
    def __init__(self, adapter="unsloth/Qwen3.5-2B", load_in_4bit=True):
        from unsloth import FastVisionModel
        self.model, self.processor = FastVisionModel.from_pretrained(adapter, load_in_4bit=load_in_4bit)
        FastVisionModel.for_inference(self.model)
        tok = self.processor.tokenizer
        self.tok, self.end = tok, [i for i in {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} if isinstance(i, int) and i >= 0]

    @torch.inference_mode()
    def __call__(self, messages, max_new_tokens=48):
        x = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to("cuda")
        o = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, eos_token_id=self.end)
        raw = self.tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.split(r"<\|im_|\n", re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip())[0].strip()
