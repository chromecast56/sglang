import random
import threading
import time
import unittest
from types import SimpleNamespace

import requests

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_PHOENIX_DRAFT_MODEL_FOR_TEST,
    DEFAULT_PHOENIX_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestPhoenixEngine(unittest.TestCase):
    BASE_CONFIG = {
        "model_path": DEFAULT_PHOENIX_TARGET_MODEL_FOR_TEST,
        "speculative_draft_model_path": DEFAULT_PHOENIX_DRAFT_MODEL_FOR_TEST,
        "speculative_algorithm": "PHOENIX",
        "speculative_num_steps": 5,
        "speculative_phoenix_topk": 8,
        "speculative_num_draft_tokens": 64,
        "mem_fraction_static": 0.7,
        "cuda_graph_max_bs": 32,
    }

    def setUp(self):
        self.prompt = "Today is a sunny day and I like"
        self.sampling_params = {"temperature": 0, "max_new_tokens": 8}

        ref_engine = sgl.Engine(model_path=DEFAULT_PHOENIX_TARGET_MODEL_FOR_TEST)
        self.ref_output = ref_engine.generate(self.prompt, self.sampling_params)["text"]
        ref_engine.shutdown()

    def test_phoenix_accuracy(self):
        configs = [
            self.BASE_CONFIG,
            {**self.BASE_CONFIG, "disable_cuda_graph": True},
        ]

        for config in configs:
            with self.subTest(
                cuda_graph=(
                    "enabled" if len(config) == len(self.BASE_CONFIG) else "disabled"
                )
            ):
                engine = sgl.Engine(**config)
                try:
                    self._test_basic_generation(engine)
                    self._test_eos_token(engine)
                    self._test_batch_generation(engine)
                finally:
                    engine.shutdown()

    def _test_basic_generation(self, engine):
        output = engine.generate(self.prompt, self.sampling_params)["text"]
        print(f"{output=}, {self.ref_output=}")
        self.assertEqual(output, self.ref_output)

    def _test_eos_token(self, engine):
        prompt = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\nToday is a sunny day and I like [/INST]"
        params = {
            "temperature": 0,
            "max_new_tokens": 1024,
            "skip_special_tokens": False,
        }

        tokenizer = get_tokenizer(DEFAULT_PHOENIX_TARGET_MODEL_FOR_TEST)
        output = engine.generate(prompt, params)["text"]
        print(f"{output=}")

        tokens = tokenizer.encode(output, truncation=False)
        self.assertNotIn(tokenizer.eos_token_id, tokens)

    def _test_batch_generation(self, engine):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        params = {"temperature": 0, "max_new_tokens": 30}

        outputs = engine.generate(prompts, params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Generated: {output['text']}")
            print("-" * 40)


prompts = [
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nToday is a sunny day and I like[/INST]"
    '[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nWhat are the mental triggers in Jeff Walker\'s Product Launch Formula and "Launch" book?[/INST]',
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nSummarize Russell Brunson's Perfect Webinar Script...[/INST]",
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwho are you?[/INST]",
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwhere are you from?[/INST]",
]


if __name__ == "__main__":
    unittest.main()
