import sys
import argparse
import torch
from pathlib import Path
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed
from transformers.tokenization_utils_base import BatchEncoding


def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument(
    #     '-m', '--model-path',
    #     type=Path,
    #     default=Path('/home/akashc/almanac/checkpoints/QA-PubMedQA-BioGPT/checkpoint_avg.pt'),
    #     help='Path to model to use',
    # )
    p.add_argument(
        '-t', '--text-path', type=Path, default=None,
        help='File containing text prompts (split by newlines)',
    )
    p.add_argument(
        '-l', '--decoding-length',
        type=int,
        default=1024,
    )
    p.add_argument('-o', '--output-path', default=None)

    p.add_argument('-s', '--seed', default=42)

    return p.parse_args()


def recursively_to_cuda(inp):
    if isinstance(inp, (dict, BatchEncoding)):
        return {k: recursively_to_cuda(v) for k, v in inp.items()}
    if isinstance(inp, torch.Tensor):
        return inp.cuda()
    if isinstance(inp, list):
        return [recursively_to_cuda(i) for i in inp]
    assert False, f'Unsupported data type {type(inp)} for input {inp}'


def get_inputs(args: argparse.Namespace):
    if args.text_path:
        return [l.strip() for l in args.text_path.read_text().splitlines()]

    return [l.strip() for l in sys.stdin.read().splitlines()]


def prepare_input_sentence(sent: str):
    if not sent.startswith('question: '):
        sent = 'question: ' + sent

    if 'context: ' not in sent:
        sent = sent + ' context: '

    if 'answer: ' not in sent:
        sent = sent + ' answer: '

    if not sent.endswith('target: the answer to the question given the context is'):
        sent = sent + ' target: the answer to the question given the context is'

    return sent



def main(args: argparse.Namespace):
    # m = TransformerLanguageModelPrompt.from_pretrained(
    #     str(args.model_path.parent),
    #     str(args.model_path.name),
    #     '',
    #     max_len_b=args.decoding_length,
    #     max_tokens=12000,
    # )
    # print(f"{m.cfg=}")

    # if m.cfg.common.fp16:
    #     print("Converting model to fp16")
    #     m.half()

    tokenizer = BioGptTokenizer.from_pretrained('microsoft/BioGPT-Large-PubMedQA')
    m = BioGptForCausalLM.from_pretrained('microsoft/BioGPT-Large-PubMedQA')

    m.cuda()

    inputs = get_inputs(args)
    outputs = []
    for inp in inputs:
        inp = prepare_input_sentence(inp)
        print(f"Input:\n{inp}")
        tokens_in = tokenizer(inp, return_tensors='pt')
        tokens_in = recursively_to_cuda(tokens_in)
        generated = m.generate(**tokens_in, max_length=args.decoding_length, num_beams=1)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs.append(decoded)
        print(f"Output:\n{decoded}")

    if args.output_path:
        with open(args.output_path, 'w') as f:
            f.write('\n'.join(outputs))
    else:
        for out in outputs:
            print(out)


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'Torch unable to find GPUs!'
    ARGS = parse_args()
    set_seed(ARGS.seed)
    main(ARGS)
