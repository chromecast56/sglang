import argparse
import json
import os
import time
import uuid
from collections import defaultdict

from tabulate import tabulate

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def load_questions(filename, category=None, max_turns=None):
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)

            # Filter by category if specified
            if category and obj.get("category") != category:
                continue

            # Clone the object to avoid modifying the original
            question = obj.copy()

            # Limit number of turns if specified
            if max_turns is not None:
                question["turns"] = question["turns"][:max_turns]

            questions.append(question)
    return questions


def write_answers(filename, model_id, questions, answers):
    with open(os.path.expanduser(filename), "w") as fout:
        for i in range(len(answers)):
            ans_json = {
                "question_id": questions[i]["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "choices": {
                    "index": 0,
                    "turns": answers[i],
                },
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


@sgl.function
def answer_spec_bench_single(s, question_1):
    s += sgl.system("")
    # print("After system message:\n", s.text())
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))


@sgl.function
def answer_spec_bench_double(s, question_1, question_2):
    s += sgl.system("")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2"))


def main(args):
    # Construct prompts
    questions = load_questions(args.question_file, args.category, args.max_turns)
    if args.num_questions != -1:
        questions = questions[: args.num_questions]

    if not questions:
        print(f"No questions found with the specified filters.")
        return

    # Group questions by category
    questions_by_category = defaultdict(list)
    for question in questions:
        category = question.get("category", "unknown")
        questions_by_category[category].append(question)

    # Determine if we're using single or double turns
    max_turns = args.max_turns if args.max_turns is not None else 2
    use_single_turn = max_turns == 1

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Process each category separately
    all_answers = []
    metrics_by_category = {}
    total_output_tokens = 0
    total_verify_tokens = 0
    total_latency = 0

    for category, category_questions in questions_by_category.items():
        print(f"Processing category: {category} ({len(category_questions)} questions)")

        # Prepare arguments for this category
        if use_single_turn:
            arguments = [{"question_1": q["turns"][0]} for q in category_questions]
            answer_func = answer_spec_bench_single
        else:
            arguments = [
                {
                    "question_1": q["turns"][0],
                    "question_2": q["turns"][1] if len(q["turns"]) > 1 else "",
                }
                for q in category_questions
            ]
            answer_func = answer_spec_bench_double

        # Run requests for this category
        tic = time.time()
        category_rets = answer_func.run_batch(
            arguments,
            temperature=0,
            max_new_tokens=args.max_new_tokens,
            num_threads=args.parallel,
            progress_bar=True,
        )
        category_latency = time.time() - tic
        total_latency += category_latency

        # Process results for this category
        if use_single_turn:
            category_answers = [[s["answer_1"]] for s in category_rets]
        else:
            category_answers = [[s["answer_1"], s["answer_2"]] for s in category_rets]

        all_answers.extend(category_answers)

        # Calculate metrics for this category
        if use_single_turn:
            category_output_tokens = sum(
                s.get_meta_info("answer_1")["completion_tokens"] for s in category_rets
            )
            has_verify = "spec_verify_ct" in category_rets[0].get_meta_info("answer_1")
            if has_verify:
                category_verify_tokens = sum(
                    s.get_meta_info("answer_1")["spec_verify_ct"] for s in category_rets
                )
            else:
                category_verify_tokens = category_output_tokens
        else:
            category_output_tokens = sum(
                s.get_meta_info("answer_1")["completion_tokens"]
                + s.get_meta_info("answer_2")["completion_tokens"]
                for s in category_rets
            )
            has_verify = "spec_verify_ct" in category_rets[0].get_meta_info("answer_1")
            if has_verify:
                category_verify_tokens = sum(
                    s.get_meta_info("answer_1")["spec_verify_ct"]
                    + s.get_meta_info("answer_2")["spec_verify_ct"]
                    for s in category_rets
                )
            else:
                category_verify_tokens = category_output_tokens

        category_throughput = (
            category_output_tokens / category_latency if category_latency > 0 else 0
        )
        category_accept_length = (
            category_output_tokens / category_verify_tokens
            if category_verify_tokens > 0
            else 1.0
        )

        metrics_by_category[category] = {
            "num_questions": len(category_questions),
            "throughput": category_throughput,
            "accept_length": category_accept_length,
            "output_tokens": category_output_tokens,
            "latency": category_latency,
        }

        total_output_tokens += category_output_tokens
        total_verify_tokens += category_verify_tokens

    # Calculate overall metrics
    overall_throughput = total_output_tokens / total_latency if total_latency > 0 else 0
    overall_accept_length = (
        total_output_tokens / total_verify_tokens if total_verify_tokens > 0 else 1.0
    )

    # Display summary table
    table_data = []
    headers = [
        "Category",
        "Questions",
        "Latency (s)",
        "Throughput (token/s)",
        "Accept Length",
    ]

    for category, metrics in sorted(metrics_by_category.items()):
        table_data.append(
            [
                category,
                metrics["num_questions"],
                f"{metrics['latency']:.2f}",
                f"{metrics['throughput']:.2f}",
                f"{metrics['accept_length']:.2f}",
            ]
        )

    # Add total row
    table_data.append(
        [
            "TOTAL",
            len(questions),
            f"{total_latency:.2f}",
            f"{overall_throughput:.2f}",
            f"{overall_accept_length:.2f}",
        ]
    )

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Write results
    model_id = backend.model_info["model_path"]
    answer_file = args.answer_file or f"tmp_output_specbench_{args.backend}.txt"
    write_answers(answer_file, model_id, questions, all_answers)

    with open(args.result_file, "a") as fout:
        # Overall metrics
        value = {
            "task": "specbench",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(total_latency, 3),
            "throughput": round(overall_throughput, 3),
            "accept_length": round(overall_accept_length, 3),
            "num_requests": len(questions),
            "other": {
                "max_turns": args.max_turns,
                "category": args.category,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")

        # Per-category metrics
        for category, metrics in metrics_by_category.items():
            category_value = {
                "task": "specbench",
                "category": category,
                "backend": args.backend,
                "num_gpus": 1,
                "latency": round(metrics["latency"], 3),
                "throughput": round(metrics["throughput"], 3),
                "accept_length": round(metrics["accept_length"], 3),
                "num_requests": metrics["num_questions"],
                "other": {
                    "max_turns": args.max_turns,
                    "parallel": args.parallel,
                },
            }
            fout.write(json.dumps(category_value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=-1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--backend", type=str, default="srt")
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument(
        "--category", type=str, default=None, help="Filter questions by category"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=2,
        help="Maximum number of turns (1 for first turn only)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate",
    )
    args = parser.parse_args()
    main(args)
