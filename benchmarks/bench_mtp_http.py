"""Measure chat SSE delivery with sampling parameters left at server defaults."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def generate(url, model, prompt, tokens):
    body = dict(model=model, messages=[dict(role="user", content=prompt)],
                max_tokens=tokens, stream=True, stream_options=dict(include_usage=True))
    request = Request(url.rstrip("/") + "/v1/chat/completions",
                      json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.perf_counter()
    first = last = None
    text, reasoning, usage, finishes, done = [], [], None, [], False
    with urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                thought = delta.get("reasoning_content") or ""
                if content or thought:
                    last = time.perf_counter()
                    first = first or last
                    text.append(content)
                    reasoning.append(thought)
                if choice.get("finish_reason"):
                    finishes.append(choice["finish_reason"])
    assert usage and first is not None and last > first, "missing streamed text or token usage"
    assert done and len(finishes) == 1, "missing DONE or duplicate finish event"
    return dict(request=body, seconds=time.perf_counter() - started,
                ttft_seconds=first - started, delivery_seconds=last - first,
                delivery_tokens_per_second=(usage["completion_tokens"] - 1) / (last - first),
                usage=usage, finish_reasons=finishes,
                text="".join(text), reasoning="".join(reasoning))


def validate(url, model, prompt):
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    def request(body):
        return urlopen(Request(endpoint, json.dumps(body).encode(),
                               {"Content-Type": "application/json"}), timeout=300)
    body = dict(model=model, messages=[dict(role="user", content=prompt)], max_tokens=128,
                temperature=0, stream=False)
    with request(body) as response:
        reference = json.load(response)
    message = reference["choices"][0]["message"]
    text = (message.get("reasoning_content") or "") + (message.get("content") or "")
    assert len(text) > 24
    marker = text[12:24]
    with request(dict(body, stop=[marker])) as response:
        stopped = json.load(response)
    assert stopped["choices"][0]["finish_reason"] == "stop"
    returned = stopped["choices"][0]["message"]
    assert marker not in (returned.get("content") or "") + (returned.get("reasoning_content") or "")
    aborted_events = 0
    with request(dict(body, stream=True, max_tokens=4096)) as response:
        for line in response:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                event = json.loads(line[6:])
                if any(c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")
                       for c in event.get("choices", [])):
                    aborted_events += 1
                if aborted_events >= 32:
                    break
    assert aborted_events >= 32
    recovery = generate(url, model, prompt, 64)
    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = list(executor.map(lambda _: generate(url, model, prompt, 128), range(2)))
    return dict(stop_marker=marker, stop_response=stopped, aborted_text_events=aborted_events,
                abort_recovery=recovery, concurrent=concurrent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="猫の魅力について熱くたくさん語って。")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    report = dict(settings=vars(args), runs=[])
    if args.warmup_tokens:
        report["warmup"] = generate(args.url, args.model, args.prompt, args.warmup_tokens)
    for repeat in range(args.repeats):
        result = generate(args.url, args.model, args.prompt, args.tokens)
        report["runs"].append(dict(repeat=repeat, **result))
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("request", "text", "reasoning")}), flush=True)
    if args.validate:
        report["validation"] = validate(args.url, args.model, args.prompt)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
