from transformers import AutoTokenizer
from typing import Dict, Any


def format_chat_with_tokenizer(tokenizer: AutoTokenizer, example: Dict[str, Any], use_template: bool = True) -> str:
    """Format a single example into a plain text prompt using the tokenizer's chat template when possible.

    Supports common schemas:
    - {"messages": [{"role": "user|assistant|system", "content": str}, ...]}
    - {"conversations": same as messages}
    - instruction-style: {instruction, input?, output/response}
    """
    if "gpt-oss-20b" in getattr(tokenizer, "name_or_path", ""):
        print("[chat_template] Detected gpt-oss tokenizer; enabling thinking and generation prompts for better alignment.")
        enable_thinking = True
        chat_template_args = {"reasoning_effort": "low"}
    else:
        enable_thinking = False
        chat_template_args = None
    
    # 1) Chat-style data
    msgs = example.get("messages") or example.get("conversations")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and "role" in msgs[0]:
        if not use_template:
            return "\n".join([m.get("content", "") for m in msgs])
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                **(chat_template_args or {})
            )
        except Exception:
            raise NotImplementedError("Please implement chat-style formatting fallback.")

    # 2) Instruction-style data
    instr = example.get("instruction") or example.get("prompt")
    inp = example.get("input") or example.get("prompt")
    out = example.get("output") or example.get("response") or example.get("answer")
    if instr or out:
        msgs = []
        if instr:
            if inp:
                full_instr = f"{instr}\n\nInput: {inp}"
            else:
                full_instr = instr
            msgs.append({"role": "user", "content": full_instr})
        if out:
            msgs.append({"role": "assistant", "content": out})
        
        if not use_template:
            return "\n".join([m["content"] for m in msgs])

        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                **(chat_template_args or {})
            )
        except Exception:
            raise NotImplementedError("Please implement instruction-style formatting fallback.")

    # 3) Plain text
    if example.get("text"):
        if not use_template:
            return example["text"]
        raise NotImplementedError("Please implement text formatting for 'text' field.")

    raise ValueError("Unable to format example into chat prompt.")
