# Cloze eval set

`cloze.jsonl` is a hand-written Urdu cloze (fill-in-the-blank) set, 45 items,
authored for this project by Salman Adnan. It is not drawn from any external
benchmark.

Each line is a JSON object:

- `id`: item number
- `stem`: the sentence up to the blank (the model conditions on this)
- `options`: candidate completions; exactly one is correct
- `answer_idx`: index of the correct option in `options`
- `translation`: an English gloss of the correct sentence (for documentation)

Design notes:

- Items test basic factual and commonsense knowledge that any fluent Urdu model
  should get (capital of Pakistan, water boils, ice is cold, cows give milk).
- Distractors are grammatical and same-shape, so the task measures meaning, not
  surface form.
- The correct answer is placed first in every item for authoring convenience;
  the harness does not use position, it scores each option's log-probability.
- This is a probe for the phase-2 base model. The tiny CPU proof-run checkpoint
  scores near chance, which is expected and reported in the top-level README.

Run with `python -m eval.cloze --ckpt <checkpoint> --data evaldata/cloze.jsonl`.
