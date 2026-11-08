# V-Vet symptom extraction

Farmers describe a sick animal in their own words. `POST /extract` turns one message into signs from a fixed list, then decides whether the case should be escalated.

The model only proposes findings. Escalation is ordinary Python: escalate when `froth_nose_mouth`, `unable_to_stand`, or `cloudy_eye` is `yes` or `not_sure`.

## Run

From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

`pytest` does not call a model and does not need an API key.

To try a real model, copy the example env file and set a key. Any OpenAI-compatible host works, including free tiers.

```powershell
copy .env.example .env
# then edit LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL
uvicorn app.main:app --reload
```

```powershell
curl.exe -s http://127.0.0.1:8000/extract -H "Content-Type: application/json" -d "{\"message\": \"There's foam coming from her nose and she can't get up.\"}"
```

The five sample messages can also be sent straight through the client:

```powershell
python scripts/try_samples.py
```

That script is not part of the tests. It needs `LLM_API_KEY`.

If a host rejects JSON mode, set `LLM_JSON_MODE=0`. The parser still accepts a raw JSON object, a ```json fence, or JSON wrapped in a sentence.

## What the samples should do

The tests below lock the rules, using a stand-in for the model. They do not prove that a live model will extract these signs. Check that with `scripts/try_samples.py`.

| Message | Findings, if the model read it correctly | Escalate |
| --- | --- | --- |
| Not eating since yesterday, breathing very fast | `reduced_appetite` yes, `rapid_breathing` yes | no |
| Foam from the nose, can't get up | `froth_nose_mouth` yes, `unable_to_stand` yes | yes |
| Eyes might be cloudy; mbandabvu under the ear | `cloudy_eye` not_sure, `swelling` yes | yes |
| Calf is fine, lots of ticks | `ticks` yes | no |
| "Haisi kudya", urine dark like tea | `reduced_appetite` yes, `red_urine` yes | no |

"The calf is fine" is not a denial of every sign. A `no` is only for an explicit denial, such as "there is no foam". An explicit `no` on an urgent sign does not escalate.

## Decisions

- The sign list lives in `app/signs.py`. The prompt is built from that list. A finding is kept only when its `sign_id` is in the list and its `value` is `yes`, `no`, or `not_sure`.
- A quote must appear in the farmer's message, ignoring case and extra spaces. A paraphrase is dropped. This is the check that stops the model inventing a sign the farmer did not mention. It does not check that the quote actually supports that sign: the word "ticks" is in the message, and a model could attach it to the wrong sign. Measuring that needs a person.
- There is no keyword list beside the model. A list would miss Shona and mixed messages ("haisi kudya") and would fight the model on paraphrases. `mbandabvu` is in the swelling definition because the brief puts it there.
- The sample messages are not written into the prompt. Putting the exam questions in the prompt would make a demo look better than it is.
- If the model returns the same sign twice, the more serious value wins: `yes`, then `not_sure`, then `no`.
- Findings come back in sign-list order so the response is stable.
- Extra keys from the model, including `escalate` and `reason`, are ignored.
- A reply that is not JSON is HTTP 502. So is a reply whose findings were all rejected. An empty `findings` list is a real answer and does not escalate. The difference matters: a dead provider must not look like a healthy animal. The WhatsApp side should treat 502 as urgent.
- Messages are stripped and limited to 4000 characters. Logs record how many findings were dropped, not the farmer's text.
- The client talks to `POST {LLM_BASE_URL}/chat/completions`, so the host can be OpenAI, Groq, or another compatible API. The key stays in `.env`, which is not committed.

## How I would measure accuracy

Escalation is a pure function of the findings, so I would not score it on its own. I would score extraction, then count how often a wrong finding would have changed the escalate bit.

I would build a gold set of real WhatsApp messages, labeled by a vet: `sign_id`, `value`, and the span that supports it. A second person would label the same messages. I would report agreement before trusting the set.

On that set I would track:

- Precision and recall per sign, for `yes`. Urgent signs (`froth_nose_mouth`, `unable_to_stand`, `cloudy_eye`) reported separately, because a miss there is worse than a false alarm.
- How often `yes` and `not_sure` are confused. Hedge words ("I think", "not sure") are their own slice.
- Exact match on the set of `(sign_id, value)`.
- Quote quality: the quote is already required to be a substring; a vet still has to say whether that span supports the sign.
- Slices for English, Shona, and mixed messages.
- The rate of dropped findings and 502s. A high drop rate means the model is paraphrasing quotes, which hides real signs.

The five samples in this repo are smoke tests for the rules. They are too small, and too familiar, to be an accuracy number. I would not tune the prompt on them and then quote a score.

## With more time

- If every finding is rejected, or the provider errors, retry once with the validation error in the prompt. If it still fails, fail closed.
- A few real dialogues in the prompt, chosen with a vet, including Shona. Keep them out of the evaluation set.
- Store the raw model text for audit. Do not log it in application logs.
- Ask a follow-up on WhatsApp when an urgent sign is `not_sure`, instead of only escalating.
- Split messages that mention more than one animal.
- Confirm data handling with the team before sending farmer messages to a third-party model. A local model may be required.
- Run the gold set on a schedule against the live model. Keep the offline tests as they are.

## AI tools

I used Cursor's coding assistant to draft this service from the task brief: the FastAPI endpoint, the sign check, the escalation function, the offline tests, and this README. I kept the design constraint that the model proposes findings and plain code decides escalation, and that a quote has to be the farmer's own words. The tests stand in for the model so the suite runs with no key and no network.
