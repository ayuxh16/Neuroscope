import json, urllib.request as u
def post(p, b):
    r = u.Request('http://127.0.0.1:8000' + p, json.dumps(b).encode(), {'Content-Type': 'application/json'})
    return json.load(u.urlopen(r))
prompt = "The Eiffel Tower is located in"
rows = []
for l in range(12):
    for h in range(12):
        r = post('/ablate', {"prompt": prompt, "heads_to_ablate": [{"layer": l, "head": h}]})
        shift = sum(d["abs_delta"] for d in r["probability_diff"])
        rows.append((shift, l, h, r["output_changed"], r["ablated"]["top_tokens"][0]["token"]))
    print("layer", l, "done", flush=True)
rows.sort(reverse=True)
print("\nTop 10 heads by total probability shift:")
for s, l, h, ch, top in rows[:10]:
    print(f"L{l}H{h}  shift={s:.3f}  text_changed={ch}  top_token_after={top!r}")
