# Sign Language → Sentence → Speech (v2 — word-level)

මේක oyage kalin project eke **major upgrade** ekak. Wenas una pradhana deyak:
letter-ekakata ekak (A, B, C...) wenuwata, **hand gesture ekakata sampurna
word ekak** (hello, thanks, water...) recognize wenawa dæn.

---

## 1. Model eka wenas kala ayi (important — please read)

Word ekak sign karana kota, ath eka **kal ekaka yanna** wenawa (motion) —
eka static photo ekakin therum ganna bæri deyak. Free, ready-made,
"download-karala-witharak-use-karanna-puluwan" word-level sign model ekak
market eke pihitiye නෑ (thiyena ewa hariම heavy video datasets download
karanna one wenawa, eka melowma unreliable).

Ehema nisa mema version eke:
- **MediaPipe Hands** (free, open-source) use karanne **ath eke points 21
  (landmarks) hoyaganna witharai** — eka word eka kiyala therum gannne නෑ.
- Ee landmark sequence eka (30 frames, ~1 second) **api ma train karana
  LSTM neural network ekakata** (`train_lstm.py`) denawa, eka thamai
  "AI intelligence" eka — word eka predict karanne.
- Data eka hoyaganne oyage ma webcam eken record karagath samples walin
  (`collect_data.py`) — internet eken pretrained word-model ekak download
  karanna one නෑ.

Meka fully free + open-source pipeline ekak, oyagema data walin, oyagema
model ekak train karagena.

---

## 2. Setup

```bash
pip install -r requirements.txt
```

## 3. Quick test — instant placeholder model (no recording needed)

Want to test the app RIGHT NOW, before spending time recording real signs?

```bash
python make_demo_model.py
```

This creates `word_model.h5` + `word_labels.json` from synthetic (made-up)
data in a few seconds. ⚠️ It will NOT actually recognise your real hand
signs correctly — it's only there so you can test the camera, power
switch, mute, speak, copy, sentence box and animations end-to-end. When
you're ready for real accuracy, follow steps 4 and 5 below with your own
recorded gestures (this overwrites the demo files with a real model).

## 4. Step 1 — Your own gesture data record karanna

```bash
python collect_data.py
```

Default vocabulary eka: **hello, thanks, yes, no, please, help, water,
food, stop, love** (10 words). Word ekak wenas karanna one nam:

```bash
python collect_data.py --words hello thanks yes no please
```

- Camera window ekak open wenawa.
- Word ekak pennanawa, "Press SPACE to record" kiyala.
- SPACE click kaloth 3,2,1 countdown ekak, passe "RECORDING..." kියලා 30
  frames record wenawa (~1 second, ehema athara sign eka pennanna).
- Word ekakata sample 30k default widiyata record wenawa (`--sequences`
  eken wenas karanna puluwan). Sample gaana godak una witharak model eka
  accurate wenne — words 10ta samples 30-40k hodai.
- **N** = ee word eka skip karala issella yanawa, **Q** = quit.

## 5. Step 2 — Model eka train karanna (real accuracy eka)

```bash
python train_lstm.py
```

Meken `word_model.h5` saha `word_labels.json` hadenawa. CPU ekaka minutes
kihipayak witharai yanne (data set eka podi nisa).

## 6. Step 3 — App eka run karanna

```bash
python app.py
```

---

## 7. App eka use karana widiya

1. **⏻ Power** button eka default OFF widiyata thiyenne — click karala ON
   karanna one, ehema unahamai recognition patan gannne.
2. **⛶ Fullscreen** button (ho keyboard eke **F11**) — app eka full screen
   widiyata daganna puluwan. Fullscreen eken naganna **Esc** click karanna.
   Window eka normal widiyata drag karala resize karanna puluwanma —
   layout eka fully responsive, ehema nisa resize/fullscreen kaloth
   buttons monawahari hide wenne නෑ.
3. Green box/ROI ekak nෑ dæn — camera eke ehema kohewath hand eka pennanna
   puluwan, MediaPipe eken automatic-ma detect wenawa.
4. Hand eka frame ekata pennana welawe witharai anything recognize wenne —
   hand ekak nætnam kisima word ekak type wenne නෑ.
5. **BUILDING WORD** bar eka (sentence box ekata issellin thiyena eka) —
   hand eka pennana athara model eka dæne monawada therum ganne kiyala
   **live** widiyata pennanawa (confirm wela commit wena kalinma). Word eka
   confidently confirm unahama:
   - **automatic-ma space ekak samaga sentence ekata add wenawa**,
   - saha (mute wela nætnam) **wenama voice ekak widiyata kiyawenawa**.
6. **🔔/🔕 Mute button** — default Unmuted. Mute kaloth aluth word
   recognize unath text widiyata witharak enawa, voice ekak enne නෑ.
   (Manual "Speak" button eka mute wela unath vada karanawa.)
7. **🔊 Speak** button eka — dæna thiyena sentence eka mulinsita kiyawagena
   yanawa. E athara aluth hand gesture ekak recognize unoth, **e kiyawana
   eka ITHIN nawattala, aluth word eka witharak kiyawanawa** (mulin idala
   sentence eka apahu kiyawanne නෑ).
8. **⧉ Copy** button — sentence eka clipboard ekata copy karagannawa.
9. **␣ Add Space** — manual widiyata space ekak add karanna.
10. **⌫ Backspace** — anthima recognize kara gath word eka ain karagannawa
    (gestures walin enne sampurna words nisa, character ekak witharak
    ain karanne නෑ — word ekma ain wenawa).
11. **🗑 Clear All** — okkoma clear karanawa.
12. **Language dropdown** (top-right) — dæne "English" witharai thiyenne.
    Language eka wenas kala prayathnayeka confirmation dialog ekak
    ehewela, "Yes" kiyoth dæna thiyena sentence eka clear wela aluthen
    patan ganninnawa.

---

## 8. Design notes (tkinter limitations - honest note)

Blue gradient background eka, rounded cards, pulsing status dot, flash
animations okkoma native Python's tkinter walin (external design library
ekak nætuwa) hadala thiyenne. Tkinter eke true CSS-style blur/shadow
support නෑ, ehema nisa "glassmorphism" wage effects perfect widiyata
gihin nෑ — eth gradient + rounded corners + smooth color-transition
animations walin app eka godak modern widiyata penenawa kalin static
grey UI ekata sæsandin.

## 9. Honest limitations

- Word eka hoda accuracy ekakin recognize wenna oyama e word 30-40k widiyaka
  consistent widiyata record karanna one (lighting/background wenas kara
  kara record kaloth accuracy hondai).
- Sign wenas karana athara hands frame eken pitatha giyoth (ho hidden
  unoth), buffer eka reset wela aluthen patan gannawa — sign eka mulinsita
  hands penena widiyata karanna one.
- Meka word-vocabulary ekak witharai (10-20 words wage), full natural sign
  language grammar ekak nemei.
