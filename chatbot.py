"""
On-device AI assistant for the pest trap dashboard.

Fully self-contained: no external API key or cloud model is needed, so it
works 100% offline. It is grounded in two sources of truth:

  1. Live trap data  — it reads the same stats / latest-detection / pipeline
     context the dashboard shows, so answers reflect what the trap saw.
  2. A pest knowledge base — short blurbs for common insects, with the
     Harmful/Beneficial/Harmless verdict computed by the *same* mapping the
     trap uses (pest_categories.label_to_category), so the chat never
     disagrees with the trap.

Supports three languages: English ("en"), Sinhala ("si"), Tamil ("ta").
reply(text, ctx, lang) is a small intent router that composes a friendly
answer in the requested language (falling back to English where a string
is not translated).
"""

import re

from pest_categories import (
    CATEGORY_BENEFICIAL,
    CATEGORY_HARMFUL,
    CATEGORY_HARMLESS,
    label_to_category,
)

# --------------------------------------------------------------- knowledge --
# term -> (display name, {lang: blurb}). Verdict comes from label_to_category
# so the chat always agrees with the trap's decision logic.
KNOWLEDGE = {
    "ladybug": ("ladybug", {"en": "a voracious aphid eater — farmers love them",
        "si": "අඵඩ් කෘමීන් භක්ෂණය කරන — ගොවීන්ට ඉතා ප්‍රිය", "ta": "அஃபிட் பூச்சிகளை தின்னும் — விவசாயிகளுக்கு பிடித்தது"}),
    "ladybird": ("ladybug", {"en": "a voracious aphid eater — farmers love them",
        "si": "අඵඩ් කෘමීන් භක්ෂණය කරන — ගොවීන්ට ඉතා ප්‍රිය", "ta": "அஃபிட் பூச்சிகளை தின்னும் — விவசாயிகளுக்கு பிடித்தது"}),
    "bee": ("bee", {"en": "a crucial pollinator; responsible for a third of what we eat",
        "si": "වැදගත් පරාගණකාරයෙක්; අප කන ආහාරයෙන් තුනෙන් එකකට වගකිව යුතු", "ta": "முக்கிய மகரந்த சேர்க்கை; நாம் உண்ணும் உணவில் மூன்றில் ஒன்றுக்கு காரணம்"}),
    "honey bee": ("bee", {"en": "a crucial pollinator; responsible for a third of what we eat",
        "si": "වැදගත් පරාගණකාරයෙක්", "ta": "முக்கிய மகரந்த சேர்க்கை"}),
    "mantis": ("praying mantis", {"en": "a predator that hunts many crop pests",
        "si": "බෝග පළිබෝදුන් දඩයම් කරන විලෝපිකයෙක්", "ta": "பயிர் பூச்சிகளை வேட்டையாடும் வேட்டைக்காரன்"}),
    "dragonfly": ("dragonfly", {"en": "an aerial hunter that eats mosquitoes and flies",
        "si": "මදුරුවන් හා මැස්සන් කන ගුවන් දඩයක්කාරයෙක්", "ta": "கொசு மற்றும் ஈக்களை தின்னும் வேட்டைக்காரன்"}),
    "butterfly": ("butterfly", {"en": "a pollinator; harmless to crops as an adult",
        "si": "පරාගණකාරයෙක්; වැඩුණු පසු බෝගවලට හානියක් නැත", "ta": "மகரந்த சேர்க்கை; வளர்ந்த பிறகு பயிர்களுக்கு தீங்கில்லை"}),
    "monarch": ("monarch butterfly", {"en": "a famous migrating pollinator",
        "si": "ප්‍රසිද්ධ සංක්‍රමණික පරාගණකාරයෙක්", "ta": "புகழ்பெற்ற வலசை மகரந்த சேர்க்கை"}),
    "lacewing": ("lacewing", {"en": "its larvae devour aphids and mites",
        "si": "එහි ලාර්වා අඵඩ් හා මයිට් භක්ෂණය කරයි", "ta": "அதன் லார்வா அஃபிட் மற்றும் மைட்டுகளை தின்னும்"}),
    "wasp": ("wasp", {"en": "often a predator of pest caterpillars",
        "si": "පළිබෝ කැටපිලන්ගේ විලෝපිකයෙක්", "ta": "பூச்சி புழுக்களின் வேட்டைக்காரன்"}),
    "weevil": ("weevil", {"en": "a beetle that bores into seeds, stems and roots",
        "si": "බීජ, කඳන් හා මූල තුළට කුහර කරන බෙටල් කෘමියෙක්", "ta": "விதை, தண்டு, வேர்களில் துளைக்கும் வண்டு"}),
    "leaf beetle": ("leaf beetle", {"en": "chews holes in foliage and can defoliate crops",
        "si": "කොළවල කුහර කපා බෝග නිර්පත්‍ර කළ හැක", "ta": "இலைகளில் ஓட்டை போட்டு பயிர்களை பாதிக்கும்"}),
    "beetle": ("beetle", {"en": "a huge group — some are pests, some are helpers",
        "si": "විශාල කණ්ඩායමක් — සමහරු පළිබෝදු, සමහරු උදව්කාරයන්", "ta": "பெரிய குழு — சில பூச்சிகள், சில உதவியாளர்கள்"}),
    "aphid": ("aphid", {"en": "sap-sucking insects that cluster on new growth",
        "si": "අලුරු වර්ධනයේ රොක් වී යුෂ උරන කෘමීන්", "ta": "புதிய வளர்ச்சியில் சாறை உறிஞ்சும் பூச்சிகள்"}),
    "grasshopper": ("grasshopper", {"en": "a chewing insect that can strip leaves",
        "si": "කොළ ඉවත් කළ හැකි චූෂක කෘමියෙක්", "ta": "இலைகளை உறிஞ்சும் பூச்சி"}),
    "locust": ("locust", {"en": "a swarming grasshopper — devastating in numbers",
        "si": "රංචු වශයෙන් එන මීයන් — විශාල සංඛ්‍යාවෙන් විනාශකාරී", "ta": "கூட்டமாக வரும் வெட்டுக்கிளி — எண்ணிக்கையில் அழிவு"}),
    "cricket": ("cricket", {"en": "mostly a nuisance; occasionally nibbles seedlings",
        "si": "බොහෝ විට කරදරයක්; සමහර විට පැළ තිබ්බ කරයි", "ta": "பெரும்பாலும் தொந்தரவு; சில நேரம் நாற்றுகளை கடிக்கும்"}),
    "cockroach": ("cockroach", {"en": "a household pest that spreads bacteria",
        "si": "බැක්ටීරියා පතුරුවන ගෘහ පළිබෝදුවෙක්", "ta": "பாக்டீரியாவை பரப்பும் வீட்டு பூச்சி"}),
    "cicada": ("cicada", {"en": "loud but mostly harmless to healthy plants",
        "si": "ඝෝෂකාරී නමුත් සෞඛ්‍ය සම්පන්න ශාකවලට හානිරහිත", "ta": "சத்தமானது ஆனால் ஆரோக்கிய தாவரங்களுக்கு தீங்கற்றது"}),
    "leafhopper": ("leafhopper", {"en": "sap-sucker that can spread plant disease",
        "si": "ශාක රෝග පතුරුවිය හැකි යුෂ උරන්නෙක්", "ta": "தாவர நோயை பரப்பும் சாற்று உறிஞ்சி"}),
    "caterpillar": ("caterpillar", {"en": "larval stage of butterflies/moths; many chew leaves",
        "si": "සමනලුන්ගේ ලාර්වා අදියර; බොහෝ අය කොළ කයි", "ta": "வண்ணான்களின் லார்வா; பல இலைகளை கடிக்கும்"}),
    "spider": ("spider", {"en": "not an insect, but a helpful predator of pests",
        "si": "කෘමියෙක් නොවේ, නමුත් පළිබෝදුන්ගේ ප්‍රයෝජනවත් විලෝපිකයෙක්", "ta": "பூச்சி அல்ல, ஆனால் பூச்சிகளின் பயனுள்ள வேட்டைக்காரன்"}),
    "fly": ("fly", {"en": "mostly a nuisance; some species pollinate",
        "si": "බොහෝ විට කරදරයක්; සමහර විශේෂ පරාගණය කරයි", "ta": "பெரும்பாலும் தொந்தரவு; சில இனங்கள் மகரந்த சேர்க்கை"}),
    "ant": ("ant", {"en": "usually harmless to crops; some farm aphids",
        "si": "බොහෝ විට බෝගවලට හානිරහිත; සමහරු අඵඩ් හදයි", "ta": "பெரும்பாலும் பயிர்களுக்கு தீங்கற்றது; சில அஃபிட் வளர்க்கும்"}),
}

# ------------------------------------------------------- language strings --
_VERDICT = {
    "en": {CATEGORY_HARMFUL: "Harmful — the trap WILL activate on it",
           CATEGORY_BENEFICIAL: "Beneficial — the trap leaves it alone",
           CATEGORY_HARMLESS: "Harmless — the trap leaves it alone"},
    "si": {CATEGORY_HARMFUL: "හානිකර — උගුල ක්‍රියාත්මක වේ",
           CATEGORY_BENEFICIAL: "ප්‍රයෝජනවත් — උගුල නොසලකා හරී",
           CATEGORY_HARMLESS: "හානිරහිත — උගුල නොසලකා හරී"},
    "ta": {CATEGORY_HARMFUL: "தீங்கு — பொறி இயங்கும்",
           CATEGORY_BENEFICIAL: "பயனுள்ள — பொறி விட்டுவிடும்",
           CATEGORY_HARMLESS: "தீங்கற்ற — பொறி விட்டுவிடும்"},
}

_STATE = {
    "en": {"running": "running", "idle": "idle"},
    "si": {"running": "ක්‍රියාත්මකයි", "idle": "නිශ්චලයි"},
    "ta": {"running": "இயங்குகிறது", "idle": "ஓய்வில்"},
}

_T = {
    "en": {
        "greeting": "Hello! I'm your pest trap assistant. Ask me things like \"how many harmful pests?\", \"what's the latest detection?\", or \"is a ladybug good or bad?\"",
        "thanks": "You're welcome! Happy to help keep your crops safe.",
        "help": ("I can (1) report live trap stats and the latest detection, "
                 "(2) tell you whether an insect is Harmful, Beneficial or Harmless "
                 "and why, and (3) explain the trap. The trap works as: "
                 "Detect (camera) → Classify (AI) → Decide (category + confidence) → "
                 "Act (servo fires only for Harmful pests above the confidence threshold)."),
        "threshold": ("The confidence threshold is {t}. Detections below it are treated as "
                      "uncertain and never trigger the trap. Raise it to avoid false triggers, "
                      "lower it for a livelier demo (CONFIDENCE_THRESHOLD in config.py)."),
        "trap": ("The detection pipeline is {state} on the '{cam}' feed. The trap has "
                 "activated {act} time(s) so far. It fires only for Harmful pests above "
                 "the confidence threshold."),
        "stats": ("So far: {total} detections — {h} harmful, {b} beneficial, {hl} harmless. "
                  "Trap activated {act} time(s)."),
        "stats_harmful": "The trap has seen {n} harmful pest(s) out of {total} total detections.",
        "stats_beneficial": "The trap has seen {n} beneficial insect(s) out of {total} total detections.",
        "stats_harmless": "The trap has seen {n} harmless visitor(s) out of {total} total detections.",
        "stats_act": "The trap has activated {n} time(s).",
        "latest": "Latest detection: {label} — classified {cat} (logged {time} UTC).",
        "latest_none": "No detections yet — start detection or upload a photo and I'll report back.",
        "insect": "A {name} is {blurb}. Verdict: {verdict}.",
        "insect_q": "A {name} is {verdict}. It's {blurb}.",
        "fallback": ("I'm not sure about that one. Try asking about your trap stats "
                     "(\"how many harmful pests?\"), the latest detection, or whether an "
                     "insect is harmful — e.g. \"is a weevil bad?\""),
    },
    "si": {
        "greeting": "ආයුබෝවන්! මම ඔබේ පළිබෝ උගුල සහායකයා. \"හානිකර පළිබෝදු කීයක්ද?\", \"අලුත්ම හඳුනාගැනීම කුමක්ද?\", \"ලේඩිබග් හොඳද නරකද?\" වැනි ප්‍රශ්න අහන්න.",
        "thanks": "සාදරයෙන් පිළිගනිමි! ඔබේ බෝග ආරක්ෂා කිරීමට උදව් කිරීමට සතුටුයි.",
        "help": "මට (1) සජීවී උගුල සංඛ්‍යාලේඛන හා අලුත්ම හඳුනාගැනීම වාර්තා කළ හැක, (2) කෘමියෙක් හානිකර/ප්‍රයෝජනවත්/හානිරහිත දැයි කියා දිය හැක, (3) උගුල පැහැදිලි කළ හැක. උගුල: හඳුනාගැනීම (කැමරාව) → වර්ගීකරණය (AI) → තීරණය → ක්‍රියාව (හානිකර පළිබෝදු සඳහා පමණක් සර්වෝ ක්‍රියාත්මක වේ).",
        "threshold": "විශ්වාසනීයත්ව සීමාව {t}. ඊට අඩු හඳුනාගැනීම් අවිනිශ්චිත ලෙස සැලකේ, උගුල ක්‍රියාත්මක නොවේ. ව්‍යාජ ක්‍රියාත්මක වීම් අඩු කිරීමට වැඩි කරන්න.",
        "trap": "හඳුනාගැනීම {state} ('{cam}' ප්‍රවාහයේ). උගුල මේ දක්වා {act} වතාවක් ක්‍රියාත්මක විය. විශ්වාසනීයත්ව සීමාවට ඉහළ හානිකර පළිබෝදු සඳහා පමණක් ක්‍රියාත්මක වේ.",
        "stats": "මේ දක්වා: {total} හඳුනාගැනීම් — හානිකර {h}, ප්‍රයෝජනවත් {b}, හානිරහිත {hl}. උගුල {act} වතාවක් ක්‍රියාත්මක විය.",
        "stats_harmful": "උගුල මුළු {total} න් හානිකර පළිබෝදු {n}ක් දුටුවේය.",
        "stats_beneficial": "උගුල මුළු {total} න් ප්‍රයෝජනවත් කෘමීන් {n}ක් දුටුවේය.",
        "stats_harmless": "උගුල මුළු {total} න් හානිරහිත අමුත්තන් {n}ක් දුටුවේය.",
        "stats_act": "උගුල {n} වතාවක් ක්‍රියාත්මක විය.",
        "latest": "අලුත්ම හඳුනාගැනීම: {label} — {cat} ලෙස වර්ගීකරණය ({time} UTC).",
        "latest_none": "තවම හඳුනාගැනීම් නැත — හඳුනාගැනීම අරඹන්න හෝ ඡායාරූපයක් උඩිගත කරන්න.",
        "insect": "{name} යනු {blurb}. තීන්දුව: {verdict}.",
        "insect_q": "{name} යනු {verdict}. {blurb}.",
        "fallback": "ඒ ගැන මට විශ්වාස නැත. උගුල සංඛ්‍යාලේඛන, අලුත්ම හඳුනාගැනීම, හෝ කෘමියෙක් හානිකර දැයි අහන්න — උදා: \"වීවිල් නරකද?\"",
    },
    "ta": {
        "greeting": "வணக்கம்! நான் உங்கள் பூச்சி பொறி உதவியாளர். \"தீங்கு பூச்சிகள் எத்தனை?\", \"சமீபத்திய கண்டறிதல் என்ன?\", \"லேடிபக் நல்லதா கெட்டதா?\" போன்று கேளுங்கள்.",
        "thanks": "நல்வரவு! உங்கள் பயிர்களை பாதுகாக்க உதவ மகிழ்ச்சி.",
        "help": "என்னால் (1) நேரடி பொறி புள்ளிவிவரங்கள் மற்றும் சமீபத்திய கண்டறிதலை தெரிவிக்க முடியும், (2) பூச்சி தீங்கு/பயனுள்ள/தீங்கற்றதா என சொல்ல முடியும், (3) பொறியை விளக்க முடியும். பொறி: கண்டறிதல் (கேமரா) → வகைப்படுத்தல் (AI) → முடிவு → செயல் (தீங்கு பூச்சிகளுக்கு மட்டும் சர்வோ இயங்கும்).",
        "threshold": "நம்பிக்கை வரம்பு {t}. அதற்கு கீழே உள்ள கண்டறிதல்கள் நிச்சயமற்றவை; பொறி இயங்காது. தவறான இயக்கங்களை குறைக்க உயர்த்தவும்.",
        "trap": "கண்டறிதல் {state} ('{cam}' ஊட்டத்தில்). பொறி இதுவரை {act} முறை இயங்கியது. நம்பிக்கை வரம்புக்கு மேல் தீங்கு பூச்சிகளுக்கு மட்டும் இயங்கும்.",
        "stats": "இதுவரை: {total} கண்டறிதல்கள் — தீங்கு {h}, பயனுள்ள {b}, தீங்கற்ற {hl}. பொறி {act} முறை இயங்கியது.",
        "stats_harmful": "பொறி மொத்த {total} இல் {n} தீங்கு பூச்சிகளை கண்டது.",
        "stats_beneficial": "பொறி மொத்த {total} இல் {n} பயனுள்ள பூச்சிகளை கண்டது.",
        "stats_harmless": "பொறி மொத்த {total} இல் {n} தீங்கற்றவற்றை கண்டது.",
        "stats_act": "பொறி {n} முறை இயங்கியது.",
        "latest": "சமீபத்திய கண்டறிதல்: {label} — {cat} என வகைப்படுத்தப்பட்டது ({time} UTC).",
        "latest_none": "இன்னும் கண்டறிதல்கள் இல்லை — கண்டறிதலை தொடங்கவும் அல்லது புகைப்படத்தை பதிவேற்றவும்.",
        "insect": "{name} என்பது {blurb}. தீர்ப்பு: {verdict}.",
        "insect_q": "{name} என்பது {verdict}. {blurb}.",
        "fallback": "அது பற்றி எனக்கு தெரியவில்லை. பொறி புள்ளிவிவரங்கள், சமீபத்திய கண்டறிதல், அல்லது பூச்சி தீங்கா என கேளுங்கள் — எ.கா: \"வீவில் தீங்கா?\"",
    },
}


def _find_insect(text: str):
    """Return (term, name, blurb_dict) for the first known insect mentioned."""
    for term, (name, blurbs) in KNOWLEDGE.items():
        if re.search(rf"\b{re.escape(term)}\b", text):
            return term, name, blurbs
    return None


def _tr(lang: str, key: str, **kw) -> str:
    table = _T.get(lang, _T["en"])
    template = table.get(key, _T["en"][key])
    return template.format(**kw) if kw else template


# ------------------------------------------------------------------ intents --
def reply(question: str, ctx: dict, lang: str = "en") -> str:
    """Produce an assistant answer for `question` in `lang` using context `ctx`."""
    if lang not in _T:
        lang = "en"
    text = (question or "").strip().lower()
    stats = ctx.get("stats") or {}
    by_cat = stats.get("by_category", {})
    last = ctx.get("last_detection")
    pipe = ctx.get("pipeline") or {}

    if not text:
        return _tr(lang, "greeting")

    if re.search(r"\b(hi|hello|hey|good (morning|afternoon|evening))\b", text):
        return _tr(lang, "greeting")

    if re.search(r"\b(thanks|thank you|thx)\b", text):
        return _tr(lang, "thanks")

    if not _find_insect(text) and \
       re.search(r"\b(help|what can you do|how do you work|how does (it|this|the trap) work|"
                 r"explain|capabilities|what is this)\b", text):
        return _tr(lang, "help")

    if re.search(r"\b(threshold|confidence|sensitiv)\b", text):
        return _tr(lang, "threshold", t=ctx.get("threshold", 0.35))

    if re.search(r"\b(trap|servo|pipeline|running|activat|fire|fired)\b", text) and \
       re.search(r"\b(status|running|on|off|work|activat|fire|is it)\b", text):
        state = _STATE[lang]["running"] if pipe.get("running") else _STATE[lang]["idle"]
        return _tr(lang, "trap", state=state, cam=pipe.get("camera") or "no camera",
                   act=stats.get("trap_activations", 0))

    if re.search(r"\b(how many|count|total|number|stats|statistics)\b", text):
        total = stats.get("total", 0)
        if "harmful" in text:
            return _tr(lang, "stats_harmful", n=by_cat.get("Harmful", 0), total=total)
        if "beneficial" in text:
            return _tr(lang, "stats_beneficial", n=by_cat.get("Beneficial", 0), total=total)
        if "harmless" in text:
            return _tr(lang, "stats_harmless", n=by_cat.get("Harmless", 0), total=total)
        if re.search(r"\b(activat|trap|fire)\b", text):
            return _tr(lang, "stats_act", n=stats.get("trap_activations", 0))
        return _tr(lang, "stats", total=total, h=by_cat.get("Harmful", 0),
                   b=by_cat.get("Beneficial", 0), hl=by_cat.get("Harmless", 0),
                   act=stats.get("trap_activations", 0))

    if re.search(r"\b(latest|last|recent|just now|newest|current)\b", text):
        if not last:
            return _tr(lang, "latest_none")
        return _tr(lang, "latest", label=last.get("label"), cat=last.get("category"),
                   time=str(last.get("timestamp"))[:19].replace("T", " "))

    insect = _find_insect(text)
    if insect:
        term, name, blurbs = insect
        blurb = blurbs.get(lang, blurbs["en"])
        verdict = _VERDICT[lang][label_to_category(name)]
        if re.search(r"\b(good|bad|harmful|beneficial|harmless|safe|dangerous|pest|friend|"
                     r"should i|kill|trap it|is (a|an|the))\b", text):
            return _tr(lang, "insect_q", name=name, verdict=verdict, blurb=blurb)
        return _tr(lang, "insect", name=name, blurb=blurb, verdict=verdict)

    return _tr(lang, "fallback")
