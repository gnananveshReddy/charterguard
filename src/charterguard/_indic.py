"""Indic-language override lexicon (data only; rules are built in patterns.py).

Exact-phrase rules are brittle for Indic languages: word order is mostly
subject-object-verb, nouns inflect for case, and "all previous instructions"
can be phrased half a dozen ways. So each language is described by three word
lists and the rule is structural:

    QUALIFIER (previous / all / above / system)
      ... up to two tokens ...
    INSTRUCTION NOUN (stem; any case suffix)
      ... up to three tokens ...
    OVERRIDE VERB (explicit imperative forms)

in either order (OV and VO), so "ignore karo sabhi pichle instructions" and
"sabhi pichle instructions ignore karo" both match.

Verbs are listed as **explicit imperative forms** rather than stems. Negation
in these languages either changes the verb form (Telugu "విస్మరించవద్దు",
"don't ignore") or inserts a particle ("अनदेखा मत करें"), and neither contains
the affirmative imperative as a contiguous string. A per-language negation
lookahead covers the remaining "verb + particle" orders.

Vocabulary was written by a non-native speaker from standard forms and is
**seed coverage**: corrections and additions from native speakers are the
most valuable contribution this file can get. Paraphrase in any language still
needs the model tier.

Malayalam has two encodings of chillu letters (atomic U+0D7A-U+0D7F, or
consonant + virama + ZWJ). The normalizer strips ZWJ, so `variants()` adds the
virama spelling of every Malayalam word.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Lexicon:
    lang: str
    label: str
    qualifiers: tuple[str, ...]
    nouns: tuple[str, ...]
    verbs: tuple[str, ...]
    negations: tuple[str, ...] = ()


_ML_CHILLU = {"ൺ": "ണ്", "ൻ": "ന്", "ർ": "ര്",
              "ൽ": "ല്", "ൾ": "ള്", "ൿ": "ക്"}


def variants(lang: str, words: tuple[str, ...]) -> tuple[str, ...]:
    """Spelling variants the normalizer will not unify on its own."""
    if lang != "ml":
        return words
    out = list(words)
    for w in words:
        alt = "".join(_ML_CHILLU.get(c, c) for c in w)
        if alt != w:
            out.append(alt)
    return tuple(out)


OVERRIDES: tuple[Lexicon, ...] = (
    Lexicon(
        "hi", "Hindi",
        qualifiers=("पिछले", "पिछली", "पिछला", "पहले", "सभी", "सारे", "सारी", "ऊपर", "पूर्व", "सिस्टम"),
        nouns=("निर्देश", "आदेश", "नियम", "प्रॉम्प्ट", "प्रोम्प्ट", "हिदायत"),
        verbs=("अनदेखा करें", "अनदेखा करो", "अनदेखा कर दो", "अनदेखा कीजिए",
               "नज़रअंदाज़ करें", "नज़रअंदाज़ करो", "नज़रअंदाज़ कर दो", "नजरअंदाज करें", "नजरअंदाज करो",
               "नजरअंदाज कर दो", "उपेक्षा करें", "उपेक्षा करो", "अवहेलना करें", "अवहेलना करो",
               "भूल जाओ", "भूल जाएं", "भूल जाइए", "भूलो", "भूलें", "छोड़ दो", "छोड़ दें", "मत मानो", "न मानें"),
        negations=("मत", "न", "नहीं"),
    ),
    Lexicon(
        "hinglish", "Hinglish (romanized Hindi)",
        qualifiers=("pichle", "pichhle", "pichli", "pichhli", "pehle", "pahle", "sabhi", "saare", "sare",
                    "saari", "upar", "purane"),
        nouns=("instruction", "nirdesh", "niyam", "rule", "prompt", "hidayat", "aadesh", "adesh"),
        verbs=("ignore karo", "ignore karein", "ignore karen", "ignore kar do", "ignore kardo", "ignore kijiye",
               "ignore kariye", "ignore kro", "nazarandaz karo", "najarandaz karo", "bhool jao", "bhul jao",
               "bhoolo", "bhulo", "chhod do", "chod do", "chhodo", "chodo", "mat maano", "mat mano"),
        negations=("mat", "na", "nahi", "nahin", "mt"),
    ),
    Lexicon(
        "te", "Telugu",
        qualifiers=("మునుపటి", "ముందటి", "గత", "అన్ని", "పై", "ఇంతకుముందు", "సిస్టమ్"),
        nouns=("సూచన", "ఆదేశ", "నియమ", "ప్రాంప్ట్"),
        verbs=("విస్మరించండి", "విస్మరించు", "విస్మరించి", "పట్టించుకోకండి", "పట్టించుకోవద్దు",
               "మర్చిపోండి", "మర్చిపో", "వదిలేయండి", "వదిలేయి"),
    ),
    Lexicon(
        "ta", "Tamil",
        qualifiers=("முந்தைய", "முன்பு", "அனைத்து", "எல்லா", "மேலே", "மேலுள்ள", "சிஸ்டம்"),
        nouns=("வழிமுறை", "அறிவுறுத்தல்", "கட்டளை", "விதி", "ப்ராம்ப்ட்"),
        verbs=("புறக்கணிக்கவும்", "புறக்கணி", "புறக்கணியுங்கள்", "மறந்துவிடு", "மறந்துவிடுங்கள்",
               "மறந்து விடுங்கள்", "கைவிடு", "கைவிடுங்கள்"),
    ),
    Lexicon(
        "bn", "Bengali",
        qualifiers=("আগের", "পূর্ববর্তী", "পূর্বের", "সমস্ত", "সব", "সকল", "উপরের", "সিস্টেম"),
        nouns=("নির্দেশ", "নিয়ম", "আদেশ", "প্রম্পট"),
        verbs=("উপেক্ষা করুন", "উপেক্ষা করো", "উপেক্ষা কর", "অগ্রাহ্য করুন", "অগ্রাহ্য করো",
               "ভুলে যাও", "ভুলে যান"),
        negations=("না", "নয়"),
    ),
    Lexicon(
        "mr", "Marathi",
        qualifiers=("मागील", "आधीच्या", "पूर्वीच्या", "सर्व", "वरील", "सिस्टम"),
        nouns=("सूचना", "निर्देश", "नियम", "आदेश"),
        verbs=("दुर्लक्ष करा", "दुर्लक्षित करा", "दुर्लक्ष कर", "विसरा", "विसरून जा"),
        negations=("नका", "नको"),
    ),
    Lexicon(
        "kn", "Kannada",
        qualifiers=("ಹಿಂದಿನ", "ಎಲ್ಲಾ", "ಎಲ್ಲ", "ಮೇಲಿನ", "ಸಿಸ್ಟಮ್"),
        nouns=("ಸೂಚನೆ", "ನಿಯಮ", "ಆದೇಶ"),
        verbs=("ನಿರ್ಲಕ್ಷಿಸಿ", "ನಿರ್ಲಕ್ಷಿಸು", "ಮರೆತುಬಿಡಿ", "ಮರೆತುಬಿಡು", "ಕಡೆಗಣಿಸಿ"),
    ),
    Lexicon(
        "ml", "Malayalam",
        qualifiers=("മുമ്പത്തെ", "മുൻപത്തെ", "എല്ലാ", "മുകളിലെ", "സിസ്റ്റം"),
        nouns=("നിർദ്ദേശ", "നിർദേശ", "നിയമ", "ആജ്ഞ"),
        verbs=("അവഗണിക്കുക", "അവഗണിക്കൂ", "അവഗണിച്ച്", "മറക്കുക", "മറന്നേക്കൂ", "മറന്നുകളയുക"),
    ),
    Lexicon(
        "gu", "Gujarati",
        qualifiers=("અગાઉની", "અગાઉના", "પહેલાની", "પહેલાંની", "બધી", "બધા", "ઉપરની", "સિસ્ટમ"),
        nouns=("સૂચના", "નિયમ", "આદેશ"),
        verbs=("અવગણો", "અવગણના કરો", "ભૂલી જાઓ", "ભૂલી જા"),
        negations=("નહીં", "નહિ", "ના"),
    ),
    Lexicon(
        "pa", "Punjabi",
        qualifiers=("ਪਿਛਲੀਆਂ", "ਪਿਛਲੇ", "ਪਿਛਲੀ", "ਸਾਰੀਆਂ", "ਸਾਰੇ", "ਉੱਪਰ", "ਸਿਸਟਮ"),
        nouns=("ਹਦਾਇਤ", "ਨਿਰਦੇਸ਼", "ਨਿਯਮ", "ਹੁਕਮ"),
        verbs=("ਅਣਡਿੱਠ ਕਰੋ", "ਅਣਡਿੱਠ ਕਰ ਦਿਓ", "ਨਜ਼ਰਅੰਦਾਜ਼ ਕਰੋ", "ਭੁੱਲ ਜਾਓ", "ਭੁੱਲ ਜਾਉ"),
        negations=("ਨਾ", "ਨਹੀਂ"),
    ),
    Lexicon(
        "ur", "Urdu",
        qualifiers=("پچھلی", "پچھلے", "تمام", "سب", "اوپر", "سسٹم"),
        nouns=("ہدایا", "ہدایت", "احکام", "اصول", "پرامپٹ"),
        verbs=("نظر انداز کریں", "نظر انداز کرو", "نظرانداز کریں", "نظرانداز کرو", "بھول جاؤ",
               "بھول جائیں", "بھول جائیے"),
        negations=("مت", "نہ", "نہیں"),
    ),
)


@dataclass(frozen=True, slots=True)
class LeakLexicon:
    lang: str
    label: str
    owners: tuple[str, ...]
    objects: tuple[str, ...]
    verbs: tuple[str, ...]


LEAKS: tuple[LeakLexicon, ...] = (
    LeakLexicon(
        "hi", "Hindi",
        owners=("अपना", "अपने", "अपनी", "तुम्हारा", "तुम्हारे", "आपका", "आपके"),
        objects=("सिस्टम प्रॉम्प्ट", "सिस्टम प्रोम्प्ट", "सिस्टम संदेश", "सिस्टम निर्देश", "मूल निर्देश"),
        verbs=("दिखाओ", "दिखाएं", "दिखाइए", "बताओ", "बताएं", "बताइए", "लिखो", "प्रिंट करो"),
    ),
    LeakLexicon(
        "hinglish", "Hinglish (romanized Hindi)",
        owners=("apna", "apne", "apni", "tumhara", "tumhare", "aapka", "aapke"),
        objects=("system prompt", "system message", "hidden instructions", "prompt"),
        verbs=("batao", "bataiye", "bataye", "dikhao", "dikhaiye", "likho", "print karo", "share karo"),
    ),
)
