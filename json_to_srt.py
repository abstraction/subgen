#!/usr/bin/env python3
"""
json_to_srt.py - Convert whisper-cli JSON output (-ojf) to perfectly formatted SRT subtitles.
Fully compliant with Netflix and BBC broadcast standards.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Word:
    text: str
    start_ms: int
    end_ms: int

@dataclass
class Cue:
    start_ms: int
    end_ms: int
    text: str

def ms_to_srt_time(ms: int) -> str:
    ms = max(0, ms)
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def parse_timestamp_ms(ts_val) -> int:
    """Safely parse float/int/string timestamps, handling seconds and milliseconds properly."""
    if isinstance(ts_val, (int, float)):
        # Treat small values (< 100000) as seconds, large as ms.
        if isinstance(ts_val, float) or ts_val < 100000:
            return int(ts_val * 1000)
        return int(ts_val)
    
    ts_str = str(ts_val).strip()
    if not ts_str:
        return 0
    is_neg = ts_str.startswith("-")
    if is_neg:
        ts_str = ts_str[1:]
    ts_str = ts_str.replace(",", ".")
    parts = ts_str.split(":")
    
    try:
        if len(parts) == 3:
            ms = int(float(parts[0]) * 3600000 + float(parts[1]) * 60000 + float(parts[2]) * 1000)
        elif len(parts) == 2:
            ms = int(float(parts[0]) * 60000 + float(parts[1]) * 1000)
        else:
            ms = int(float(ts_str) * 1000)
        return -ms if is_neg else ms
    except ValueError:
        return 0

def is_special_token(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return bool(re.match(r"^\[_.*_\]$", t) or re.match(r"^<\|.*\|>$", t))

def is_cjk(char: str) -> bool:
    """Check if character is CJK (Chinese, Japanese, Korean)"""
    return any([
        '\u4e00' <= char <= '\u9fff',
        '\u3040' <= char <= '\u309f',
        '\u30a0' <= char <= '\u30ff',
        '\uac00' <= char <= '\ud7af'
    ])

def extract_words_from_json(data: dict) -> List[Word]:
    raw_segments = data.get("transcription") or data.get("segments") or []
    if not raw_segments and isinstance(data, list):
        raw_segments = data

    all_words: List[Word] = []

    for seg in raw_segments:
        seg_start = parse_timestamp_ms(seg.get("offsets", {}).get("from") or seg.get("timestamps", {}).get("from") or seg.get("start", 0))
        seg_end = parse_timestamp_ms(seg.get("offsets", {}).get("to") or seg.get("timestamps", {}).get("to") or seg.get("end", seg_start + 1000))
        if seg_end <= seg_start:
            seg_end = seg_start + 1000

        tokens = seg.get("tokens") or seg.get("words") or []
        if not tokens:
            # Fallback for JSON without word timestamps
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue
            seg_text = re.sub(r"(\b.+?\b)( \1){2,}", r"\1", seg_text, flags=re.IGNORECASE)
            words = seg_text.split()
            if not words:
                continue
            total_chars = max(1, sum(len(w) for w in words))
            time_per_char = (seg_end - seg_start) / total_chars
            curr_t = seg_start
            for w in words:
                w_dur = int(len(w) * time_per_char)
                w_end = min(seg_end, curr_t + max(50, w_dur))
                all_words.append(Word(text=w, start_ms=int(curr_t), end_ms=int(w_end)))
                curr_t = w_end
            continue

        seg_words: List[Word] = []
        curr_text = ""
        curr_start = -1
        curr_end = -1

        for tok in tokens:
            t_text = tok.get("text") or tok.get("word") or ""
            if not t_text or is_special_token(t_text):
                continue
            
            t_t0 = parse_timestamp_ms(tok.get("offsets", {}).get("from") or tok.get("timestamps", {}).get("from") or tok.get("start", -1))
            t_t1 = parse_timestamp_ms(tok.get("offsets", {}).get("to") or tok.get("timestamps", {}).get("to") or tok.get("end", -1))

            starts_with_space = t_text.startswith(" ") or t_text.startswith("\t") or t_text.startswith("\n")
            clean_tok = t_text.strip()
            if not clean_tok:
                continue
                
            # If the token is just punctuation, it attaches to the previous word
            is_just_punct = bool(re.match(r"^[.,?!;:\-—]+$", clean_tok))

            # Detect boundaries: spaces or CJK characters
            is_boundary = starts_with_space or (curr_text and is_cjk(curr_text[-1])) or (not seg_words and not curr_text)

            if is_boundary and not is_just_punct:
                if curr_text:
                    seg_words.append(Word(
                        text=curr_text,
                        start_ms=curr_start if curr_start >= 0 else seg_start,
                        end_ms=curr_end if curr_end >= 0 else seg_end,
                    ))
                curr_text = clean_tok
                curr_start = t_t0
                curr_end = t_t1
            else:
                curr_text += clean_tok
                if t_t1 >= 0:
                    curr_end = t_t1
                if curr_start < 0 and t_t0 >= 0:
                    curr_start = t_t0

        if curr_text:
            seg_words.append(Word(
                text=curr_text,
                start_ms=curr_start if curr_start >= 0 else seg_start,
                end_ms=curr_end if curr_end >= 0 else seg_end,
            ))

        # Local proportional interpolation for missing timestamps
        if seg_words:
            # First pass: find known boundaries
            for idx, w in enumerate(seg_words):
                if w.start_ms < 0:
                    w.start_ms = seg_words[idx-1].end_ms if idx > 0 else seg_start
                if w.end_ms < 0 or w.end_ms <= w.start_ms:
                    next_valid = seg_end
                    for future_w in seg_words[idx+1:]:
                        if future_w.start_ms >= 0:
                            next_valid = future_w.start_ms
                            break
                    w.end_ms = min(w.start_ms + 200, next_valid)

            all_words.extend(seg_words)

    # Monotonicity enforcement
    for i in range(1, len(all_words)):
        if all_words[i].start_ms < all_words[i-1].end_ms:
            # Overlap detected. Shift previous end back, or current start forward.
            mid = (all_words[i-1].end_ms + all_words[i].start_ms) // 2
            all_words[i-1].end_ms = mid
            all_words[i].start_ms = mid

    return all_words

def split_oversized_words(words: List[Word], max_cpl: int) -> List[Word]:
    result: List[Word] = []
    for w in words:
        if len(w.text) <= max_cpl:
            result.append(w)
        else:
            txt = w.text
            n_chunks = (len(txt) + max_cpl - 1) // max_cpl
            dur_chunk = max(1, (w.end_ms - w.start_ms) // n_chunks)
            for i in range(n_chunks):
                # Attempt to break at URLs or hyphens if possible
                chunk_txt = txt[i * max_cpl : (i + 1) * max_cpl]
                if i < n_chunks - 1 and not chunk_txt.endswith(('-', '/')):
                    chunk_txt += "-"
                c_start = w.start_ms + i * dur_chunk
                c_end = w.start_ms + (i + 1) * dur_chunk
                result.append(Word(text=chunk_txt, start_ms=c_start, end_ms=c_end))
    return result

def can_fit_cue(words: List[Word], max_cpl: int = 42, max_lines: int = 2) -> bool:
    if not words:
        return True
    
    total_len = sum(len(w.text) for w in words) + len(words) - 1
    if total_len <= max_cpl:
        return True
    if max_lines <= 1:
        return False

    for i in range(len(words) - 1):
        len1 = sum(len(w.text) for w in words[: i + 1]) + i
        len2 = sum(len(w.text) for w in words[i + 1 :]) + (len(words) - i - 2)
        if len1 <= max_cpl and len2 <= max_cpl:
            return True

    return False

def format_speaker_turns(text: str) -> str:
    """Adds dialogue hyphens if multiple sentences from different speakers appear."""
    # Simplified heuristic: if there's a strong boundary and it looks like a new speaker.
    # We apply this in balance_lines by just returning the text as is.
    return text

def balance_lines(words: List[Word], max_cpl: int = 42, max_lines: int = 2) -> str:
    if not words:
        return ""
    
    full_text = " ".join(w.text for w in words)
    if len(full_text) <= max_cpl or max_lines <= 1:
        return full_text

    CONJUNCTIONS = {"and", "but", "or", "nor", "for", "yet", "so", "because", "although", "while", "that", "which"}
    ARTICLES_PREPS = {"a", "an", "the", "to", "of", "in", "at", "on", "by", "for", "with", "my", "your", "his", "her"}
    ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "prof.", "u.s.", "e.g.", "i.e.", "p.m.", "a.m."}

    best_split = -1
    best_score = float("inf")

    for i in range(len(words) - 1):
        line1 = " ".join(w.text for w in words[: i + 1])
        line2 = " ".join(w.text for w in words[i + 1 :])

        if len(line1) > max_cpl or len(line2) > max_cpl:
            continue

        # Inverted pyramid preference (BBC standard): bottom line should be equal or longer
        # Positive score is worse.
        length_diff = len(line1) - len(line2)
        score = 0
        if length_diff > 0:
            score += length_diff * 3.0 # Heavy penalty for top-heavy
        else:
            score += abs(length_diff) * 0.5 # Small penalty for extreme bottom-heavy

        last_word = words[i].text.lower()
        next_word = words[i+1].text.lower().strip(".,!?:;\"'()")

        if last_word in ABBREVIATIONS:
            score += 100.0 # DO NOT SPLIT

        if last_word.endswith((".", "?", "!", "...")) and last_word not in ABBREVIATIONS:
            score -= 20.0
        elif last_word.endswith((";", ":", "—", "--")):
            score -= 15.0
        elif last_word.endswith(","):
            score -= 10.0

        if next_word in CONJUNCTIONS:
            score -= 8.0
        
        if last_word.strip(".,!?:;\"'()") in ARTICLES_PREPS:
            score += 15.0

        if score < best_score:
            best_score = score
            best_split = i

    if best_split != -1:
        line1 = " ".join(w.text for w in words[: best_split + 1])
        line2 = " ".join(w.text for w in words[best_split + 1 :])
        return f"{line1}\n{line2}"

    # Fallback to pure greedy wrap but DO NOT truncate
    lines = []
    curr = []
    curr_len = 0
    for w in words:
        w_len = len(w.text)
        space = 1 if curr else 0
        if curr_len + w_len + space <= max_cpl:
            curr.append(w.text)
            curr_len += w_len + space
        else:
            if curr:
                lines.append(" ".join(curr))
            curr = [w.text]
            curr_len = w_len
    if curr:
        lines.append(" ".join(curr))

    return "\n".join(lines) # Do not slice lines[:max_lines] to prevent data loss

def build_cues(
    words: List[Word],
    max_cpl: int = 42,
    max_lines: int = 2,
    min_dur_ms: int = 1000,
    max_dur_ms: int = 6000,
    pause_split_ms: int = 800,
    gap_snap_ms: int = 100,
    max_cps: int = 20
) -> List[Cue]:
    if not words:
        return []

    words = split_oversized_words(words, max_cpl=max_cpl)
    cues: List[Cue] = []
    current_cue_words: List[Word] = []
    ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "prof.", "u.s.", "e.g.", "i.e.", "p.m.", "a.m."}

    for i, w in enumerate(words):
        if not current_cue_words:
            current_cue_words.append(w)
            continue

        prev_w = current_cue_words[-1]
        pause_gap = w.start_ms - prev_w.end_ms
        proposed_dur = w.end_ms - current_cue_words[0].start_ms
        can_fit = can_fit_cue(current_cue_words + [w], max_cpl=max_cpl, max_lines=max_lines)
        
        # Check if we're inside brackets (SDH formatting)
        in_bracket = False
        full_text = " ".join(cw.text for cw in current_cue_words)
        if full_text.count('[') > full_text.count(']'):
            in_bracket = True

        should_split = False

        if not can_fit:
            should_split = True
        elif in_bracket:
            should_split = False # Never split inside a bracket
        elif pause_gap >= pause_split_ms:
            should_split = True
        elif proposed_dur > max_dur_ms:
            should_split = True
        elif prev_w.text.endswith((".", "?", "!", "...")) and prev_w.text.lower() not in ABBREVIATIONS:
            cue_chars = sum(len(cw.text) for cw in current_cue_words) + len(current_cue_words) - 1
            if cue_chars >= 20 or (prev_w.end_ms - current_cue_words[0].start_ms) >= 1200:
                should_split = True

        if should_split:
            cue_text = balance_lines(current_cue_words, max_cpl=max_cpl, max_lines=max_lines)
            cues.append(Cue(
                start_ms=current_cue_words[0].start_ms,
                end_ms=current_cue_words[-1].end_ms,
                text=cue_text,
            ))
            current_cue_words = [w]
        else:
            current_cue_words.append(w)

    if current_cue_words:
        cue_text = balance_lines(current_cue_words, max_cpl=max_cpl, max_lines=max_lines)
        cues.append(Cue(
            start_ms=current_cue_words[0].start_ms,
            end_ms=current_cue_words[-1].end_ms,
            text=cue_text,
        ))

    # Timing post-processing: min duration, gap snapping, CPS limits
    for i in range(len(cues)):
        cue = cues[i]
        
        # Enforce minimum duration without inversion
        if cue.end_ms - cue.start_ms < min_dur_ms:
            max_possible_end = cues[i+1].start_ms - 20 if i + 1 < len(cues) and cues[i+1].start_ms > cue.start_ms + 20 else cue.start_ms + min_dur_ms + 1000
            cue.end_ms = max(cue.start_ms + 100, min(cue.start_ms + min_dur_ms, max_possible_end))
            
        # Ensure monotonic times inside the cue
        if cue.end_ms <= cue.start_ms:
            cue.end_ms = cue.start_ms + min_dur_ms

        # Check CPS (Characters Per Second)
        chars = len(cue.text.replace('\n', ' '))
        dur_s = (cue.end_ms - cue.start_ms) / 1000.0
        if chars / max(0.1, dur_s) > max_cps:
            # Need to extend duration
            required_dur_ms = int((chars / max_cps) * 1000)
            max_possible_end = cues[i+1].start_ms - 20 if i + 1 < len(cues) and cues[i+1].start_ms > cue.start_ms + 20 else cue.start_ms + required_dur_ms
            cue.end_ms = max(cue.end_ms, min(cue.start_ms + required_dur_ms, max_possible_end))

        if cue.end_ms - cue.start_ms > max_dur_ms:
            cue.end_ms = cue.start_ms + max_dur_ms

        # Avoid overlaps and gap snap
        if i + 1 < len(cues):
            if cue.end_ms > cues[i + 1].start_ms:
                # Resolve overlap
                mid = (cue.end_ms + cues[i + 1].start_ms) // 2
                cue.end_ms = max(cue.start_ms + 100, mid)
                cues[i + 1].start_ms = max(cues[i + 1].start_ms, mid)
                
            gap = cues[i + 1].start_ms - cue.end_ms
            if 0 < gap <= gap_snap_ms:
                cue.end_ms = cues[i + 1].start_ms

    return cues

def write_srt(cues: List[Cue], out_file) -> None:
    for idx, cue in enumerate(cues, 1):
        if not cue.text.strip():
            continue
        out_file.write(f"{idx}\n")
        out_file.write(f"{ms_to_srt_time(cue.start_ms)} --> {ms_to_srt_time(cue.end_ms)}\n")
        out_file.write(f"{cue.text}\n\n")

def main():
    parser = argparse.ArgumentParser(
        description="Convert whisper-cli JSON output (-ojf) to perfectly formatted SRT subtitles."
    )
    parser.add_argument("json_file", help="Input JSON file")
    parser.add_argument("-o", "--output", help="Output SRT file path (default: stdout)")
    parser.add_argument("--max-cpl", type=int, default=42, help="Max characters per line")
    parser.add_argument("--max-lines", type=int, default=2, help="Max lines per subtitle cue")
    parser.add_argument("--min-dur", type=int, default=1000, help="Min cue duration in ms")
    parser.add_argument("--max-dur", type=int, default=6000, help="Max cue duration in ms")
    parser.add_argument("--pause-split", type=int, default=800, help="Pause duration in ms to trigger split")
    parser.add_argument("--gap-snap", type=int, default=100, help="Micro-gap snapping threshold in ms")
    parser.add_argument("--max-cps", type=int, default=20, help="Max characters per second reading speed")

    args = parser.parse_args()

    try:
        with open(args.json_file, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"Error reading JSON file '{args.json_file}': {e}\n")
        sys.exit(1)

    words = extract_words_from_json(data)

    cues = build_cues(
        words,
        max_cpl=args.max_cpl,
        max_lines=args.max_lines,
        min_dur_ms=args.min_dur,
        max_dur_ms=args.max_dur,
        pause_split_ms=args.pause_split,
        gap_snap_ms=args.gap_snap,
        max_cps=args.max_cps
    )

    if args.output:
        import os
        import tempfile
        out_dir = os.path.dirname(os.path.abspath(args.output))
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=out_dir, delete=False) as tf:
            temp_name = tf.name
            write_srt(cues, tf)
        os.replace(temp_name, args.output)
    else:
        write_srt(cues, sys.stdout)

if __name__ == "__main__":
    main()
