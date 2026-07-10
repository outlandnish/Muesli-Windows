namespace Muesli.Windows.Services;

public static class TranscriptFormatter
{
    private const int ConsolidationGapThresholdMs = 2000;

    public static string Merge(
        List<TranscriptSegment> micSegments,
        List<TranscriptSegment> systemSegments,
        List<DiarizedSegment> diarizationSegments,
        DateTime meetingStart)
    {
        if (diarizationSegments is null || diarizationSegments.Count == 0)
        {
            // Fallback to legacy merge when diarization is unavailable
            return LegacyMerge(micSegments, systemSegments);
        }

        // 1. Map raw diarization speaker IDs → "Speaker 1", "Speaker 2" in first-appearance order
        var speakerLabelMap = new Dictionary<string, string>(StringComparer.Ordinal);
        int nextSpeakerNumber = 1;
        foreach (var seg in diarizationSegments.OrderBy(s => s.StartMs))
        {
            if (!speakerLabelMap.ContainsKey(seg.SpeakerId))
            {
                speakerLabelMap[seg.SpeakerId] = $"Speaker {nextSpeakerNumber++}";
            }
        }

        // 2. Assign speaker labels to system transcript segments by time overlap.
        //    Fine-grained ASR phrases occasionally fall in a brief gap between
        //    diarization turns (no speaker overlaps them); carry forward the previous
        //    phrase's speaker so those gaps join the surrounding turn instead of
        //    fragmenting the transcript with "Others". Falls back to the next matched
        //    speaker (then "Others") for a leading gap.
        var taggedSystem = new List<TaggedSegment>();
        string? carriedSpeaker = null;
        foreach (var seg in systemSegments.OrderBy(s => s.StartMs))
        {
            var matched = FindSpeakerByOverlap(seg, diarizationSegments, speakerLabelMap);
            if (matched is not null)
            {
                carriedSpeaker = matched;
            }
            taggedSystem.Add(new TaggedSegment(seg, matched ?? carriedSpeaker ?? "Others"));
        }
        // Backfill any leading segments that had no speaker yet (before the first match).
        var firstKnown = taggedSystem.FirstOrDefault(t => t.Speaker != "Others")?.Speaker;
        if (firstKnown is not null)
        {
            for (var i = 0; i < taggedSystem.Count && taggedSystem[i].Speaker == "Others"; i++)
            {
                taggedSystem[i] = taggedSystem[i] with { Speaker = firstKnown };
            }
        }

        // 3. Tag mic segments as "You" — but drop segments that are just system-audio
        //    bleed/echo (e.g. speakers leaking into an open mic while only a stream
        //    plays). Segments are word-level, so a single word carries too little
        //    context to judge echo reliably; instead test each mic word against a
        //    sliding CONTEXT WINDOW of its neighbouring mic words vs the overlapping
        //    system text. A word whose local neighbourhood is echoed in the system
        //    audio is bleed (already covered as an anonymous Speaker N), so drop it.
        var micOrdered = micSegments.OrderBy(s => s.StartMs).ToList();
        var taggedMic = new List<TaggedSegment>();
        for (var i = 0; i < micOrdered.Count; i++)
        {
            if (!IsMicWordBleed(micOrdered, i, systemSegments))
            {
                taggedMic.Add(new TaggedSegment(micOrdered[i], "You"));
            }
        }

        // 4. Sort chronologically
        var all = taggedMic.Concat(taggedSystem)
            .OrderBy(t => t.Segment.StartMs)
            .ToList();

        if (all.Count == 0)
        {
            return "";
        }

        // 5. Consolidate consecutive same-speaker segments (within 2s gap)
        var consolidated = Consolidate(all, ConsolidationGapThresholdMs);

        // 6. Format: [HH:mm:ss] Speaker: text
        return string.Join("\n", consolidated.Select(t =>
        {
            var timestamp = meetingStart.AddMilliseconds(t.Segment.StartMs);
            var text = t.Segment.Text.Trim();
            return $"[{timestamp:HH:mm:ss}] {t.Speaker}: {text}";
        }));
    }

    private static string LegacyMerge(List<TranscriptSegment> micSegments, List<TranscriptSegment> systemSegments)
    {
        var parts = new List<string>();

        var micText = string.Join(" ", micSegments.Select(s => s.Text.Trim()));
        if (!string.IsNullOrWhiteSpace(micText))
        {
            parts.Add($"[You] {micText.Trim()}");
        }

        var systemText = string.Join(" ", systemSegments.Select(s => s.Text.Trim()));
        if (!string.IsNullOrWhiteSpace(systemText))
        {
            parts.Add($"[System audio] {systemText.Trim()}");
        }

        return string.Join(Environment.NewLine + Environment.NewLine, parts);
    }

    private static string? FindSpeakerByOverlap(
        TranscriptSegment transcriptSegment,
        List<DiarizedSegment> diarizationSegments,
        Dictionary<string, string> speakerLabelMap)
    {
        // Find diarization segment with maximum time overlap
        DiarizedSegment? bestMatch = null;
        double bestOverlap = 0;

        foreach (var diarSeg in diarizationSegments)
        {
            var overlap = CalculateOverlap(
                transcriptSegment.StartMs, transcriptSegment.EndMs,
                diarSeg.StartMs, diarSeg.EndMs);

            if (overlap > bestOverlap)
            {
                bestOverlap = overlap;
                bestMatch = diarSeg;
            }
        }

        if (bestMatch is null || bestOverlap <= 0)
        {
            return null;
        }

        return speakerLabelMap.TryGetValue(bestMatch.SpeakerId, out var label)
            ? label
            : "Others";
    }

    private static double CalculateOverlap(int startA, int endA, int startB, int endB)
    {
        var overlapStart = Math.Max(startA, startB);
        var overlapEnd = Math.Min(endA, endB);
        return Math.Max(0, overlapEnd - overlapStart);
    }

    // A mic word is treated as system-audio bleed/echo (not the local speaker) when
    // its local neighbourhood of mic words is largely echoed in the time-overlapping
    // system text. Judged over a WINDOW of neighbouring mic words (not the single
    // word, which carries too little signal), so genuine local speech that only
    // happens to share a few common words with the stream is not discarded.
    private const double BleedTextSimilarity = 0.6;   // >=60% of the mic window echoed
    private const int BleedContextWords = 6;          // +/- neighbour mic words
    // System echo can arrive shifted vs the mic (playback + capture latency), so
    // widen the time window when gathering overlapping system text.
    private const int BleedTimeToleranceMs = 2000;

    private static bool IsMicWordBleed(
        List<TranscriptSegment> micOrdered,
        int index,
        List<TranscriptSegment> systemSegments)
    {
        // Context window of neighbouring mic words (this word +/- BleedContextWords).
        var lo = Math.Max(0, index - BleedContextWords);
        var hi = Math.Min(micOrdered.Count - 1, index + BleedContextWords);
        var micWindowText = string.Join(" ",
            micOrdered.Skip(lo).Take(hi - lo + 1).Select(s => s.Text));

        // System text overlapping this window's (tolerance-widened) time span.
        var windowStart = micOrdered[lo].StartMs - BleedTimeToleranceMs;
        var windowEnd = micOrdered[hi].EndMs + BleedTimeToleranceMs;
        var overlappingSystemText = string.Join(" ", systemSegments
            .Where(s => CalculateOverlap(windowStart, windowEnd, s.StartMs, s.EndMs) > 0)
            .Select(s => s.Text));
        if (string.IsNullOrWhiteSpace(overlappingSystemText))
        {
            return false;
        }
        // CONTAINMENT: fraction of the mic window's tokens echoed in the system text.
        return TextContainment(micWindowText, overlappingSystemText) >= BleedTextSimilarity;
    }

    // Fraction of `inner`'s word tokens that also appear in `outer`.
    private static double TextContainment(string inner, string outer)
    {
        var innerTokens = Tokenize(inner);
        if (innerTokens.Count == 0)
        {
            return 0;
        }
        var outerTokens = Tokenize(outer);
        var present = innerTokens.Count(outerTokens.Contains);
        return (double)present / innerTokens.Count;
    }

    private static HashSet<string> Tokenize(string text)
    {
        return text
            .ToLowerInvariant()
            .Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':' },
                StringSplitOptions.RemoveEmptyEntries)
            .ToHashSet(StringComparer.Ordinal);
    }

    private static List<TaggedSegment> Consolidate(List<TaggedSegment> segments, int gapThresholdMs)
    {
        var result = new List<TaggedSegment>();
        if (segments.Count == 0)
        {
            return result;
        }

        var currentSpeaker = segments[0].Speaker;
        var currentStartMs = segments[0].Segment.StartMs;
        var currentEndMs = segments[0].Segment.EndMs;
        var currentText = segments[0].Segment.Text;

        for (int i = 1; i < segments.Count; i++)
        {
            var seg = segments[i];
            var gap = Math.Max(0, seg.Segment.StartMs - currentEndMs);

            if (seg.Speaker == currentSpeaker && gap <= gapThresholdMs)
            {
                // Same speaker, temporally close — accumulate text
                currentText = AppendText(currentText, seg.Segment.Text, gap);
                currentEndMs = seg.Segment.EndMs;
            }
            else
            {
                // Different speaker or too far apart — flush current
                result.Add(new TaggedSegment(
                    new TranscriptSegment("", currentSpeaker, currentStartMs, currentEndMs, currentText),
                    currentSpeaker));

                currentSpeaker = seg.Speaker;
                currentStartMs = seg.Segment.StartMs;
                currentEndMs = seg.Segment.EndMs;
                currentText = seg.Segment.Text;
            }
        }

        // Flush final accumulated segment
        result.Add(new TaggedSegment(
            new TranscriptSegment("", currentSpeaker, currentStartMs, currentEndMs, currentText),
            currentSpeaker));

        return result;
    }

    private static string AppendText(string current, string next, int gapMs)
    {
        current = current.Trim();
        next = next.Trim();

        if (string.IsNullOrEmpty(current))
        {
            return next;
        }

        if (string.IsNullOrEmpty(next))
        {
            return current;
        }

        // If there's a significant gap (>1s), treat as separate sentences
        if (gapMs > 1000)
        {
            var separator = current.EndsWith('.') || current.EndsWith('!') || current.EndsWith('?')
                ? " "
                : ". ";
            return current + separator + next;
        }

        // Small gap — just space-separate
        return current + " " + next;
    }

    private sealed record TaggedSegment(TranscriptSegment Segment, string Speaker);
}
