# Speaker attribution — reframe: meeting platform first, diarization as fallback

**Status:** design/next-phase. The sherpa-onnx CPU diarizer in this branch is the
*fallback*; this doc scopes the better primary path.

## The insight

Muesli is a **meeting** app. In a real Zoom/Teams/Meet call the platform already
knows *who is speaking* (per-participant streams, active-speaker events, attendee
names). Unsupervised diarization of a single **mixed** loopback stream — guessing
"Speaker 1/2" from muddy audio — is solving the hard version of a problem the
meeting platform solves trivially, and it is unreliable in practice:

- No single clustering threshold is right for all content (validated this session:
  fast overlapping F1 commentary needs a *high* threshold to avoid over-splitting;
  a 2-person podcast with similar voices *merges* at the same threshold). Over-
  split vs. merge is inherent to fixed-threshold unsupervised diarization.
- Embedding models vary wildly by content (a "better" English model did *worse*
  than the zh 3dspeaker model on real broadcast audio).

So: use the platform's speaker identity when it's a real meeting; treat audio
diarization as a best-effort fallback only when no platform data exists.

## Target model

| Mode | Trigger | Speaker attribution | Summary template |
|---|---|---|---|
| **Meeting** | `MeetingDetectionService` detects Zoom/Teams/Meet | Platform speaker identity (real names / streams). Diarization only if platform data is unavailable. | Meeting notes + action items (appropriate) |
| **Local / ad-hoc** | No meeting detected (podcast, memo, system audio) | Single speaker, or best-effort sherpa diarization | Summarize, but **not** the meeting-notes/action-items template (see summary-template split) |

## What already exists in the codebase (leverage, don't rebuild)

- `Services/MeetingDetectionService.cs` — `DetectPlatform()` already classifies
  **Zoom / Teams / Google Meet** by window title + process name + browser URL.
- `_activeSpeakerAliases` (MainWindow) — a full label→real-name mapping UI with
  persistence + export. The app is *already* built around named speakers; blind
  diarization was just feeding it bad labels. Platform names would feed it good ones.
- `worker/diarize_sherpa.py` — the CPU fallback (this branch).

## Feasibility of "per-speaker streams from the meeting" (the hard part)

Windows **WASAPI loopback captures a single MIXED stream** — not per-participant.
Real per-speaker data needs one of:

1. **Platform SDK / API**
   - Zoom Meeting SDK (raw audio per participant) — requires app registration, and
     the SDK's Windows-on-ARM support must be verified.
   - Microsoft Graph / Teams (meeting transcripts w/ speaker labels) — cloud, needs
     tenant admin consent; conflicts with Muesli's local-first stance.
   - Google Meet — no first-party raw-audio API; captions via unofficial routes only.
2. **Meeting bot** joins the call as a participant and receives separate streams.
   Heaviest; also not local-first.
3. **Active-speaker UI events** — read the platform's on-screen "who's talking"
   highlight via UI Automation / accessibility, and time-align to the transcript.
   Lightest and local, but brittle and platform-specific.
4. **Diarization (this branch)** — the universal fallback when none of the above
   are available. Keep it.

**Recommendation:** start with (3) active-speaker UI events for the detected
platform (local, no cloud, no SDK registration), time-aligned to segments, with
diarization (4) as the floor. Revisit SDKs (1) only if UI events prove too flaky.
This is a substantial new capability — its own effort, not part of the diarization
backend PR.

## Summary-template split (smaller, independent, immediately useful)

Separate from speaker attribution: the meeting-notes/action-items template
(`MeetingSummaryService`, "Standard Meeting Notes") is wrong for non-meeting audio
(a podcast got "Action Items"). Gate the template on mode: meeting → notes+actions;
local → a plain summary (or transcript-only) with no action-items scaffolding.
Local transcriptions should still be summarized — just not with the meeting shape.

## Sequencing

1. (done, this branch) sherpa CPU diarization fallback + arm64 wheel + packaging.
2. Summary-template split (meeting vs local) — small, high value.
3. Active-speaker UI-event attribution for detected platforms — the real win.
4. Platform SDKs only if (3) is insufficient.
