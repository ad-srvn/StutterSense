# Disfluency-aware endpointing research plan

## Hypothesis

A voice interface that observes clip-level disfluency probabilities may interrupt less often during blocks, repetitions, prolongations, and filled pauses than an endpointing system based only on silence. This is a hypothesis for a later experiment, not a conclusion supported by the current StutterSense results.

## Why SEP-28k is insufficient

SEP-28k provides short-clip annotations for stuttering events. Those labels do not identify turn boundaries, intent completion, interruption outcomes, or when a listener should respond. A clip classifier therefore cannot be evaluated as a streaming end-of-turn detector, and its outputs must not be presented as word timestamps or turn-end probabilities.

## Required Phase 2 data

Phase 2 requires consented conversational audio with timestamps for speaker turns, pauses, overlap, system responses, and disfluency spans. It should include varied speakers, acoustic settings, devices, speaking styles, and disfluency prevalence. Speaker and conversation identifiers must be verified so all related segments remain in one partition. Licensing and permitted model use must be documented before collection or training.

## Annotation protocol

Annotators should mark:

1. speech and nonspeech regions;
2. hesitation, block, prolongation, sound repetition, word repetition, and interjection spans;
3. syntactic or pragmatic completion of each user turn;
4. legitimate turn endings and intentional holds;
5. system responses that begin before the user is finished;
6. overlap, backchannels, noise, and unintelligible intervals.

At least two independent annotators should label each evaluation conversation, with adjudication and disagreement retained. Guidance must distinguish a disfluent pause from a completed turn and a benign overlap from a harmful interruption.

## Baselines and ablation

Compare fixed silence thresholds, a voice-activity-detection endpoint, and a learned turn-completion predictor. Evaluate the learned predictor both with and without the five StutterSense event probabilities while holding all other inputs, splits, and tuning procedures fixed. Thresholds and patience parameters must be selected on validation conversations only.

## Metrics

- premature-interruption rate;
- endpointing latency after legitimate turn completion;
- false-continuation rate;
- per-speaker and per-conversation robustness;
- error rates conditioned on each disfluency type;
- risk-coverage behavior and calibration of turn-completion confidence.

Report distributions and paired confidence intervals, not only means. Latency and interruption trade-offs should be shown together rather than reduced to one favorable operating point.

## Leakage and uncertainty controls

Use speaker- and conversation-disjoint splits based on verified identifiers. Keep all chunks from an original recording together, deduplicate audio before splitting, and prohibit test labels from model or threshold selection. Resample whole speakers or conversations for intervals. Analyze annotator disagreement, low-prevalence events, acoustic quality, and demographic coverage separately. If identity or annotation provenance cannot be verified, label the evaluation `NOT VERIFIED` and do not claim generalization.

Phase 1 does not create a Phase 2 dataset or train a turn-completion model.
