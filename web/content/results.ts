// Numbers from evaluate.py against annotations/ground_truth.json (our dev labels
// of the four sample clips). Final row set is predictions_samples.json.

export const finalScores = {
	scoreA: 0.112,
	micro: { '0.3': 0.29, '0.5': 0.21, '0.7': 0.113 },
	classAgnostic: { '0.3': 0.371, '0.5': 0.242, '0.7': 0.129 },
}

/** [class, F1@0.3, F1@0.5, F1@0.7, TP, FP, FN at IoU 0.5] */
export const perClass: [string, number, number, number, number, number, number][] = [
	['stop_line', 0.57, 0.57, 0.29, 2, 1, 2],
	['jaywalking', 0.53, 0.34, 0.15, 9, 12, 23],
	['solid_line_crossing', 0.14, 0.14, 0.14, 1, 5, 7],
	['failure_to_yield', 0.05, 0.05, 0.05, 1, 10, 26],
	['near_miss', 0, 0, 0, 0, 3, 1],
	['red_light', 0, 0, 0, 0, 0, 1],
	['illegal_turn', 0, 0, 0, 0, 0, 4],
	['illegal_u_turn', 0, 0, 0, 0, 0, 2],
	['congestion', 0, 0, 0, 0, 0, 1],
]

/** Each step adds to the previous one. Score A on the four dev clips. */
export const ablations: { step: string; scoreA: number; note?: string }[] = [
	{ step: 'Earlier pipeline: 0.5 FPS on CPU, 1.5 s merge gap', scoreA: 0.049 },
	{ step: '+ 8 s merge gap for jaywalking', scoreA: 0.062, note: 'on the later preview run, 12 s gave 0.094 and 16 s 0.082 against 0.104, so 8 s stays' },
	{ step: 'New rules at 2 FPS on every device (540p previews)', scoreA: 0.07 },
	{ step: '+ stop reporting wrong_way and stopped_vehicle', scoreA: 0.085, note: '43 and 10 false positives, 0 labelled events' },
	{ step: '+ resolution-scaled stop line, preview alignment, solid-line clearance, near-miss fix', scoreA: 0.104 },
	{ step: '+ drop segments shorter than 1 s', scoreA: 0.115, note: 'F1 at IoU 0.7: 0.069 → 0.078' },
	{ step: 'Final: same code on the 4K originals (predictions_samples.json)', scoreA: 0.112 },
]

/** Minimum segment length sweep on the preview run: [seconds, Score A, mean F1@0.7] */
export const minLengthSweep: [number, number, number][] = [
	[0.5, 0.104, 0.069],
	[1.0, 0.115, 0.078],
	[1.5, 0.117, 0.08],
	[2.0, 0.118, 0.077],
	[3.0, 0.119, 0.078],
]

/** What happened to each of the 80 labelled events (best same-class IoU). */
export const labelOutcomes = { matched: 13, boundary: 23, missed: 44 }

/** [class, matched (IoU ≥ 0.5), found with wrong boundaries, missed] */
export const outcomesByClass: [string, number, number, number][] = [
	['jaywalking', 9, 19, 4],
	['failure_to_yield', 1, 3, 23],
	['solid_line_crossing', 1, 1, 6],
	['stop_line', 2, 0, 2],
	['illegal_turn', 0, 0, 4],
	['illegal_u_turn', 0, 0, 2],
	['congestion', 0, 0, 1],
	['red_light', 0, 0, 1],
	['near_miss', 0, 0, 1],
]

/** Where the 31 unmatched predictions fall. */
export const falsePositives: [string, string, number][] = [
	['jaywalking', 'right class, boundaries off', 11],
	['failure_to_yield', 'right class, boundaries off', 3],
	['failure_to_yield', 'overlaps a labelled jaywalking', 3],
	['failure_to_yield', 'nothing labelled there', 3],
	['near_miss', 'overlaps a labelled jaywalking', 3],
	['solid_line_crossing', 'right class, boundaries off', 2],
	['other', 'overlaps a different labelled class', 5],
	['solid_line_crossing', 'nothing labelled there', 1],
]
