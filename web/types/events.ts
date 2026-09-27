// Mirrors CLASSES in ../../solution.py. Keep the order identical.
export const EVENT_CLASSES = [
	'accident',
	'near_miss',
	'red_light',
	'wrong_way',
	'illegal_u_turn',
	'stopped_vehicle',
	'jaywalking',
	'failure_to_yield',
	'illegal_turn',
	'solid_line_crossing',
	'stop_line',
	'congestion',
	'road_obstacle',
	'fire_smoke',
] as const

export type EventClass = (typeof EVENT_CLASSES)[number]

/** [start_sec, end_sec, label], the exact shape solution.detect_events returns. */
export type TrafficEvent = [number, number, EventClass]

/** [t_sec, score] pairs written by run_submission.py from RiskEstimator.step. */
export type RiskPoint = [number, number]
