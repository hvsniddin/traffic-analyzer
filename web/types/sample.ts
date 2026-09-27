import type { RiskPoint, TrafficEvent } from './events'

// Shape of public/samples/<id>.json, written by scripts/build_sample_data.py.
// Every section that the pipeline has not produced yet is null, never faked.

export const DETECTOR_CLASSES = [
	'car',
	'bus',
	'truck',
	'motorcycle',
	'bicycle',
	'person',
] as const

export type DetectorClass = (typeof DETECTOR_CLASSES)[number]

export type VideoMeta = {
	file: string
	width: number
	height: number
	fps: number
	duration_sec: number
	n_frames: number
	codec: string
	bitrate_mbps: number
	size_gb: number
}

export type EventSet = {
	/** Where the events came from, e.g. "predictions_samples.json". */
	source: string
	note?: string | null
	events: TrafficEvent[]
}

export type CountsRow = { t: number } & Record<DetectorClass, number>

export type SignalRow = { t: number; red: number; green: number }

export type SampleEda = {
	step_sec: number
	detector: { weights: string; imgsz: number; conf: number; input: string }
	/** [t_sec, mean luma 0–255] */
	brightness: [number, number][]
	counts: CountsRow[]
	signal: SignalRow[]
	/** Pixel-area share with motion, per sampled second: [t_sec, 0–1]. */
	motion: [number, number][]
	motion_heatmap_url: string | null
	/** Bottom-centre of every detection, drawn over the scene. */
	positions_url: string | null
	flow_url: string | null
}

export type Sample = {
	id: string
	video: VideoMeta
	poster_url: string | null
	preview_url: string | null
	annotated_url: string | null
	predictions: (EventSet & { risk: RiskPoint[] | null }) | null
	labels: EventSet | null
	eda: SampleEda | null
	failures: { start_sec: number; end_sec: number; note: string }[]
}
