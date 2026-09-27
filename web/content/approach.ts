import type { EventClass } from '@/types/events'

export type Stage = {
	title: string
	kind: 'learned' | 'rule' | 'io'
	detail: string
}

export const pipeline: Stage[] = [
	{ title: 'MP4 video', kind: 'io', detail: '4K, 29.97 fps, fixed camera' },
	{ title: 'Frame sampling', kind: 'rule', detail: 'Decode once, analyse a subset of frames' },
	{ title: 'YOLO detector', kind: 'learned', detail: 'Fine-tuned, 8 classes incl. red/green signal heads' },
	{ title: 'ByteTrack', kind: 'rule', detail: 'Track ids and trajectories per road user' },
	{ title: 'Scene alignment', kind: 'rule', detail: 'SIFT + RANSAC homography maps scene.json onto the clip' },
	{ title: 'Event rules', kind: 'rule', detail: 'Per-class geometry and timing rules on tracks and zones' },
	{ title: 'Segment merging', kind: 'rule', detail: 'Merge fragments, drop blips, no same-class overlap' },
	{ title: 'Events', kind: 'io', detail: '[start_sec, end_sec, label]' },
]

export type ClassApproach = { status: 'rule' | 'planned'; how: string }

// Mirrors the modules in ../detections/. Update when a class gains a detector.
export const classApproach: Record<EventClass, ClassApproach> = {
	jaywalking: {
		status: 'rule',
		how: 'Person ground point inside the carriageway polygon and outside every crossing, refuge island and kerb waiting zone.',
	},
	failure_to_yield: {
		status: 'rule',
		how: 'A tracked vehicle traverses a crossing polygon while a person is inside the same crossing; the event spans its traversal.',
	},
	red_light: {
		status: 'rule',
		how: 'The estimated front edge of a vehicle crosses a stop line while the signal counts as red: a red head is detected, or no green head was seen in the last 25 processed frames.',
	},
	stop_line: {
		status: 'rule',
		how: 'A vehicle rests within 30 px of a stop line during red without entering the junction.',
	},
	solid_line_crossing: {
		status: 'rule',
		how: 'A box corner crosses a solid marking segment; the event ends when the vehicle is fully inside the target lane.',
	},
	illegal_turn: {
		status: 'rule',
		how: 'A vehicle enters the junction, leaves it and then crosses the south-east crossing, a turn the markings prohibit from that approach.',
	},
	congestion: {
		status: 'rule',
		how: 'At least 75% of vehicles in a lane-direction group move slower than 0.6 box heights per second for 2 s; closes after 1 s clear.',
	},
	accident: { status: 'planned', how: 'Not detected yet. Candidate: box overlap plus abrupt stop of both tracks.' },
	near_miss: { status: 'planned', how: 'Not detected yet. Candidate: time-to-collision below a threshold plus sharp deceleration.' },
	wrong_way: { status: 'planned', how: 'Not detected yet. Candidate: track heading against the lane direction from scene.json.' },
	illegal_u_turn: { status: 'planned', how: 'Not detected yet. Candidate: heading change above 150° inside a no-U-turn zone.' },
	stopped_vehicle: {
		status: 'planned',
		how: 'Not detected yet. Candidate: stationary track for 10 s or more outside signal queues.',
	},
	road_obstacle: { status: 'planned', how: 'Not detected yet. Needs a detector beyond the 8 road-user classes.' },
	fire_smoke: { status: 'planned', how: 'Not detected yet. Needs a smoke/fire classifier.' },
}
