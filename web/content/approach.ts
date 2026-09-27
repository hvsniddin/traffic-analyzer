import type { EventClass } from '@/types/events'

export type Stage = {
	title: string
	kind: 'learned' | 'rule' | 'io'
	detail: string
}

export const pipeline: Stage[] = [
	{ title: 'MP4 video', kind: 'io', detail: '4K, 29.97 fps, fixed camera' },
	{ title: 'Frame sampling', kind: 'rule', detail: '2 FPS on every device, stops at 2.7× clip length' },
	{ title: 'YOLO detector', kind: 'learned', detail: 'Fine-tuned, 8 classes incl. red/green signal heads' },
	{ title: 'ByteTrack', kind: 'rule', detail: 'Track ids and trajectories per road user' },
	{ title: 'Scene alignment', kind: 'rule', detail: 'SIFT + RANSAC homography maps scene.json onto the clip' },
	{ title: 'Event rules', kind: 'rule', detail: 'Per-class geometry and timing rules on tracks and zones' },
	{ title: 'Segment merging', kind: 'rule', detail: 'Merge same-class gaps under 1.5 s (8 s for jaywalking)' },
	{ title: 'Events', kind: 'io', detail: '[start_sec, end_sec, label]' },
]

// 'computed' rules run but are not reported (SUPPRESSED_CLASSES in solution.py).
export type ClassApproach = { status: 'rule' | 'computed' | 'planned'; how: string }

// Mirrors the modules in ../detections/. Update when a class gains a detector.
export const classApproach: Record<EventClass, ClassApproach> = {
	jaywalking: {
		status: 'rule',
		how: 'A person’s foot point is inside the carriageway and outside every crossing, refuge island and kerb waiting zone. Crossings get a 0.5 m margin, scaled from the person’s box height. One person’s segments merge across tracking gaps under 8 s.',
	},
	failure_to_yield: {
		status: 'rule',
		how: 'A vehicle is on a crossing while a person on the same crossing stands in its path: up to 3 vehicle widths ahead along its heading, within its width sideways. Flagged until the vehicle leaves the crossing.',
	},
	red_light: {
		status: 'rule',
		how: 'The front edge of a vehicle crosses a stop line while the signal counts as red: a red head is detected, or no green head was seen in the last 25 processed frames. Confirmed once the rear clears the line; the event starts at the front crossing and ends when the vehicle leaves the junction. Stopping on the line cancels it.',
	},
	stop_line: {
		status: 'rule',
		how: 'During red, a vehicle’s road contact point stays within 20 px of a stop line, moving under 8 px/s, for at least 0.5 s. The event starts when it stopped and lasts while the signal is red.',
	},
	solid_line_crossing: {
		status: 'rule',
		how: 'The midpoint of a vehicle’s bottom edge moves across a solid marking segment. The event lasts until all four box corners are inside a lane other than the one it came from.',
	},
	illegal_turn: {
		status: 'rule',
		how: 'A vehicle first seen in an approach lane other than lane_2 enters the junction, leaves it and then reaches the south-east crossing, a turn the markings prohibit from those lanes. Flagged while it is on that crossing.',
	},
	near_miss: {
		status: 'rule',
		how: 'A vehicle brakes hard (speed roughly halves within 0.5 s) or swerves (heading turns more than about 45°) while another road user’s projected closest approach is under 0.85 box widths within 1.25 s. Pairs whose boxes overlap for two or more samples are dropped as possible contact.',
	},
	congestion: {
		status: 'computed',
		how: 'At least five slow vehicles (under 0.6 box heights per second) in a close chain within one lane direction, confirmed after 2 s and closed after 1 s clear. stopped_vehicle also uses it to recognise queues. Not reported: it fired on ordinary red-light queues (37 false positives, 1 labelled event).',
	},
	wrong_way: {
		status: 'computed',
		how: 'A vehicle’s ~1 s displacement points more than 120° away from its lane’s drawn direction for 1.5 s. The event starts when the opposing motion is first seen. Not reported: normal flow through lane_2 does not follow its drawn direction (43 false positives, 0 labelled events).',
	},
	stopped_vehicle: {
		status: 'computed',
		how: 'A vehicle’s road contact point stays still for 10 s or more in a lane or the junction. Stops in signal approach lanes during red or congestion, or next to another stopped vehicle, count as queues and are excluded. Not reported: 10 false positives, 0 labelled events.',
	},
	accident: {
		status: 'planned',
		how: 'Not detected yet. Part B’s time-to-collision score anticipates collisions; an accident event would add box contact plus an abrupt stop of both tracks.',
	},
	illegal_u_turn: { status: 'planned', how: 'Not detected yet. Candidate: heading change above 150° inside a no-U-turn zone.' },
	road_obstacle: { status: 'planned', how: 'Not detected yet. Needs a detector beyond the 8 road-user classes.' },
	fire_smoke: { status: 'planned', how: 'Not detected yet. Needs a smoke/fire classifier.' },
}
